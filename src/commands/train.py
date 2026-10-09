import os
import time
import torch
import importlib
import torch.distributed as dist
from typing import Any
from rich import print
from copy import deepcopy
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP

from src.utils.init_utils import init_config, init_distributed, init_wandb_and_backup
from src.utils.training_utils import (
    create_optimizer, create_lr_scheduler, auto_resume_job, print_rank0,
    update_ema, requires_grad,
    save_checkpoint, log_to_console, log_to_wandb,
)
from src.datasets import get_train_data_loader


AMP_DTYPE = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "tf32": torch.float32,
}


# ---------------------------------------------------------------------------
# Model initialization
# ---------------------------------------------------------------------------

def load_model(config: DictConfig, device: torch.device) -> tuple[DDP, torch.nn.Module]:
    """Instantiate model + EMA, load pretrained weights, and wrap model in DDP."""
    module_path, class_name = config.model.class_name.rsplit(".", 1)
    ModelClass = importlib.import_module(module_path).__dict__[class_name]

    model = ModelClass(config).to(device)
    # load_msg = model.load_ckpt(config.model.mvp_weights_path)
    # print(load_msg)

    ema = deepcopy(model).to(device)
    requires_grad(ema, False)

    model = DDP(model, find_unused_parameters=True, device_ids=[device.index])
    return model, ema


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def set_epoch(data_loader: torch.utils.data.DataLoader, epoch: int) -> None:
    """Propagate epoch to sampler, batch_sampler, and dataset for reproducible shuffling."""
    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, "sampler") and hasattr(data_loader.sampler, "set_epoch"):
        data_loader.sampler.set_epoch(epoch)
    if hasattr(data_loader, "batch_sampler") and hasattr(data_loader.batch_sampler, "set_epoch"):
        data_loader.batch_sampler.set_epoch(epoch)


def build_input_target_dicts(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Move batch tensors to device and split into input-view and target-view dicts."""
    tensors = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    input_dict = {
        "image": tensors["input_image"],
        "fxfycxcy": tensors["input_fxfycxcy"],
        "c2w": tensors["input_c2w"],
        "image_indices": tensors["input_indices"],
    }
    target_dict = {
        "image": tensors["target_image"],
        "fxfycxcy": tensors["target_fxfycxcy"],
        "c2w": tensors["target_c2w"],
        "image_indices": tensors["target_indices"],
    }
    return input_dict, target_dict


# ---------------------------------------------------------------------------
# Training step utilities
# ---------------------------------------------------------------------------

def forward_step(
    model: DDP,
    input_dict: dict[str, torch.Tensor],
    target_dict: dict[str, torch.Tensor],
    amp_context: torch.autocast,
) -> Any:
    """Run one forward pass under the given AMP context."""
    with amp_context:
        return model(input_dict, target_dict)


def backward_step(
    model: DDP,
    loss: torch.Tensor,
    grad_accum_steps: int,
    should_sync_grads: bool,
) -> None:
    """Scale loss and backward; suppress AllReduce during accumulation steps."""
    scaled_loss = loss / grad_accum_steps
    if should_sync_grads:
        scaled_loss.backward()
    else:
        with model.no_sync():
            scaled_loss.backward()


def clip_and_step(
    optimizer: torch.optim.Optimizer,
    param_list: list[torch.nn.Parameter],
    clip_norm: float,
) -> float:
    """Clip gradients and step the optimizer. Returns the grad norm."""
    grad_norm = torch.nn.utils.clip_grad_norm_(param_list, max_norm=clip_norm).item()
    optimizer.step()
    return grad_norm


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def training_loop(
    training_cfg: DictConfig,
    model: DDP,
    ema: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    ddp_info: Any,
    amp_context: torch.autocast,
    start_step: int,
    cur_step: int,
    cur_param_step: int,
    total_forward_steps: int,
    grad_accum_steps: int,
    optim_params: list[torch.nn.Parameter],
) -> None:
    """Outer epoch loop; iterates until total_forward_steps is reached."""
    for epoch in range(1_000_000):
        set_epoch(data_loader, epoch)

        for batch_data in data_loader:
            if cur_step == total_forward_steps:
                return

            step_start_time = time.time()
            input_dict, target_dict = build_input_target_dicts(batch_data, ddp_info.device)
            model_output = forward_step(model, input_dict, target_dict, amp_context)

            # Sync gradients only on the final accumulation step so AllReduce
            # is not triggered on every micro-step.
            is_last_accum_step = (
                (cur_step + 1) % grad_accum_steps == 0
                or cur_step + 1 == total_forward_steps
            )
            backward_step(model, model_output.loss_metrics.loss, grad_accum_steps, is_last_accum_step)
            cur_step += 1

            grad_norm = 0.0
            if is_last_accum_step:
                loss_is_finite = not (
                    torch.isnan(model_output.loss_metrics.loss)
                    or torch.isinf(model_output.loss_metrics.loss)
                )
                if loss_is_finite:
                    grad_norm = clip_and_step(optimizer, optim_params, training_cfg.grad_clip_norm)
                    cur_param_step += 1
                else:
                    print_rank0(f"NaN or Inf loss at step {cur_step}, skipping optimizer step")
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                update_ema(ema, model.module)

            if ddp_info.is_main_process:
                loss_dict = {k: float(f"{v.item():.6f}") for k, v in model_output.loss_metrics.items()}
                iter_time = time.time() - step_start_time
                lr = optimizer.param_groups[0]["lr"]

                log_to_console(epoch, cur_step, cur_param_step, iter_time, lr, loss_dict,
                               training_cfg.print_every, start_step)
                log_to_wandb(cur_step, cur_param_step, iter_time, lr, grad_norm, loss_dict,
                             training_cfg.wandb_log_every, start_step)

                if cur_step % training_cfg.checkpoint_every == 0 or cur_step == total_forward_steps:
                    save_checkpoint(model, ema, optimizer, lr_scheduler,
                                    cur_step, cur_param_step, training_cfg.checkpoint_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Initialize distributed training, build all components, and run the training loop."""
    config = init_config()

    os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))

    ddp_info = init_distributed(seed=777)
    dist.barrier()

    if ddp_info.is_main_process:
        init_wandb_and_backup(config)
    dist.barrier()

    # Disable TF32 for numerical stability in distributed training
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    amp_context = torch.autocast(
        enabled=config.training.use_amp,
        device_type="cuda",
        dtype=AMP_DTYPE[config.training.amp_dtype],
    )

    data_loader = get_train_data_loader(
        config,
        num_workers=config.training.num_workers,
        shuffle=True,
        drop_last=True,
        pin_mem=True,
    )

    param_update_steps = config.training.train_steps
    grad_accum_steps = config.training.grad_accum_steps
    total_forward_steps = param_update_steps * grad_accum_steps

    model, ema = load_model(config, ddp_info.device)

    optimizer, optimized_param_dict, _ = create_optimizer(
        model,
        config.training.weight_decay,
        config.training.lr,
        (config.training.beta1, config.training.beta2),
    )
    optim_params = list(optimized_param_dict.values())

    scheduler_type = config.training.get("scheduler_type", "cosine")
    lr_scheduler = create_lr_scheduler(
        optimizer, param_update_steps, config.training.warmup, scheduler_type=scheduler_type
    )

    ckpt_load_path = config.training.get("resume_ckpt", "") or config.training.checkpoint_dir
    reset_training_state = config.training.get("reset_training_state", False)
    optimizer, lr_scheduler, cur_step, cur_param_step = auto_resume_job(
        ckpt_load_path, model, optimizer, lr_scheduler, reset_training_state
    )
    dist.barrier()

    start_step = cur_step
    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()

    training_loop(
        config.training, model, ema, data_loader, optimizer, lr_scheduler,
        ddp_info, amp_context, start_step, cur_step, cur_param_step,
        total_forward_steps, grad_accum_steps, optim_params,
    )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
