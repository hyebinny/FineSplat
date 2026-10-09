import os
import wandb
import torch
import traceback
from collections import OrderedDict
from rich import print
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)


# ---------------------------------------------------------------------------
# General-purpose utilities
# ---------------------------------------------------------------------------

def print_rank0(*args, **kwargs) -> None:
    """Print only from rank-0 in a distributed setting, or unconditionally otherwise."""
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)


def format_number(num: int | float) -> str:
    """Format a large number as a human-readable string with B/M/K suffixes.

    Args:
        num (int | float): The number to format.

    Returns:
        str: Human-readable representation (e.g. '1.50B', '3.20M', '512.00K').
    """
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    return str(num)


# ---------------------------------------------------------------------------
# Optimizer and learning rate scheduler creation
# ---------------------------------------------------------------------------

def create_optimizer(
    model: torch.nn.Module,
    weight_decay: float,
    learning_rate: float,
    betas: tuple[float, float],
) -> tuple[torch.optim.AdamW, dict[str, torch.nn.Parameter], dict[str, torch.nn.Parameter]]:
    """Build an AdamW optimizer with separate weight-decay groups for the model.

    1D parameters (biases, norms) and any parameter flagged with
    ``_no_weight_decay`` are placed in a zero-decay group; all others receive
    the specified weight decay.

    Args:
        model (torch.nn.Module): The model whose parameters will be optimized.
        weight_decay (float): Weight decay applied to multi-dimensional parameters.
        learning_rate (float): Base learning rate.
        betas (tuple[float, float]): AdamW beta coefficients.

    Returns:
        tuple: (optimizer, optimized_param_dict, all_param_dict) where
            optimized_param_dict contains only trainable parameters and
            all_param_dict contains every named parameter.
    """
    all_param_dict = {name: param for name, param in model.named_parameters()}
    optimized_param_dict = {name: param for name, param in all_param_dict.items() if param.requires_grad}

    decay_params, nodecay_params = [], []
    for name, param in optimized_param_dict.items():
        if param.dim() == 1 or getattr(param, '_no_weight_decay', False):
            nodecay_params.append(param)
        else:
            decay_params.append(param)
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

    if dist.is_initialized():
        if dist.get_rank() == 0:
            def get_module_name(name):
                """Returns a two-level module prefix from a dotted parameter name."""
                parts = name.split('.')
                if len(parts) > 2 and parts[0] == 'module':
                    return parts[1] + '.' + parts[2]
                return parts[0]  # Fallback to first part if no 'module.' prefix
            print(f'Optimizer: AdamW, learning rate: {learning_rate}, weight decay: {weight_decay}, betas: {betas}')
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in optimized_param_dict.values())
            optim_module_names = sorted(set(get_module_name(name) for name in optimized_param_dict.keys()))
            frozen_module_names = sorted(set(get_module_name(name) for name in set(all_param_dict.keys()) - set(optimized_param_dict.keys())))

            print(f'Total parameters: {format_number(total_params)}, Trainable parameters: {format_number(trainable_params)}')
            print(f'Optimized parameters: {optim_module_names}')
            print(f'Frozen parameters: {frozen_module_names}')

    return optimizer, optimized_param_dict, all_param_dict


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    param_update_steps: int,
    warm_up_steps: int,
    scheduler_type: str = 'cosine',
) -> torch.optim.lr_scheduler.LRScheduler:
    """Create a learning rate scheduler with linear warmup.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to schedule.
        param_update_steps (int): Total number of parameter update steps.
        warm_up_steps (int): Number of warmup steps at the start of training.
        scheduler_type (str): One of 'linear', 'cosine', or 'constant'.

    Returns:
        torch.optim.lr_scheduler.LRScheduler: The configured scheduler.

    Raises:
        ValueError: If scheduler_type is not one of the supported values.
    """
    if scheduler_type == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, warm_up_steps, param_update_steps)
    elif scheduler_type == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(optimizer, warm_up_steps, param_update_steps)
    elif scheduler_type == 'constant':
        scheduler = get_constant_schedule_with_warmup(optimizer, warm_up_steps)
    else:
        raise ValueError(f'Invalid scheduler type: {scheduler_type}')
    return scheduler


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def find_checkpoints(load_path: str) -> list[str]:
    """Return sorted checkpoint paths found at load_path.

    Args:
        load_path (str): Either a directory containing .pt files, or a direct
            path to a single .pt file.

    Returns:
        list[str]: Sorted list of absolute checkpoint file paths.
    """
    if os.path.isdir(load_path):
        ckpt_names = [file_name for file_name in os.listdir(load_path) if file_name.endswith(".pt")]
        ckpt_names = sorted(ckpt_names, key=lambda x: x)
        ckpt_paths = [os.path.join(load_path, ckpt_name) for ckpt_name in ckpt_names]
    else:
        if load_path.endswith(".pt"):
            ckpt_paths = [load_path]
        else:
            ckpt_paths = []
    return ckpt_paths


def auto_resume_job(
    load_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    reset_training_state: bool,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, int, int]:
    """Resume training from the latest checkpoint in the specified directory.

    Args:
        load_path (str): If a directory, loads the last checkpoint in it;
            otherwise treats the path as a direct checkpoint file.
        model (torch.nn.Module): Model whose weights will be loaded.
        optimizer (torch.optim.Optimizer): Optimizer to restore.
        lr_scheduler (torch.optim.lr_scheduler.LRScheduler): Scheduler to restore.
        reset_training_state (bool): If True, only model weights are restored and
            optimizer/scheduler state is left at initialization.

    Returns:
        tuple: (optimizer, lr_scheduler, forward_pass_step, param_update_step).
    """
    forward_pass_step = 0
    param_update_step = 0
    all_ckpt_paths = find_checkpoints(load_path)
    if len(all_ckpt_paths) == 0:
        print_rank0(f"No checkpoint found in {load_path}, we will start from scratch")
        return optimizer, lr_scheduler, forward_pass_step, param_update_step
    try:
        ckpt_path = all_ckpt_paths[-1]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    except:
        traceback.print_exc()
        print_rank0(f"Failed to load {ckpt_path}, we will start from scratch")
        return optimizer, lr_scheduler, forward_pass_step, param_update_step

    if isinstance(model, DDP):
        status = model.module.load_state_dict(checkpoint['model'], strict=False)
    else:
        status = model.load_state_dict(checkpoint['model'], strict=False)
    print_rank0(f"Loaded model from {os.path.abspath(ckpt_path)}, the status is {status}")

    if not reset_training_state:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            forward_pass_step = checkpoint["fwdbwd_pass_step"]
            param_update_step = checkpoint["param_update_step"]
            print_rank0(f"Resumed optimizer and lr_scheduler from {ckpt_path}")
        except:
            traceback.print_exc()
            print_rank0(f"Failed to load optimizer and lr_scheduler from {ckpt_path}")

    return optimizer, lr_scheduler, forward_pass_step, param_update_step


def save_checkpoint(
    model: torch.nn.Module,
    ema: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    cur_step: int,
    param_step: int,
    checkpoint_dir: str,
) -> None:
    """Serialize model/EMA/optimizer/scheduler to a timestamped checkpoint file."""
    checkpoint = {
        "model": strip_module_prefix(model.state_dict()),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "fwdbwd_pass_step": cur_step,
        "param_update_step": param_step,
    }
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"ckpt_{cur_step:016}.pt")
    torch.save(checkpoint, ckpt_path)
    print(f"Saved checkpoint at step {cur_step} to {os.path.abspath(ckpt_path)}")


# ---------------------------------------------------------------------------
# Model parameter utilities
# ---------------------------------------------------------------------------

@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float = 0.999) -> None:
    """Step the EMA model towards the current model."""
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model: torch.nn.Module, flag: bool = True) -> None:
    """Set requires_grad on all parameters of a model.

    Args:
        model (torch.nn.Module): Model to modify.
        flag (bool): Value to assign to requires_grad on every parameter.
    """
    for p in model.parameters():
        p.requires_grad = flag


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove DDP/FSDP/torch.compile key prefixes from a state dict.

    Strips '_checkpoint_wrapped_module.', '_orig_mod.', and any number of
    leading 'module.' prefixes so that weights can be loaded into an unwrapped
    model.

    Args:
        state_dict (dict[str, torch.Tensor]): The raw state dict to clean.

    Returns:
        dict[str, torch.Tensor]: State dict with prefixes removed.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        key = key.replace("_checkpoint_wrapped_module.", "")
        key = key.replace("_orig_mod.", "")
        while key.startswith("module."):
            key = key[len("module."):]
        new_state_dict[key] = value
    return new_state_dict


# ---------------------------------------------------------------------------
# Logging utilities
# ---------------------------------------------------------------------------

def log_to_console(
    epoch: int,
    cur_step: int,
    param_step: int,
    iter_time: float,
    lr: float,
    loss_dict: dict[str, float],
    print_every: int,
    start_step: int,
) -> None:
    """Print training progress when the logging criteria are met."""
    if cur_step % print_every != 0 and cur_step >= start_step + 100:
        return
    loss_str = " | ".join(f"{k}: {v:.6f}" for k, v in loss_dict.items())
    print(
        f"[Epoch {epoch:>3d}] | "
        f"Forward step: {cur_step:>6d} (Param update step: {param_step:>6d}) | "
        f"Iter time: {iter_time:.2f}s | LR: {lr:.6f}\n"
        + loss_str
    )


def log_to_wandb(
    cur_step: int,
    param_step: int,
    iter_time: float,
    lr: float,
    grad_norm: float,
    loss_dict: dict[str, float],
    wandb_log_every: int,
    start_step: int,
) -> None:
    """Log metrics to W&B when the logging criteria are met."""
    if cur_step % wandb_log_every != 0 and cur_step >= start_step + 200:
        return
    log_dict = {
        "iter": cur_step,
        "forward_pass_step": cur_step,
        "param_update_step": param_step,
        "lr": lr,
        "iter_time": iter_time,
        "grad_norm": grad_norm,
    }
    log_dict.update({"train/" + k: v for k, v in loss_dict.items()})
    wandb.log(log_dict, step=cur_step)
