"""Evaluate FineSplat (Stage 1 + Stage 2): per-scene renders and metrics, plus a summary over the test set.
    python -m src.commands.inference_finesplat --config src/configs/DL3DV/eval/6v/posed.yaml
inference.checkpoint: {ckpt_path, use_ema, full_state}; an empty ckpt_path leaves the Stage-2
    head zero-initialised, i.e. the Stage-1-only control through the same decode path.
"""

import importlib
import os
import warnings

import torch
from easydict import EasyDict as edict
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.utils.checkpoints import checkpoint_state
from src.utils.init_utils import init_config
from src.utils.eval_paths import resolve_out_dir
from src.utils.ply_io import gaussians_to_ply, save_scene_cameras
from src.utils.metric_utils import export_results, summarize_evaluation

warnings.filterwarnings('ignore', category=FutureWarning)

AMP_DTYPE = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "tf32": torch.float32,
}


def build_dataloader(config: DictConfig) -> DataLoader:
    """Instantiate the inference dataset and wrap it in a DataLoader."""
    assert config.inference.batch_size_per_gpu == 1, \
        "Currently only batch size of 1 is supported for streaming inference."

    dataset_name = config.inference.get("dataset_name", "src.datasets.dataset.InferenceDataset")
    module_name, class_name = dataset_name.rsplit(".", 1)
    Dataset = importlib.import_module(module_name).__dict__[class_name]
    dataset = Dataset(config)

    return DataLoader(
        dataset,
        batch_size=config.inference.batch_size_per_gpu,
        shuffle=False,
        num_workers=config.inference.num_workers,
        prefetch_factor=config.inference.prefetch_factor,
        persistent_workers=True,
        pin_memory=False,
    )


def load_model(config: DictConfig, device: torch.device) -> torch.nn.Module:
    """Instantiate the model and load checkpoint weights."""
    module_name, class_name = config.model.class_name.rsplit(".", 1)
    ModelClass = importlib.import_module(module_name).__dict__[class_name]
    model = ModelClass(config).to(device)
    if config.inference.ckpt_path == "":
        print("No checkpoint path provided; the pretrained model weights will be used for inference.")
    else:
        print(model.load_ckpt(config.inference.ckpt_path)) 
    return model


def load_finesplat(config, base, device):
    """Build FineSplat on top of ``base`` and load ``inference.checkpoint``, if any."""
    from src.commands.train_finesplat import build_finesplat

    model, _ = build_finesplat(base, config)
    model = model.to(device)
    cconf = config.inference.get("checkpoint", {}) or {}
    ckpt_path = cconf.get("ckpt_path", "")
    if not ckpt_path:
        print("No Stage-2 checkpoint: the zero-init head makes this the Stage-1-only control.")
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # full_state keeps the model.* keys, so an end-to-end checkpoint's Stage 1 wins over inference.ckpt_path
        state = checkpoint_state(ckpt, cconf.get("use_ema", True))
        if not cconf.get("full_state", True):
            state = {k: v for k, v in state.items() if not k.startswith("model.")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        trained = [k for k in missing if not k.startswith("model.")]
        assert not trained, f"Stage-2 weights missing from checkpoint: {trained[:5]}"
        assert not unexpected, f"unexpected keys in checkpoint: {unexpected[:5]}"
        print(f"Loaded FineSplat from {ckpt_path} (step {ckpt.get('step', '?')})")
    model.eval()
    return model


def main() -> None:
    config = init_config()

    os.environ["OMP_NUM_THREADS"] = str(config.inference.get("num_threads", 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cuda.matmul.allow_tf32 = config.inference.use_tf32
    torch.backends.cudnn.allow_tf32 = config.inference.use_tf32

    amp_context = torch.autocast(
        enabled=config.inference.use_amp,
        device_type="cuda",
        dtype=AMP_DTYPE[config.inference.amp_dtype],
    )

    dataloader = build_dataloader(config)
    base = load_model(config, device)
    base.eval()
    model = load_finesplat(config, base, device)

    visualize_first_n = config.inference.get("visualize_first_n", 0)
    num_input = config.data.num_input_frames
    out_dir = resolve_out_dir(config)
    print(f"Running inference; saving results to: {out_dir}")

    with torch.no_grad(), amp_context:
        for idx, batch in enumerate(dataloader):
            print(f"Processing batch {idx + 1}/{len(dataloader)}", flush=True)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            input_data = {
                k: v[:, :num_input] if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            target_data = {
                k: v[:, num_input:] if isinstance(v, torch.Tensor) else None
                for k, v in batch.items()
            }

            out = model(input_data, target_data)
            if base.use_pose_optimization:
                cameras = out["cameras"]
                prev_grad_state = torch.is_grad_enabled()
                torch.set_grad_enabled(True)
                renderings, _ = base.pose_optimization(
                    gaussians=out["gaussians"],
                    t_c2w=cameras.pred_t_c2w,
                    t_fxfycxcy=cameras.pred_t_fxfycxcy,
                    w=base.width, h=base.height,
                    target_images=target_data["image"],
                )
                torch.set_grad_enabled(prev_grad_state)
            else:
                renderings = out["render"]

            # save_gaussians: one reference-layout .ply plus the normalized cameras per scene
            if config.inference.get("save_gaussians", False):
                scene_dir = os.path.join(out_dir, f"{idx + 1:06d}_{input_data['scene_name'][0]}")
                os.makedirs(scene_dir, exist_ok=True)
                n = gaussians_to_ply(os.path.join(scene_dir, "point_cloud.ply"), out["gaussians"])
                save_scene_cameras(os.path.join(scene_dir, "cameras.json"), out["cameras"],
                                   input_data["index"][0].cpu().numpy(),
                                   target_data["index"][0].cpu().numpy(), config.data.resize_h)
                print(f"  saved {n} Gaussians", flush=True)

            export_results(
                edict(input=input_data, target=target_data, render=renderings.float()),
                out_dir,
                compute_metrics=config.inference.get("compute_metrics"),
                visualize=idx < visualize_first_n,
                metric_resolution=config.inference.get("metric_resolution"),
                metric_downsample=config.inference.get("metric_downsample", "bilinear_aa"),
                uid=idx + 1,
            )
        torch.cuda.empty_cache()

    if config.inference.get("compute_metrics", False):
        summarize_evaluation(out_dir)


if __name__ == "__main__":
    main()
