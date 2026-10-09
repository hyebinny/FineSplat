"""Evaluate the PTv3 token refiner on the DL3DV benchmark protocol.

Same 140-scene protocol, metrics and outputs as ``src.commands.inference``, so the
numbers drop straight into the same table. Per scene:

    frozen 2Xplat -> tokens -> context render -> error map -> lifted error
      -> PTv3 refiner -> token residual -> refined Gaussians
      -> render target views, or EPA          (inference.pose_optimization)
      -> metrics / renderings

``inference.refiner.ckpt_path: ""`` leaves the head zero-initialised, so the run
reproduces frozen 2Xplat through the refiner's own fp32 decode path. That is the control
this evaluation should be read against - not the bf16 numbers from ``inference.py``, which
decode the Gaussians in a different precision.

    python -m src.commands.inference_refiner --config src/configs/refiner/eval/6v_GTpose_refiner.yaml
"""

import os
import warnings

import torch
from easydict import EasyDict as edict

from src.commands.inference import AMP_DTYPE, build_dataloader, load_model
from src.commands.train_refiner import _deep_merge
from src.model.ptv3_refiner import TwoXplatPTv3Refiner, default_conf
from src.utils.init_utils import init_config
from src.utils.eval_paths import resolve_out_dir
from src.utils.metric_utils import export_results, summarize_evaluation

warnings.filterwarnings('ignore', category=FutureWarning)


def load_refiner(config, base, device):
    """Build the refiner named by ``refiner_model.arch`` and load its checkpoint, if any."""
    from src.commands.train_refiner import build_refiner

    model, _ = build_refiner(base, config)
    model = model.to(device)
    ckpt_path = config.inference.get("refiner", {}).get("ckpt_path", "")
    if not ckpt_path:
        print("No refiner checkpoint: the zero-init head makes this the frozen-2Xplat control.")
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        rconf = config.inference.get("refiner", {})
        # use_ema: read the EMA weights an e2e checkpoint carries. full_state: keep the
        # model.* keys too (e2e trains the base, so the checkpoint's base must win over
        # inference.ckpt_path). Both default off: frozen-refiner checkpoints as before.
        state = ckpt.get("ema") if rconf.get("use_ema", False) else None
        if state is None:
            state = ckpt.get("refiner", ckpt)
        if not rconf.get("full_state", False):
            # The frozen base is saved alongside the refiner; it is already loaded.
            state = {k: v for k, v in state.items() if not k.startswith("model.")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        trained = [k for k in missing if not k.startswith("model.")]
        assert not trained, f"refiner weights missing from checkpoint: {trained[:5]}"
        assert not unexpected, f"unexpected keys in checkpoint: {unexpected[:5]}"
        print(f"Loaded refiner from {ckpt_path} (step {ckpt.get('step', '?')})")
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
    model = load_refiner(config, base, device)

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

            export_results(
                edict(input=input_data, target=target_data, render=renderings.float()),
                out_dir,
                compute_metrics=config.inference.get("compute_metrics"),
                visualize=idx < visualize_first_n,
                uid=idx + 1,
            )
        torch.cuda.empty_cache()

    if config.inference.get("compute_metrics", False):
        summarize_evaluation(out_dir)


if __name__ == "__main__":
    main()
