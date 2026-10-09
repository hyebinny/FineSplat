"""Benchmark test-time optimization (TTO) on a checkpoint's Stage-1 tokens.

The checkpoint is loaded as in eval_passes (end-to-end checkpoints carry the fine-tuned
Stage 1), but Stage 2's output is not used: TTO starts from Stage 1's own fused tokens
(``tokens_base``, step 0) and optimizes them against the context views
(src.model.tto.optimize_tokens). Every lr in ``inference.tto.lrs`` restarts from the
same tokens, so one model load covers a whole sweep. Targets are then rendered at the
config's cameras, or after EPA with ``inference.pose_optimization``.

    <out_dir>/step0_tto<lr>[_s<k>]/000001_<scene>/{metrics.json, tto.json}, summary.csv

``inference.tto.steps`` [1, 5, 50, ...] scores the tokens after each step count from a
single max(steps)-step run; tto.json then holds the time to reach that step.

``inference.tto`` other keys override src.model.tto.default_conf (num_steps,
photometric, perceptual_weight, ...). ``inference.scene_range`` [lo, hi) runs a shard
and leaves the summaries to a final pass.

    python -m src.commands.eval_tto <eval_config> [out_dir] [inference.key=value ...]
"""
import json
import os
import sys
import time
from pathlib import Path

import torch
from easydict import EasyDict as edict
from omegaconf import OmegaConf

from src.commands.inference import AMP_DTYPE, build_dataloader, load_model
from src.commands.inference_finesplat import load_finesplat
from src.model.tto import optimize_tokens
from src.utils.eval_paths import load_eval_config, resolve_out_dir
from src.utils.metric_utils import export_results, summarize_evaluation


def main():
    config, out_arg = load_eval_config()
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = config.inference.use_tf32
    torch.backends.cudnn.allow_tf32 = config.inference.use_tf32
    amp = torch.autocast("cuda", enabled=config.inference.use_amp,
                         dtype=AMP_DTYPE[config.inference.amp_dtype])

    dataloader = build_dataloader(config)
    base = load_model(config, device)
    base.eval()
    model = load_finesplat(config, base, device)

    tto = OmegaConf.to_container(config.inference.tto)
    lrs = [float(x) for x in tto.pop("lrs")]
    # tto.steps: also score the tokens after each of these step counts, from one run of
    # max(steps) (constant lr, so identical to separate shorter runs).
    steps = [int(k) for k in tto.pop("steps", [])] or [None]
    num_input = config.data.num_input_frames
    out_dir = out_arg or Path(resolve_out_dir(config))
    key = lambda lr, k: f"step0_tto{lr:g}" + ("" if k is None else f"_s{k}")
    metric_dirs = {(lr, k): out_dir / key(lr, k) for lr in lrs for k in steps}
    for d in metric_dirs.values():
        os.makedirs(d, exist_ok=True)
    print(f"writing to {out_dir} (lrs {lrs}, tto {tto}, EPA {base.use_pose_optimization})", flush=True)

    lo, hi = config.inference.get("scene_range", None) or (0, None)
    with torch.no_grad(), amp:
        for idx, batch in enumerate(dataloader):
            if idx < lo or (hi is not None and idx >= hi):
                continue
            print(f"scene {idx + 1}/{len(dataloader)}", flush=True)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            inp = {k: v[:, :num_input] if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            tgt = {k: v[:, num_input:] if isinstance(v, torch.Tensor) else None for k, v in batch.items()}
            out = model(inp, tgt)
            cam, tokens = out["cameras"], out["tokens_base"]
            h, w = inp["image"].shape[-2:]
            scene = inp["scene_name"][0]

            def score(lr, k, tok, sec):
                g = base.appearance_expert.decode_from_tokens(
                    tok, cam.pred_i_fxfycxcy, cam.pred_i_c2w, cam.pred_t_c2w, h, w)
                if base.use_pose_optimization:
                    with torch.enable_grad():
                        render, _ = base.pose_optimization(
                            gaussians=g, t_c2w=cam.pred_t_c2w, t_fxfycxcy=cam.pred_t_fxfycxcy,
                            w=base.width, h=base.height, target_images=tgt["image"],
                        )
                else:
                    render = model.render_baseline_targets(g, cam)
                d = metric_dirs[(lr, k)]
                export_results(
                    edict(input=inp, target=tgt, render=render.float()), str(d),
                    compute_metrics=True, visualize=False,
                    metric_resolution=config.inference.get("metric_resolution"),
                    metric_downsample=config.inference.get("metric_downsample", "bilinear_aa"),
                    uid=idx + 1,
                )
                with open(d / f"{idx + 1:06d}_{scene}" / "tto.json", "w") as f:
                    json.dump({"lr": lr, "steps": k, "tto_seconds": sec, **tto}, f)

            for lr in lrs:
                conf = {**tto, "lr": lr}
                if steps == [None]:
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    tok = optimize_tokens(base, tokens, cam.pred_i_fxfycxcy, cam.pred_i_c2w,
                                          inp["image"], h, w, conf=conf)
                    torch.cuda.synchronize()
                    runs = {None: (tok, time.perf_counter() - t0)}
                else:
                    runs = optimize_tokens(base, tokens, cam.pred_i_fxfycxcy, cam.pred_i_c2w,
                                           inp["image"], h, w, conf=conf, snapshot_steps=steps)
                for k, (tok, sec) in runs.items():
                    score(lr, k, tok, sec)
        torch.cuda.empty_cache()

    if config.inference.get("scene_range", None):
        return
    for d in metric_dirs.values():
        print(f"\n=== {d.name}", flush=True)
        summarize_evaluation(str(d))


if __name__ == "__main__":
    main()
