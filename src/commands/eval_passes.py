"""Benchmark FineSplat, scored after every Stage-2 step (step 0 = Stage 1) in one forward per scene.

    python -m src.commands.eval_passes <eval_config> [out_dir] [inference.key=value ...]

    <out_dir>/step<i>/000001_<scene>/metrics.json, <out_dir>/step<i>/summary.csv

With ``inference.pose_optimization`` each step is scored after EPA realigns the targets to its Gaussians.
inference.eval_steps: steps to evaluate, e.g. [0, 4] (default: all).
inference.step_dir_fmt: per-step directory under out_dir (default "step{i}").
inference.scene_range: [lo, hi) shard; summaries are left to a final pass.
inference.vis_save: {root, subdir, tags: {step: tag}} writes target renders to <root>/<tag>/<subdir>/<scene>/<frame:03d>.png.
inference.ply_save: {root, tags: {step: name}} writes each step's Gaussians to <root>/<tag>/<scene>/point_cloud.ply + cameras.json.
"""
import json
import os
import sys
from pathlib import Path

import torch
import torchvision
from easydict import EasyDict as edict
from omegaconf import OmegaConf

from src.commands.inference import AMP_DTYPE, build_dataloader, load_model
from src.commands.inference_finesplat import load_finesplat
from src.utils.eval_paths import load_eval_config, resolve_out_dir
from src.utils.metric_utils import export_results, summarize_evaluation
from src.utils.ply_io import gaussians_to_ply, save_scene_cameras


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
    model.keep_step_gaussians = True
    num_steps = model.num_steps
    num_input = config.data.num_input_frames
    visualize_first_n = config.inference.get("visualize_first_n", 0)
    out_dir = out_arg or Path(resolve_out_dir(config))
    eval_steps = list(config.inference.get("eval_steps", None) or range(num_steps + 1))
    step_dir_fmt = config.inference.get("step_dir_fmt", "step{i}")
    step_dirs = [out_dir / step_dir_fmt.format(i=i) for i in eval_steps]
    for d in step_dirs:
        os.makedirs(d, exist_ok=True)
    print(f"writing to {out_dir}", flush=True)

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
            cam = out["cameras"]

            gaussians = [out["gaussians_base"]] + list(out["gaussians_all"])
            gaussians = [gaussians[i] for i in eval_steps]
            aligned_cams = [None] * len(gaussians)  # EPA-aligned target (c2w, fxfycxcy) per step
            if base.use_pose_optimization:
                renders = []
                for k, g in enumerate(gaussians):
                    with torch.enable_grad():
                        aligned, aligned_cams[k] = base.pose_optimization(
                            gaussians=g, t_c2w=cam.pred_t_c2w, t_fxfycxcy=cam.pred_t_fxfycxcy,
                            w=base.width, h=base.height, target_images=tgt["image"],
                        )
                    renders.append(aligned)
            else:
                frozen = model.render_baseline_targets(out["gaussians_base"], cam)
                renders = [frozen] + list(out["renders"])
                renders = [renders[i] for i in eval_steps]

            # frame numbers are the targets' indices in the benchmark json
            vis = config.inference.get("vis_save", None)
            if vis:
                frames = tgt["index"][0].reshape(-1).long().tolist()
                for it, r in zip(eval_steps, renders):
                    tag = vis["tags"].get(it)
                    if tag is None:
                        continue
                    vd = Path(vis["root"]) / tag / vis["subdir"] / inp["scene_name"][0]
                    os.makedirs(vd, exist_ok=True)
                    for vi, fi in enumerate(frames):
                        torchvision.utils.save_image(r[0, vi].float().clamp(0, 1), vd / f"{fi:03d}.png")

            # with EPA on, cameras.json also gets "target_epa": the aligned target cameras
            ply = config.inference.get("ply_save", None)
            if ply:
                for it, g, ref in zip(eval_steps, gaussians, aligned_cams):
                    tag = (ply.get("tags") or {}).get(it, f"step{it}")
                    pd = Path(ply["root"]) / tag / inp["scene_name"][0]
                    os.makedirs(pd, exist_ok=True)
                    gaussians_to_ply(pd / "point_cloud.ply", g)
                    save_scene_cameras(pd / "cameras.json", cam, inp["index"][0].cpu().numpy(),
                                       tgt["index"][0].cpu().numpy(), config.data.resize_h)
                    if ref is not None:
                        cj = json.loads((pd / "cameras.json").read_text())
                        cj["target_epa"] = {"frame_index": cj["target"]["frame_index"],
                                            "c2w": ref[0][0].detach().float().cpu().numpy().tolist(),
                                            "fxfycxcy": ref[1][0].detach().float().cpu().numpy().tolist()}
                        (pd / "cameras.json").write_text(json.dumps(cj))

            for d, r in zip(step_dirs, renders):
                export_results(
                    edict(input=inp, target=tgt, render=r.float()),
                    str(d),
                    compute_metrics=config.inference.get("compute_metrics", True),
                    visualize=idx < visualize_first_n,
                    metric_resolution=config.inference.get("metric_resolution"),
                    metric_downsample=config.inference.get("metric_downsample", "bilinear_aa"),
                    uid=idx + 1,
                )
        torch.cuda.empty_cache()

    if config.inference.get("scene_range", None):
        return
    for d in step_dirs:
        print(f"\n=== {d.name}", flush=True)
        summarize_evaluation(str(d))


if __name__ == "__main__":
    main()
