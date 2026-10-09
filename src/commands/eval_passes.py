"""Benchmark a multi-iteration refiner, scored after every iteration.

One forward already renders the target views after each refinement iteration, so a single
sweep over the benchmark yields iter 1..N together - no need for N runs. Iteration 0 is the
frozen 2Xplat prediction the refiner starts from, rendered here too so that every delta is
measured inside one run, under the same preprocessing, poses and seed.

Results follow 2Xplat's own evaluation layout - one directory per scene holding
metrics.json, then summary.csv and average_metrics.txt beside them - with the iteration as
one extra level:

    <out_dir>/iter0/000001_<scene>/metrics.json
    <out_dir>/iter0/summary.csv
    <out_dir>/iter4/...

so a frozen baseline directory and one iteration of a refiner run are the same shape and
read by the same tools.

With ``inference.pose_optimization`` the target poses are DA3's, so every iteration is
scored after EPA realigns them against that iteration's own Gaussians - the unposed
protocol, applied per iteration rather than to the last one only. That costs one alignment
per iteration, iteration 0 included.

    python -m src.commands.eval_passes <eval_config> [out_dir] [inference.key=value ...]

Without an out_dir the config's inference.run files the results under experiments/eval by
protocol; see src/utils/eval_paths.py.

The eval config supplies data/inference sections (json_path, num_input_frames, ckpt) and a
refiner_model block matching the checkpoint's training config.
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
from src.commands.inference_refiner import load_refiner
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
    model = load_refiner(config, base, device)
    # iteration 0 needs the frozen Gaussians, and EPA needs every iteration's separately
    model.keep_iter_gaussians = True
    num_iters = model.num_iters
    num_input = config.data.num_input_frames
    visualize_first_n = config.inference.get("visualize_first_n", 0)
    out_dir = out_arg or Path(resolve_out_dir(config))
    # inference.eval_iters: evaluate only these iterations (e.g. [0, 4]); EPA is the
    # expensive part and runs once per evaluated iteration. Default: all of them.
    eval_iters = list(config.inference.get("eval_iters", None) or range(num_iters + 1))
    # inference.iter_dir_fmt: where each iteration's results go under out_dir
    # ("iter{i}" by default; e.g. "iter{i}/6v" puts several runs under shared iter dirs).
    iter_dir_fmt = config.inference.get("iter_dir_fmt", "iter{i}")
    iter_dirs = [out_dir / iter_dir_fmt.format(i=i) for i in eval_iters]
    for d in iter_dirs:
        os.makedirs(d, exist_ok=True)
    print(f"writing to {out_dir}", flush=True)

    # inference.scene_range: [lo, hi) shard of the benchmark; summaries are then left to a
    # final summarize pass once every shard is done.
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
            gaussians = [gaussians[i] for i in eval_iters]
            refined = [None] * len(gaussians)  # EPA-aligned (target c2w, fxfycxcy) per evaluated iteration
            if base.use_pose_optimization:
                renders = []
                for k, g in enumerate(gaussians):
                    with torch.enable_grad():
                        aligned, refined[k] = base.pose_optimization(
                            gaussians=g, t_c2w=cam.pred_t_c2w, t_fxfycxcy=cam.pred_t_fxfycxcy,
                            w=base.width, h=base.height, target_images=tgt["image"],
                        )
                    renders.append(aligned)
            else:
                frozen = model.render_baseline_targets(out["gaussians_base"], cam)
                renders = [frozen] + list(out["renders"])
                renders = [renders[i] for i in eval_iters]

            # inference.vis_save: {root, subdir, tags: {iter: tag}} writes the evaluated target
            # renders as <root>/<tag>/<subdir>/<scene>/<frame:03d>.png, the frame being the
            # target's index in the benchmark json, so figures come from exactly the renders
            # the metrics were computed on.
            vis = config.inference.get("vis_save", None)
            if vis:
                frames = tgt["index"][0].reshape(-1).long().tolist()
                for it, r in zip(eval_iters, renders):
                    tag = vis["tags"].get(it)
                    if tag is None:
                        continue
                    vd = Path(vis["root"]) / tag / vis["subdir"] / inp["scene_name"][0]
                    os.makedirs(vd, exist_ok=True)
                    for vi, fi in enumerate(frames):
                        torchvision.utils.save_image(r[0, vi].float().clamp(0, 1), vd / f"{fi:03d}.png")

            # inference.ply_save: {root, tags: {iter: name}}: each evaluated iteration's Gaussians as
            # <root>/<tag>/<scene>/point_cloud.ply + cameras.json (layout: src/utils/ply_io.py);
            # the tag defaults to iter{i}. With EPA on, cameras.json also gets "target_epa": the target
            # cameras after EPA aligned them against that iteration's Gaussians.
            ply = config.inference.get("ply_save", None)
            if ply:
                for it, g, ref in zip(eval_iters, gaussians, refined):
                    tag = (ply.get("tags") or {}).get(it, f"iter{it}")
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

            for d, r in zip(iter_dirs, renders):
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
    for d in iter_dirs:
        print(f"\n=== {d.name}", flush=True)
        summarize_evaluation(str(d))


if __name__ == "__main__":
    main()
