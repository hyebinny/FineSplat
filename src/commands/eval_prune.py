"""Benchmark FineSplat's last-step Gaussians after pruning, once per pruning level (EPA with inference.pose_optimization).

    python -m src.commands.eval_prune <eval_config> [out_dir] [inference.key=value ...]
    <out_dir>/step4_prune/000001_<scene>/{metrics.json, gs_count.json}, <out_dir>/step4_prune/summary.csv
inference.prune_mode: "contribution" keeps the top prune_keep_per_view * V Gaussians per scene, ranked by
    mean compositing weight over the context views, one run per budget k listed in prune_keep_per_view
    (step4_prune/ for the default budget, step4_contrib<k>pv/ otherwise); "opacity" thresholds the opacity
    at each tau in prune_taus (step4_prune<tau>/).
inference.scene_range: [lo, hi) shard; summaries are left to a final pass.
inference.vis_save: {root, subdir, tag} writes renders to <root>/<tag>_<level>/<subdir>/<scene>/<frame:03d>.png
    (vis_save.dir replaces <tag>_<level>); with cameras: true also <scene>/cameras.json (+ "target_epa" with EPA).
"""
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from easydict import EasyDict as edict
from gsplat import rasterization
from omegaconf import OmegaConf

from src.commands.inference_finesplat import AMP_DTYPE, build_dataloader, load_finesplat, load_model
from src.utils.eval_paths import load_eval_config, resolve_out_dir
from src.utils.metric_utils import export_results, summarize_evaluation
from src.model.sh_eval import _spherical_harmonics
from src.utils.camera_utils import fxfycxcy_to_K
from src.utils.ply_io import SH_C0, save_scene_cameras


def select(g, keep):
    """The Gaussians of batch element 0 where keep is True."""
    return edict(xyz=g.xyz[:, keep], feature=g.feature[:, keep], scale=g.scale[:, keep],
                 rotation=g.rotation[:, keep], opacity=g.opacity[:, keep], opacity_precompute=None)


def opacity_keep(g, tau: float):
    """Effective opacity >= tau (opacity is view-independent)."""
    return torch.sigmoid(SH_C0 * g.opacity[0, :, 0, 0].float()) >= tau


def contribution_scores(g, c2w, fxfycxcy, w: int, h: int, model, reduce: str = "mean"):
    """Mean (or max) over views of sum_pixels(T_i * alpha_i), read off as d(sum(render)) / d(color_i)."""
    with torch.inference_mode(False), torch.enable_grad(), torch.autocast("cuda", enabled=False):
        xyz = g.xyz[0].detach().float()
        rotation = F.normalize(g.rotation[0].detach().float(), dim=-1)
        scale = g.scale[0].detach().float().exp()
        opacity_sh = g.opacity[0].detach().float()
        scores = xyz.new_zeros(len(xyz))
        for i in range(c2w.shape[1]):
            cam = c2w[0, i].float()
            opacity = _spherical_harmonics(model.opacity_degree, (xyz - cam[:3, 3])[None], opacity_sh[None])[0]
            colors = xyz.new_ones(len(xyz), 1, requires_grad=True)
            render, _, _ = rasterization(
                xyz, rotation, scale, opacity.sigmoid().squeeze(-1), colors,
                cam.inverse()[None], fxfycxcy_to_K(fxfycxcy[0, i:i + 1].float()), w, h,
                sh_degree=None, near_plane=model.near_plane, far_plane=model.far_plane,
                packed=False, rasterize_mode="classic")
            s_i = torch.autograd.grad(render.sum(), colors)[0][:, 0]
            scores = torch.maximum(scores, s_i) if reduce == "max" else scores + s_i
    return scores if reduce == "max" else scores / c2w.shape[1]


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

    mode = config.inference.get("prune_mode", "opacity")
    last = f"step{config.stage2.num_steps}"
    if mode == "opacity":
        levels = [float(t) for t in config.inference.get("prune_taus", [0.01])]
        name = lambda t: last if t == 0 else f"{last}_prune{t}"
    elif mode == "contribution":
        levels = [int(k) for k in config.inference.prune_keep_per_view]
        reduce = config.inference.get("contrib_reduce", "mean")
        assert reduce in ("mean", "max"), reduce
        tag = lambda k: "prune" if k == 10368 else f"contrib{k}pv"
        name = lambda k: f"{last}_{tag(k)}" + ("" if reduce == "mean" else f"_{reduce}")
    else:
        raise ValueError(f"unknown inference.prune_mode: {mode!r}")
    num_input = config.data.num_input_frames
    out_dir = out_arg or Path(resolve_out_dir(config))
    metric_dirs = {t: out_dir / name(t) for t in levels}
    for d in metric_dirs.values():
        os.makedirs(d, exist_ok=True)
    vis = config.inference.get("vis_save", None)
    print(f"writing to {out_dir} ({mode} {levels}, EPA {base.use_pose_optimization})", flush=True)

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
            cam, g = out["cameras"], out["gaussians_all"][-1]
            scene = inp["scene_name"][0]
            frames = tgt["index"][0].reshape(-1).long().tolist()
            if mode == "contribution":
                scores = contribution_scores(g, cam.pred_i_c2w, cam.pred_i_fxfycxcy,
                                             base.width, base.height, base, reduce)

            for t in levels:
                if mode == "opacity":
                    keep = opacity_keep(g, t)
                else:
                    keep = torch.zeros_like(scores, dtype=torch.bool)
                    keep[scores.topk(min(t * num_input, len(scores))).indices] = True
                gp = select(g, keep)
                aligned = None
                if base.use_pose_optimization:
                    with torch.enable_grad():
                        render, aligned = base.pose_optimization(
                            gaussians=gp, t_c2w=cam.pred_t_c2w, t_fxfycxcy=cam.pred_t_fxfycxcy,
                            w=base.width, h=base.height, target_images=tgt["image"],
                        )
                else:
                    render = model.render_baseline_targets(gp, cam)
                export_results(
                    edict(input=inp, target=tgt, render=render.float()), str(metric_dirs[t]),
                    compute_metrics=True, visualize=False,
                    metric_resolution=config.inference.get("metric_resolution"),
                    metric_downsample=config.inference.get("metric_downsample", "bilinear_aa"),
                    uid=idx + 1,
                )
                with open(metric_dirs[t] / f"{idx + 1:06d}_{scene}" / "gs_count.json", "w") as f:
                    json.dump({"num_gaussians": int(keep.sum()), "num_gaussians_unpruned": keep.numel(),
                               "mode": mode, "level": t}, f)
                if vis:
                    vd = Path(vis["root"]) / (vis.get("dir") or f"{vis['tag']}_{name(t)}") / vis["subdir"] / scene
                    os.makedirs(vd, exist_ok=True)
                    for vi, fi in enumerate(frames):
                        torchvision.utils.save_image(render[0, vi].float().clamp(0, 1), vd / f"{fi:03d}.png")
                    if vis.get("cameras", False):
                        save_scene_cameras(vd / "cameras.json", cam, inp["index"][0].cpu().numpy(),
                                           tgt["index"][0].cpu().numpy(), config.data.resize_h, target_epa=aligned)
        torch.cuda.empty_cache()

    if config.inference.get("scene_range", None):
        return
    for d in metric_dirs.values():
        print(f"\n=== {d.name}", flush=True)
        summarize_evaluation(str(d))


if __name__ == "__main__":
    main()
