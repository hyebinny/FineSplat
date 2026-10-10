"""Benchmark FineSplat's last-step Gaussians after pruning, once per pruning level (EPA with inference.pose_optimization).

    python -m src.commands.eval_prune <eval_config> [out_dir] [inference.key=value ...]
    <out_dir>/step4_prune/000001_<scene>/{metrics.json, gs_count.json}, <out_dir>/step4_prune/summary.csv
inference.prune_mode: "contribution" keeps the top prune_keep_per_view * V by mean compositing weight over
    the context views (10368/view -> step4_prune/, else step4_contrib<k>pv/); "radsplat" keeps the top-k by
    max blending weight (step4_radsplat<k>pv/); "opacity" thresholds at each tau in prune_taus (step4_prune<tau>/).
inference.scene_range: [lo, hi) shard; summaries are left to a final pass.
inference.vis_save: {root, subdir, tag} writes renders to <root>/<tag>_<level>/<subdir>/<scene>/<frame:03d>.png.
"""
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from easydict import EasyDict as edict
from gsplat import rasterization
from gsplat.cuda._wrapper import rasterize_to_indices_in_range
from omegaconf import OmegaConf

from src.commands.inference import AMP_DTYPE, build_dataloader, load_model
from src.commands.inference_finesplat import load_finesplat
from src.utils.eval_paths import load_eval_config, resolve_out_dir
from src.utils.metric_utils import export_results, summarize_evaluation
from src.model.sh_eval import _spherical_harmonics
from src.utils.camera_utils import fxfycxcy_to_K
from src.utils.ply_io import SH_C0


def select(g, keep):
    """The Gaussians of batch element 0 where keep is True."""
    return edict(xyz=g.xyz[:, keep], feature=g.feature[:, keep], scale=g.scale[:, keep],
                 rotation=g.rotation[:, keep], opacity=g.opacity[:, keep], opacity_precompute=None)


def opacity_keep(g, tau: float):
    """Effective opacity >= tau (degree-0 SH, so view-independent)."""
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


def radsplat_scores(g, c2w, fxfycxcy, w: int, h: int, model, batch_per_iter: int = 100,
                    return_alpha: bool = False):
    """RadSplat importance: max blending weight alpha_i * T_i over every ray, enumerated via rasterize_to_indices_in_range."""
    with torch.autocast("cuda", enabled=False):
        xyz = g.xyz[0].detach().float()
        rotation = F.normalize(g.rotation[0].detach().float(), dim=-1)
        scale = g.scale[0].detach().float().exp()
        opacity_sh = g.opacity[0].detach().float()
        n = len(xyz)
        scores = xyz.new_zeros(n)
        for i in range(c2w.shape[1]):
            cam = c2w[0, i].float()
            opacity = _spherical_harmonics(model.opacity_degree, (xyz - cam[:3, 3])[None], opacity_sh[None])[0]
            _, _, meta = rasterization(
                xyz, rotation, scale, opacity.sigmoid().squeeze(-1), xyz.new_ones(n, 1),
                cam.inverse()[None], fxfycxcy_to_K(fxfycxcy[0, i:i + 1].float()), w, h,
                sh_degree=None, near_plane=model.near_plane, far_plane=model.far_plane,
                packed=False, rasterize_mode="classic")
            means2d, conics, opac = meta["means2d"], meta["conics"], meta["opacities"]
            depths, offsets, flat = meta["depths"][0], meta["isect_offsets"], meta["flatten_ids"]
            tile = meta["tile_size"]
            trans = xyz.new_ones(1, h, w)
            bounds = torch.cat([offsets.flatten(), offsets.new_tensor([len(flat)])])
            n_batches = (int((bounds[1:] - bounds[:-1]).max()) + tile * tile - 1) // (tile * tile)
            for step in range(0, n_batches, batch_per_iter):
                gid, pid, _ = rasterize_to_indices_in_range(
                    step, step + batch_per_iter, trans, means2d, conics, opac, w, h, tile, offsets, flat)
                if len(gid) == 0:
                    break
                # front to back within each pixel: depth, then a stable sort by pixel
                o = depths[gid].argsort()
                o = o[pid[o].argsort(stable=True)]
                gid, pid = gid[o], pid[o]
                d = torch.stack([pid % w, pid // w], -1) + 0.5 - means2d[0, gid]
                c = conics[0, gid]
                sigma = 0.5 * (c[:, 0] * d[:, 0] ** 2 + c[:, 2] * d[:, 1] ** 2) + c[:, 1] * d[:, 0] * d[:, 1]
                alpha = (opac[0, gid] * torch.exp(-sigma)).clamp_max(0.999)
                log_t = torch.log1p(-alpha)
                cum = log_t.cumsum(0)
                first = torch.ones_like(pid, dtype=torch.bool)
                first[1:] = pid[1:] != pid[:-1]
                start = torch.where(first, torch.arange(len(pid), device=pid.device), 0).cummax(0).values
                excl = cum - log_t - (cum[start] - log_t[start])
                weight = alpha * trans.view(-1)[pid] * torch.exp(excl)
                scores.scatter_reduce_(0, gid, weight, reduce="amax")
                trans.view(-1).mul_(torch.exp(torch.zeros(h * w, device=pid.device).index_add_(0, pid, log_t)))
    return (scores, 1 - trans[0]) if return_alpha else scores


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
    elif mode == "radsplat":
        levels = [int(k) for k in config.inference.prune_keep_per_view]
        name = lambda k: f"{last}_radsplat{k}pv"
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
            if mode == "radsplat":
                scores = radsplat_scores(g, cam.pred_i_c2w, cam.pred_i_fxfycxcy,
                                         base.width, base.height, base)
            elif mode == "contribution":
                scores = contribution_scores(g, cam.pred_i_c2w, cam.pred_i_fxfycxcy,
                                             base.width, base.height, base, reduce)

            for t in levels:
                if mode == "opacity":
                    keep = opacity_keep(g, t)
                else:
                    keep = torch.zeros_like(scores, dtype=torch.bool)
                    keep[scores.topk(min(t * num_input, len(scores))).indices] = True
                gp = select(g, keep)
                if base.use_pose_optimization:
                    with torch.enable_grad():
                        render, _ = base.pose_optimization(
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
                    vd = Path(vis["root"]) / f"{vis['tag']}_{name(t)}" / vis["subdir"] / scene
                    os.makedirs(vd, exist_ok=True)
                    for vi, fi in enumerate(frames):
                        torchvision.utils.save_image(render[0, vi].float().clamp(0, 1), vd / f"{fi:03d}.png")
        torch.cuda.empty_cache()

    if config.inference.get("scene_range", None):
        return
    for d in metric_dirs.values():
        print(f"\n=== {d.name}", flush=True)
        summarize_evaluation(str(d))


def _selfcheck():
    """The enumerated compositing must reproduce gsplat's own alpha map on a random scene."""
    torch.manual_seed(0)
    n, w, h = 20000, 128, 128
    g = edict(xyz=torch.randn(1, n, 3, device="cuda") * 0.5 + torch.tensor([0, 0, 3.0], device="cuda"),
              rotation=torch.randn(1, n, 4, device="cuda"),
              scale=torch.full((1, n, 3), -3.5, device="cuda") + 0.5 * torch.randn(1, n, 3, device="cuda"),
              opacity=torch.randn(1, n, 1, 1, device="cuda") * 2)
    model = edict(opacity_degree=0, near_plane=0.01, far_plane=100.0)
    c2w = torch.eye(4, device="cuda")[None, None]
    k = torch.tensor([[[100.0, 100.0, 64.0, 64.0]]], device="cuda")
    scores, acc = radsplat_scores(g, c2w, k, w, h, model, return_alpha=True)
    op = torch.sigmoid(SH_C0 * g.opacity[0, :, 0, 0])
    _, ref, _ = rasterization(g.xyz[0], F.normalize(g.rotation[0], dim=-1), g.scale[0].exp(), op,
                              torch.ones(n, 1, device="cuda"), torch.eye(4, device="cuda")[None],
                              fxfycxcy_to_K(k[0]), w, h, sh_degree=None, near_plane=0.01, far_plane=100.0,
                              packed=False, rasterize_mode="classic")
    err = (acc - ref[0, ..., 0]).abs().max().item()
    print(f"max |alpha - gsplat alpha| = {err:.2e}, scores in [{scores.min():.3f}, {scores.max():.3f}], "
          f"{int((scores > 0).sum())}/{n} visible")
    assert err < 1e-3 and scores.max() <= 1.0 and (scores <= op + 1e-6).all()
    print("RADSPLAT SELFCHECK OK")


if __name__ == "__main__":
    _selfcheck() if sys.argv[1:] == ["--selfcheck"] else main()
