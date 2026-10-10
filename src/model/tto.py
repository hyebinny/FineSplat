"""Test-time optimization (TTO) of Stage 1's fused patch tokens.

The model is frozen; only the tokens from ``AppearanceExpert.fuse_tokens`` (right before
``gaussian_decoder``) are optimized against a photometric + perceptual loss on the
context views. Target views stay held out."""

import time

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.model.gaussians import GaussianRenderer
from src.model.sh_eval import _spherical_harmonics

default_conf = {
    "num_steps": 50,
    "lr": 4e-1,
    "beta2": 0.95,  # beta1 fixed at 0.9
    "weight_decay": 0.05,
    "grad_clip": 1.0,  # max grad norm; None -> no clipping
    "photometric": "l1",  # "l1" or "mse"
    "perceptual_weight": 0.2,  # 0 -> photometric only, and no VGG19 is loaded
}

_PERCEPTUAL = None


def _perceptual(device):
    """VGG19, built once and shared - optimize_tokens runs once per scene."""
    global _PERCEPTUAL
    if _PERCEPTUAL is None:
        from src.model.loss import PerceptualLoss

        _PERCEPTUAL = PerceptualLoss(device=str(device)).to(device).eval()
        _PERCEPTUAL.requires_grad_(False)
    return _PERCEPTUAL


def _render_context_views(
    model,
    gaussians,
    i_c2w: torch.Tensor,
    i_fxfycxcy: torch.Tensor,
    h: int,
    w: int,
) -> torch.Tensor:
    """Differentiably render the context views. Returns (B, V, 3, H, W)."""
    b, v = i_c2w.shape[:2]
    xyz = gaussians.xyz
    dirs = xyz[:, None, :, :] - i_c2w[..., :3, 3][..., None, :]  # (B, V, N, 3)
    opacity_broad = torch.broadcast_to(
        gaussians.opacity[..., None, :, :, :], (b, v, gaussians.opacity.shape[1], -1, 1),
    )
    opacity = _spherical_harmonics(model.opacity_degree, dirs, opacity_broad)

    renderings = GaussianRenderer.apply(
        xyz, gaussians.feature, gaussians.scale, gaussians.rotation, opacity,
        i_c2w, i_fxfycxcy, w, h,
        model.sh_degree, model.near_plane, model.far_plane,
    )
    return renderings.permute(0, 1, 4, 2, 3).contiguous()


def optimize_tokens(
    model,
    tokens: torch.Tensor,
    i_fxfycxcy: torch.Tensor,
    i_c2w: torch.Tensor,
    context_images: torch.Tensor,
    h: int,
    w: int,
    conf=None,
    snapshot_steps=None,
):
    """Optimize fused tokens (B, V*L, dim3) against context images (B, V, 3, H, W) with cameras
    i_c2w (B, V, 4, 4), i_fxfycxcy (B, V, 4); returns new tokens. ``conf`` overrides ``default_conf``.
    ``snapshot_steps`` (e.g. [1, 5, 10, 50]) instead runs max(snapshot_steps) steps and returns
    {step: (tokens, seconds_so_far)}; the lr is constant, so step k equals a k-step run."""
    if model.training:
        raise RuntimeError("TTO requires the model in eval mode - call model.eval() first.")
    if not model.inference_mode:
        raise RuntimeError(
            "TTO requires inference_mode: decode_from_tokens is called with t_c2w=None, "
            "which the training branch would dereference."
        )

    conf = OmegaConf.merge(OmegaConf.create(default_conf), OmegaConf.create(conf or {}))
    expert = model.appearance_expert

    # fp32 leaf: Adam on a bf16 leaf is fragile, and the Gaussian decode path upcasts anyway.
    tokens = tokens.detach().float().clone().requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [tokens],
        lr=conf.lr,
        betas=(0.9, conf.beta2),
        weight_decay=conf.weight_decay,
    )

    gt = context_images.float()
    assert conf.photometric in ("l1", "mse"), conf.photometric
    perceptual_fn = _perceptual(tokens.device) if conf.perceptual_weight > 0 else None
    flat_gt = gt.flatten(0, 1) if perceptual_fn is not None else None
    # enable_grad() overrides the ambient no_grad() the inference loop puts around eval.
    snaps = sorted(set(int(k) for k in snapshot_steps)) if snapshot_steps else None
    num_steps = snaps[-1] if snaps else conf.num_steps
    out = {}
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.enable_grad(), torch.autocast(device_type="cuda", enabled=False):
        for step in range(1, num_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            gaussians = expert.decode_from_tokens(tokens, i_fxfycxcy, i_c2w, None, h, w)
            render = _render_context_views(model, gaussians, i_c2w, i_fxfycxcy, h, w)
            loss = (F.l1_loss(render, gt) if conf.photometric == "l1"
                    else F.mse_loss(render, gt))
            if perceptual_fn is not None:
                # Unclamped, as in training; clamping would zero the gradient where pixels overshoot.
                loss = loss + conf.perceptual_weight * perceptual_fn(
                    render.flatten(0, 1), flat_gt)
            loss.backward()
            if conf.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_([tokens], conf.grad_clip)
            optimizer.step()
            if snaps and step in snaps:
                torch.cuda.synchronize()
                out[step] = (tokens.detach().clone(), time.perf_counter() - t0)

    return out if snaps else tokens.detach()
