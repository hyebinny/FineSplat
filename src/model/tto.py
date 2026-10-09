"""Test-time optimization (TTO) of 2Xplat's fused patch tokens.

The whole model is frozen. Only the tokens produced by
``AppearanceExpert.fuse_tokens`` - the DPT-head output sitting immediately before
``gaussian_decoder`` - are optimized, against a photometric loss on the context
views the encoder already saw. Target views stay held out.

This mirrors ZipSplat's ``splatfactory/models/tto.py``, which optimizes its
scene tokens for 50 AdamW steps at lr 3e-3. The analogue is exact: in both models
the tokens are the last latent before a per-token MLP Gaussian head. ZipSplat
decodes 32 Gaussians per token; 2Xplat decodes ``gaussians_per_patch`` (4) per
patch token.

Two things differ from ZipSplat by necessity:
  - The colour skip connection ZipSplat freezes has no counterpart here; 2Xplat's
    head consumes a single token stream, so everything in it is optimized.
  - Opacity goes through 2Xplat's SH path per view. At the shipped
    ``opacity_degree: 0`` that path is view-independent (a single constant DC band),
    so the per-view evaluation is redundant; it is kept because the surrounding
    code is written for any degree.
"""

import time

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.model.gaussians import GaussianRenderer
from src.model.sh_eval import _spherical_harmonics

# ZipSplat's TTO recipe (splatfactory/models/tto.py) except the learning rate. ZipSplat's
# 3e-3 was tuned for tokens that each steer 32 Gaussians; 2Xplat's steer 4, and a sweep on
# the 140-scene DL3DV benchmark (3e-4 .. 9e-1, results in experiments/TTO/README.md) put
# the optimum at 4e-1: +1.76 PSNR against +0.16 at 3e-3, collapsing from 7e-1.
default_conf = {
    "num_steps": 50,
    "lr": 4e-1,
    "beta2": 0.95,  # beta1 fixed at 0.9
    "weight_decay": 0.05,
    "grad_clip": 1.0,  # max grad norm; None -> no clipping
    # 2Xplat's own objective (src/model/loss.py), so test-time optimisation pulls the
    # tokens toward what training already asked of them. ZipSplat's TTO used plain L1;
    # matching the base model matters more here, since these are 2Xplat's tokens. The
    # lr sweep in experiments/TTO/README.md predates this and was run under L1 alone.
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
    """Optimize fused tokens against the context views. Returns the optimized tokens.

    ``snapshot_steps`` (e.g. [1, 5, 10, 50]): run max(snapshot_steps) steps and return
    {step: (tokens, seconds_so_far)} instead. The lr is constant, so the tokens after k
    steps are exactly those of a k-step run.

    Args:
        model: frozen TwoExpertModel in eval mode with ``inference_mode`` set.
        tokens: (B, V*L, dim3) from ``AppearanceExpert.fuse_tokens``. The caller's
            tensor is not modified - it is cloned into a fresh fp32 leaf.
        i_fxfycxcy: context intrinsics (B, V, 4).
        i_c2w: context camera-to-world (B, V, 4, 4).
        context_images: ground-truth context images (B, V, 3, H, W).
        h, w: render resolution, matching the context images.
        conf: overrides for ``default_conf``.

    The loss on the context renders is ``photometric + perceptual_weight * VGG19``,
    2Xplat's own training objective. LPIPS is deliberately excluded: it roughly
    doubles step time for a negligible metric change.
    """
    if model.training:
        raise RuntimeError("TTO requires the model in eval mode - call model.eval() first.")
    if not model.inference_mode:
        raise RuntimeError(
            "TTO requires inference_mode: decode_from_tokens is called with t_c2w=None, "
            "which the training branch would dereference."
        )

    conf = OmegaConf.merge(OmegaConf.create(default_conf), OmegaConf.create(conf or {}))
    expert = model.appearance_expert

    # fp32 leaf: the encoder runs in bf16, but Adam on a bf16 leaf is fragile and
    # the Gaussian decode path upcasts anyway.
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
                # Unclamped, as LossComputer feeds it during training - clamping would
                # zero the gradient on exactly the pixels that overshoot.
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
