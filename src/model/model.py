"""Stage 1 of FineSplat: the 2Xplat feed-forward 3DGS model.

A DA3 geometry expert predicts cameras; an MVP appearance expert maps ray-encoded
images to a 3D Gaussian field. Each patch is a ``gaussian_side x gaussian_side``
sub-grid of Gaussians; each Gaussian owns a ``gaussian_stride x gaussian_stride``
pixel cell and predicts a bounded image-space offset inside it."""

import contextlib
import os
import random
import torch
import traceback
import torch.nn.functional as F
from typing import Any
from torch import nn
from easydict import EasyDict as edict
from einops.layers.torch import Rearrange
from einops import rearrange
from safetensors.torch import load_file
from collections import OrderedDict
from dataclasses import dataclass

from src.model.sh_eval import _spherical_harmonics
from src.model.transformer import TransformerBlock
from src.model.dpt_head import DPTHead
from src.model.prope_custom import PropeDotProductAttention
from src.model.depth_anything.da_model.da3 import DepthAnything3
from src.model.gaussians import GaussianRenderer, GaussianField
from src.utils.camera_utils import (
    invert_SE3, compute_plucmap, fxfycxcy_to_K, mat_to_quat, quat_to_mat
)


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.RMSNorm, nn.LayerNorm)):
        module.reset_parameters()
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


def _fit_similarity(
    source_c2w: torch.Tensor,     # (B, V, 4, 4)
    reference_c2w: torch.Tensor,  # (B, V, 4, 4)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit the Sim(3) (rotation (B, 3, 3), scale (B,), translation (B, 3)) that maps
    ``source`` context poses onto ``reference`` ones. Fitted on detached poses: it is a
    gauge correction, and the SVD backward of a near-rotation matrix is ill-conditioned."""
    source_c2w, reference_c2w = source_c2w.detach(), reference_c2w.detach()
    b = source_c2w.shape[0]

    src_rot, ref_rot = source_c2w[:, :, :3, :3], reference_c2w[:, :, :3, :3]
    u, _, vh = torch.linalg.svd((ref_rot @ src_rot.transpose(-1, -2)).sum(dim=1))
    correction = torch.eye(3, dtype=u.dtype, device=u.device).expand(b, 3, 3).clone()
    correction[:, 2, 2] = torch.det(u @ vh)
    rotation = u @ correction @ vh

    src_c, ref_c = source_c2w[:, :, :3, 3], reference_c2w[:, :, :3, 3]
    src_mean, ref_mean = src_c.mean(dim=1), ref_c.mean(dim=1)
    rotated = (src_c - src_mean[:, None]) @ rotation.transpose(-1, -2)
    scale = (
        (rotated * (ref_c - ref_mean[:, None])).sum(dim=(1, 2))
        / rotated.square().sum(dim=(1, 2)).clamp_min(1e-12)
    )
    translation = ref_mean - scale[:, None] * (src_mean[:, None] @ rotation.transpose(-1, -2))[:, 0]
    return rotation, scale, translation


def _apply_similarity(
    c2w: torch.Tensor,
    rotation: torch.Tensor,
    scale: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    """Apply ``c -> scale * rotation @ c + translation`` to c2w (B, N, 4, 4), with
    rotation (B, 3, 3), scale (B,), translation (B, 3) as from ``_fit_similarity``."""
    rot = rotation[:, None] @ c2w[:, :, :3, :3]
    trans = (
        scale[:, None, None] * (c2w[:, :, :3, 3] @ rotation.transpose(-1, -2))
        + translation[:, None]
    )
    # Out-of-place: in-place writes on sliced views break autograd version tracking.
    return torch.cat(
        [torch.cat([rot, trans.unsqueeze(-1)], dim=-1), c2w[:, :, 3:, :]], dim=-2,
    )


@dataclass
class CameraBundle:
    pred_i_fxfycxcy: torch.Tensor  # (B, V, 4)
    pred_i_c2w:      torch.Tensor  # (B, V, 4, 4)
    pred_t_fxfycxcy: torch.Tensor  # (B, T, 4)
    pred_t_c2w:      torch.Tensor  # (B, T, 4, 4)
    gt_i_fxfycxcy:   torch.Tensor  # (B, V, 4)
    gt_i_c2w:        torch.Tensor  # (B, V, 4, 4)
    gt_t_fxfycxcy:   torch.Tensor  # (B, T, 4)
    gt_t_c2w:        torch.Tensor  # (B, T, 4, 4)


class GeometryExpert:
    """Predicts poses and intrinsics for all views with DA3, normalized to the scene scale."""

    def __init__(
        self,
        pose_regressor: DepthAnything3,
        scene_scale: float,
        inference_mode: bool,
        camera_mode: str | None,
    ):
        """``camera_mode`` (inference only) selects GT vs. predicted cameras:
        ``gt_pose_gt_intr``, ``pred_pose_gt_intr`` or ``pred_pose_pred_intr``."""
        self.pose_regressor = pose_regressor
        self.scene_scale = scene_scale
        self.inference_mode = inference_mode
        self.camera_mode = camera_mode

    def predict_cameras(
        self,
        input_data_dict: dict,
        target_data_dict: dict,
    ) -> CameraBundle:
        """Run DA3 on all views, normalize poses to scene scale, and apply camera mode."""
        gt_i_fxfycxcy = input_data_dict["fxfycxcy"].float()
        gt_i_c2w = input_data_dict["c2w"].float()
        gt_t_fxfycxcy = target_data_dict["fxfycxcy"].float()
        gt_t_c2w = target_data_dict["c2w"].float()

        if self.inference_mode and self.camera_mode == "gt_pose_gt_intr":
            # Fully posed: skip DA3, _select_cameras would discard its output anyway.
            return CameraBundle(
                pred_i_fxfycxcy=gt_i_fxfycxcy, pred_i_c2w=gt_i_c2w,
                pred_t_fxfycxcy=gt_t_fxfycxcy, pred_t_c2w=gt_t_c2w,
                gt_i_fxfycxcy=gt_i_fxfycxcy, gt_i_c2w=gt_i_c2w,
                gt_t_fxfycxcy=gt_t_fxfycxcy, gt_t_c2w=gt_t_c2w,
            )

        pred_i_fxfycxcy, pred_i_c2w, pred_t_fxfycxcy, pred_t_c2w = self._run_da3_and_normalize(
            input_data_dict["image"], target_data_dict["image"],
            torch.cat([gt_i_fxfycxcy, gt_t_fxfycxcy], dim=1),
        )

        if self.inference_mode:
            pred_i_fxfycxcy, pred_i_c2w, pred_t_fxfycxcy, pred_t_c2w = self._select_cameras(
                pred_i_fxfycxcy, pred_i_c2w, pred_t_fxfycxcy, pred_t_c2w,
                gt_i_fxfycxcy, gt_i_c2w, gt_t_fxfycxcy, gt_t_c2w,
            )

        return CameraBundle(
            pred_i_fxfycxcy=pred_i_fxfycxcy,
            pred_i_c2w=pred_i_c2w,
            pred_t_fxfycxcy=pred_t_fxfycxcy,
            pred_t_c2w=pred_t_c2w,
            gt_i_fxfycxcy=gt_i_fxfycxcy,
            gt_i_c2w=gt_i_c2w,
            gt_t_fxfycxcy=gt_t_fxfycxcy,
            gt_t_c2w=gt_t_c2w,
        )

    def _run_da3_and_normalize(
        self,
        input_images: torch.Tensor,
        target_images: torch.Tensor,
        gt_fxfycxcy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict cameras for all views. At inference the context cameras come from a
        context-only pass; a context+target pass gives the target poses, which are mapped
        onto the context-only gauge via Sim(3). Training uses one joint pass."""
        if not self.inference_mode or self.pose_regressor.training:
            return self._forward_and_normalize(input_images, target_images, gt_fxfycxcy)

        v = input_images.shape[1]
        context = self._forward_and_normalize(
            input_images, input_images[:, :0], gt_fxfycxcy[:, :v],
        )
        if target_images.shape[1] == 0:
            return context

        _, pass2_i_c2w, t_fxfycxcy, t_c2w = self._forward_and_normalize(
            input_images, target_images, gt_fxfycxcy,
        )

        with torch.autocast(device_type="cuda", enabled=False):
            rotation, scale, translation = _fit_similarity(pass2_i_c2w, context[1])
            t_c2w = _apply_similarity(t_c2w, rotation, scale, translation)
        return context[0], context[1], t_fxfycxcy, t_c2w

    def _forward_and_normalize(
        self,
        input_images: torch.Tensor,
        target_images: torch.Tensor,
        gt_fxfycxcy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward DA3 on all input+target images and normalize cameras to scene scale."""
        _, v, _, h, w = input_images.shape
        with torch.autocast(device_type="cuda", enabled=False):
            all_images = torch.cat([input_images, target_images], dim=1)
            output = self._forward_pose_regressor(all_images, gt_fxfycxcy)

            i_fxfycxcy_raw = output["fxfycxcy"][:, :v].float()
            i_c2w_raw = output["extrinsics"][:, :v].float()
            t_fxfycxcy_raw = output["fxfycxcy"][:, v:].float()
            t_c2w_raw = output["extrinsics"][:, v:].float()

            # DA3 outputs intrinsics at its own resolution; rescale to the model's input resolution.
            da_h, da_w = input_images.shape[-2:]
            intr_scale = torch.tensor(
                [w / float(da_w), h / float(da_h), w / float(da_w), h / float(da_h)],
                dtype=i_fxfycxcy_raw.dtype,
                device=i_fxfycxcy_raw.device,
            ).view(1, 1, 4)

            i_fxfycxcy = i_fxfycxcy_raw * intr_scale
            t_fxfycxcy = t_fxfycxcy_raw * intr_scale
            i_c2w, t_c2w = self._shared_scene_normalization(
                input_c2ws=i_c2w_raw,
                target_c2ws=t_c2w_raw,
                scene_scale=self.scene_scale,
            )
        return i_fxfycxcy, i_c2w, t_fxfycxcy, t_c2w

    def _forward_pose_regressor(self, images: torch.Tensor, gt_fxfycxcy: torch.Tensor) -> dict:
        return self.pose_regressor(images)

    @staticmethod
    def _shared_scene_normalization(
        input_c2ws: torch.Tensor,                 # (V, 4, 4) or (B, V, 4, 4)
        target_c2ws: torch.Tensor | None = None,  # (T, 4, 4) or (B, T, 4, 4)
        scene_scale: float = 1.0,
    ):
        """Scene normalization as in dataset.py: Gram-Schmidt canonical frame from the input
        poses, applied to input and target, then translations scaled by max |t| of the inputs."""
        squeeze_input = input_c2ws.ndim == 3
        squeeze_target = target_c2ws is not None and target_c2ws.ndim == 3

        if squeeze_input:
            input_c2ws = input_c2ws.unsqueeze(0)
        if target_c2ws is not None and squeeze_target:
            target_c2ws = target_c2ws.unsqueeze(0)

        position_avg = input_c2ws[:, :, :3, 3].mean(dim=1)
        forward_avg = input_c2ws[:, :, :3, 2].mean(dim=1)
        down_avg = input_c2ws[:, :, :3, 1].mean(dim=1)

        forward_avg = F.normalize(forward_avg, dim=-1)
        down_proj = (down_avg * forward_avg).sum(dim=-1, keepdim=True) * forward_avg
        down_avg = F.normalize(down_avg - down_proj, dim=-1)
        right_avg = torch.cross(down_avg, forward_avg, dim=-1)

        pos_avg = torch.eye(
            4, dtype=input_c2ws.dtype, device=input_c2ws.device
        ).expand(input_c2ws.shape[0], 4, 4).clone()
        pos_avg[:, :3, 0] = right_avg
        pos_avg[:, :3, 1] = down_avg
        pos_avg[:, :3, 2] = forward_avg
        pos_avg[:, :3, 3] = position_avg
        pos_avg_inv = torch.linalg.inv(pos_avg)

        input_c2ws = torch.matmul(pos_avg_inv.unsqueeze(1), input_c2ws)
        if target_c2ws is not None:
            target_c2ws = torch.matmul(pos_avg_inv.unsqueeze(1), target_c2ws)

        translations = input_c2ws[:, :, :3, 3].clone().detach()
        scene_extent = translations.abs().amax(dim=(1, 2))

        scale = 1.0 / (scene_scale * scene_extent)
        # Out-of-place: in-place writes on sliced views break autograd version tracking.
        input_scaled_t = input_c2ws[:, :, :3, 3] * scale[:, None, None]
        input_c2ws = torch.cat(
            [
                torch.cat([input_c2ws[:, :, :3, :3], input_scaled_t.unsqueeze(-1)], dim=-1),
                input_c2ws[:, :, 3:, :],
            ],
            dim=-2,
        )
        if target_c2ws is not None:
            target_scaled_t = target_c2ws[:, :, :3, 3] * scale[:, None, None]
            target_c2ws = torch.cat(
                [
                    torch.cat([target_c2ws[:, :, :3, :3], target_scaled_t.unsqueeze(-1)], dim=-1),
                    target_c2ws[:, :, 3:, :],
                ],
                dim=-2,
            )

        if squeeze_input:
            input_c2ws = input_c2ws.squeeze(0)
        if target_c2ws is not None and squeeze_target:
            target_c2ws = target_c2ws.squeeze(0)

        if target_c2ws is None:
            return input_c2ws
        return input_c2ws, target_c2ws

    def _select_cameras(
        self,
        pred_i_fxfycxcy: torch.Tensor,
        pred_i_c2w: torch.Tensor,
        pred_t_fxfycxcy: torch.Tensor,
        pred_t_c2w: torch.Tensor,
        gt_i_fxfycxcy: torch.Tensor,
        gt_i_c2w: torch.Tensor,
        gt_t_fxfycxcy: torch.Tensor,
        gt_t_c2w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Choose between predicted and GT cameras according to camera_mode."""
        if self.camera_mode == "gt_pose_gt_intr":
            return gt_i_fxfycxcy, gt_i_c2w, gt_t_fxfycxcy, gt_t_c2w
        if self.camera_mode == "pred_pose_gt_intr":
            return gt_i_fxfycxcy, pred_i_c2w, gt_t_fxfycxcy, pred_t_c2w
        if self.camera_mode == "pred_pose_pred_intr":
            return pred_i_fxfycxcy, pred_i_c2w, pred_t_fxfycxcy, pred_t_c2w
        raise ValueError(
            f"Unsupported camera_mode: {self.camera_mode!r}. "
            "Use one of: gt_pose_gt_intr, pred_pose_gt_intr, pred_pose_pred_intr."
        )


class AppearanceExpert:
    """MVP three-stage transformer mapping ray-encoded images to a 3D Gaussian field.

    Holds non-owning references to modules registered on ``TwoExpertModel``. ``gaussian_side``
    is the per-patch sub-grid side; ``gaussian_stride`` the pixels per side of each cell."""

    def __init__(
        self,
        image_tokenizer: nn.Sequential,
        stage1: nn.ModuleList,
        stage2: nn.ModuleList,
        stage3: nn.ModuleList,
        merge_block1: nn.Conv2d,
        resize_block1: nn.Linear,
        merge_block2: nn.Conv2d,
        resize_block2: nn.Linear,
        attention2: PropeDotProductAttention,
        attention3: PropeDotProductAttention,
        dpt_head: DPTHead,
        gaussian_decoder: nn.Sequential,
        register_token_init: nn.Parameter,
        patch_size: int,
        gaussian_side: int,
        gaussian_stride: int,
        num_register_tokens: int,
        group_size: int,
        color_dim: int,
        opacity_dim: int,
        sh_degree: int,
        opacity_degree: int,
        scale_bias: float,
        scale_max: float,
        opacity_bias: float,
        max_dist: float,
        inference_mode: bool,
        dynamic_grouping: bool = False,
    ):
        self.image_tokenizer = image_tokenizer
        self.stage1 = stage1
        self.stage2 = stage2
        self.stage3 = stage3
        self.merge_block1 = merge_block1
        self.resize_block1 = resize_block1
        self.merge_block2 = merge_block2
        self.resize_block2 = resize_block2
        self.attention2 = attention2
        self.attention3 = attention3
        self.dpt_head = dpt_head
        self.gaussian_decoder = gaussian_decoder
        self.register_token_init = register_token_init
        self.patch_size = patch_size
        self.gaussian_side = gaussian_side
        self.gaussian_stride = gaussian_stride
        self.num_register_tokens = num_register_tokens
        self.group_size = group_size
        self.color_dim = color_dim
        self.opacity_dim = opacity_dim
        self.sh_degree = sh_degree
        self.opacity_degree = opacity_degree
        self.scale_bias = scale_bias
        self.scale_max = scale_max
        self.opacity_bias = opacity_bias
        self.max_dist = max_dist
        # Training only: sample the stage-2 view-group size per forward (see grouping()).
        self.dynamic_grouping = dynamic_grouping
        self.training = True  # mirrors the owning TwoExpertModel (set in its train())
        self.inference_mode = inference_mode

    def sample_group_size(self, v: int) -> int:
        """A divisor of v in [2, v), larger groups more likely; the fixed size if none."""
        candidates = [g for g in range(2, v) if v % g == 0]
        if not candidates:
            return self.group_size
        return random.choices(candidates, weights=candidates)[0]

    @contextlib.contextmanager
    def grouping(self, v: int, enabled: bool):
        """Run stages 1-3 under a sampled view-group size (dynamic_grouping and enabled).

        All three stages read self.group_size, so the override must span the whole encode."""
        if not (self.dynamic_grouping and enabled):
            yield self.group_size
            return
        fixed = self.group_size
        self.group_size = self.sample_group_size(v)
        try:
            yield self.group_size
        finally:
            self.group_size = fixed

    def predict_gaussians(
        self,
        raymap_images: torch.Tensor,   # (B, V, C, H, W)
        i_w2c: torch.Tensor,           # (B, V, 4, 4)
        Ks: torch.Tensor,              # (B, V, 3, 3)
        i_fxfycxcy: torch.Tensor,      # (B, V, 4)
        i_c2w: torch.Tensor,           # (B, V, 4, 4)
        t_c2w: torch.Tensor,           # (B, T, 4, 4)
    ) -> GaussianField:
        """Run the full three-stage transformer pipeline and decode Gaussian parameters."""
        _, v, _, h, w = raymap_images.shape
        with self.grouping(v, enabled=self.training):
            x, s1_patch_tokens = self._stage1_tokenize_and_encode(raymap_images)
            x, s2_patch_tokens = self._stage2_cross_view_and_downsample(x, i_w2c, Ks, h, w)
            s3_patch_tokens = self._stage3_global_cross_view(x, i_w2c, Ks)
        return self._decode_gaussians(
            s1_patch_tokens, s2_patch_tokens, s3_patch_tokens,
            i_fxfycxcy, i_c2w, t_c2w, h, w,
        )

    def _stage1_tokenize_and_encode(
        self,
        raymap_images: torch.Tensor,
    ):
        """Patch-embed ray-map images with register tokens, then run stage-1 transformer."""
        b, v, _, h, w = raymap_images.shape
        register_tokens = self.register_token_init.repeat(b, v, 1, 1)
        x = self.image_tokenizer(raymap_images)
        x = rearrange(x, "b (v l) d -> b v l d", v=v)
        x = torch.cat([register_tokens, x], dim=2)
        x = rearrange(x, "b v l d -> (b v) l d")
        x = self._run_stage1_blocks(x, None)

        r_tokens, s1_patch_tokens = x[:, :self.num_register_tokens], x[:, self.num_register_tokens:]
        r_tokens = self.resize_block1(r_tokens)

        h_patches = h // self.patch_size
        w_patches = w // self.patch_size
        i_tokens = rearrange(s1_patch_tokens, "b (hh ww) d -> b d hh ww", hh=h_patches, ww=w_patches)
        i_tokens = self.merge_block1(i_tokens)
        i_tokens = rearrange(i_tokens, "b d hh ww -> b (hh ww) d", hh=h_patches // 2, ww=w_patches // 2)

        x = torch.cat([r_tokens, i_tokens], dim=1)
        x = rearrange(x, "(b g v) l d -> (b g) (v l) d", g=v // self.group_size, v=self.group_size)
        return x, s1_patch_tokens

    def _stage2_cross_view_and_downsample(
        self,
        x: torch.Tensor,
        i_w2c: torch.Tensor,
        Ks: torch.Tensor,
        h: int,
        w: int,
    ):
        """Run grouped cross-view attention (stage 2) and spatially downsample patch tokens."""
        v = i_w2c.shape[1]
        info = {
            "num_input_views": v,
            "w2c": rearrange(i_w2c, "b (g v) ... -> (b g) v ...", g=v // self.group_size, v=self.group_size),
            "Ks": rearrange(Ks, "b (g v) ... -> (b g) v ...", g=v // self.group_size, v=self.group_size),
            "attn2": self.attention2,
        }
        x = self._run_stage2_blocks(x, info)

        r_tokens, s2_patch_tokens = x[:, :self.num_register_tokens], x[:, self.num_register_tokens:]
        r_tokens = self.resize_block2(r_tokens)

        h_patches = (h // self.patch_size) // 2
        w_patches = (w // self.patch_size) // 2
        i_tokens = rearrange(s2_patch_tokens, "b (hh ww) d -> b d hh ww", hh=h_patches, ww=w_patches)
        i_tokens = self.merge_block2(i_tokens)
        i_tokens = rearrange(i_tokens, "b d hh ww -> b (hh ww) d", hh=h_patches // 2, ww=w_patches // 2)

        x = torch.cat([r_tokens, i_tokens], dim=1)
        x = rearrange(x, "(b v) l d -> b (v l) d", v=v)
        return x, s2_patch_tokens

    def _stage3_global_cross_view(
        self,
        x: torch.Tensor,
        i_w2c: torch.Tensor,
        Ks: torch.Tensor,
    ) -> torch.Tensor:
        """Run global cross-view attention (stage 3) across all input views."""
        v = i_w2c.shape[1]
        info = {
            "num_input_views": v,
            "attn3": self.attention3,
            "w2c": i_w2c,
            "Ks": Ks,
        }
        x = self._run_stage3_blocks(x, info)
        return x[:, self.num_register_tokens:]

    def _compute_offset_rays(
        self,
        fxfycxcy: torch.Tensor,     # (B, V, 4)
        c2w: torch.Tensor,          # (B, V, 4, 4)
        h: int,
        w: int,
        image_offset: torch.Tensor,  # (B, V*gh*gw, 2)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rays through sub-cell centers displaced by the (pre-tanh, xy) offset; returns
        ``ray_o, ray_d`` of shape (B, V*gh*gw, 3). The offset is squashed to +-0.5 cell so a
        Gaussian never leaves its cell; intrinsics are in input-image pixels like the grid."""
        b, v = fxfycxcy.size(0), fxfycxcy.size(1)
        stride = self.gaussian_stride
        gh, gw = h // stride, w // stride

        offset_px = 0.5 * stride * image_offset.tanh()
        offset_px = offset_px.reshape(b * v, gh * gw, 2)

        fxfycxcy = fxfycxcy.reshape(b * v, 4)
        c2w = c2w.reshape(b * v, 4, 4)

        idx_x = torch.arange(gw, device=c2w.device)[None, :].expand(gh, -1)
        idx_y = torch.arange(gh, device=c2w.device)[:, None].expand(-1, gw)

        x = ((idx_x.flatten() + 0.5) * stride + offset_px[..., 0]
             - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
        y = ((idx_y.flatten() + 0.5) * stride + offset_px[..., 1]
             - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
        z = torch.ones_like(x)

        ray_d = torch.stack([x, y, z], dim=1)
        ray_d = torch.bmm(c2w[:, :3, :3], ray_d)
        ray_d = ray_d / torch.norm(ray_d, dim=1, keepdim=True)

        ray_o = c2w[:, :3, 3:4].expand(b * v, -1, gh * gw)

        ray_o = rearrange(ray_o, "(b v) c n -> b (v n) c", b=b, v=v)
        ray_d = rearrange(ray_d, "(b v) c n -> b (v n) c", b=b, v=v)

        return ray_o, ray_d

    def _decode_gaussians(
        self,
        s1_patch_tokens: torch.Tensor,
        s2_patch_tokens: torch.Tensor,
        s3_patch_tokens: torch.Tensor,
        i_fxfycxcy: torch.Tensor,
        i_c2w: torch.Tensor,
        t_c2w: torch.Tensor,
        h: int,
        w: int,
    ) -> GaussianField:
        output_tokens = self.fuse_tokens(
            s1_patch_tokens, s2_patch_tokens, s3_patch_tokens, i_c2w.shape[1], h, w,
        )
        return self.decode_from_tokens(output_tokens, i_fxfycxcy, i_c2w, t_c2w, h, w)

    def fuse_tokens(
        self,
        s1_patch_tokens: torch.Tensor,
        s2_patch_tokens: torch.Tensor,
        s3_patch_tokens: torch.Tensor,
        v: int,
        h: int,
        w: int,
    ) -> torch.Tensor:
        """Fuse the three stages with the DPT head into per-patch tokens (B, V*L, dim3):
        the tokens right before ``gaussian_decoder``, which Stage 2 and TTO operate on."""
        output_tokens = self.dpt_head(
            [s1_patch_tokens, s2_patch_tokens, s3_patch_tokens], [h, w], self.patch_size,
        )
        return rearrange(output_tokens, "(b v) l d -> b (v l) d", v=v)

    def decode_from_tokens(
        self,
        output_tokens: torch.Tensor,
        i_fxfycxcy: torch.Tensor,
        i_c2w: torch.Tensor,
        t_c2w: torch.Tensor,
        h: int,
        w: int,
    ) -> GaussianField:
        """Decode fused tokens into 3D Gaussian parameters."""
        raw = self.raw_from_tokens(output_tokens, i_c2w.shape[1], h, w)
        return self.gaussians_from_raw(raw, i_fxfycxcy, i_c2w, t_c2w, h, w)

    def raw_from_tokens(self, output_tokens: torch.Tensor, v: int, h: int, w: int) -> torch.Tensor:
        """Gaussian head + unfold: raw pre-activation parameters (B, N, 23) laid out as
        ``[depth 1 | image_offset 2 | feature C | scale 3 | rotation 4 | opacity O]``."""
        gaussians = self.gaussian_decoder(output_tokens)
        return rearrange(
            gaussians, "b (v hh ww) (ph pw d) -> b (v hh ph ww pw) d",
            v=v,
            hh=h // self.patch_size,
            ww=w // self.patch_size,
            ph=self.gaussian_side,
            pw=self.gaussian_side,
        )

    def gaussians_from_raw(
        self,
        raw: torch.Tensor,
        i_fxfycxcy: torch.Tensor,
        i_c2w: torch.Tensor,
        t_c2w: torch.Tensor,
        h: int,
        w: int,
    ) -> GaussianField:
        """Apply the activations and ray parameterisation to raw (B, N, 23) parameters."""
        b, v = i_c2w.shape[:2]
        t = t_c2w.shape[1] if t_c2w is not None else 0
        depth, image_offset, feature, scale, rotation, opacity = torch.split(
            raw, [1, 2, self.color_dim, 3, 4, self.opacity_dim], dim=-1,
        )
        depth = depth.float()
        image_offset = image_offset.float()
        feature = feature.float()
        scale = scale.float()
        rotation = rotation.float()
        opacity = opacity.float()

        with torch.autocast(device_type="cuda", enabled=False):
            rayo_gs, rayd_gs = self._compute_offset_rays(
                i_fxfycxcy, i_c2w, h, w, image_offset,
            )
            scale = (scale + self.scale_bias).clamp(max=self.scale_max)
            # Bias only the DC (sh0) opacity. Out-of-place: `opacity` is a torch.split
            # view, and in-place writes into it fail once tokens require grad (TTO).
            dc_bias = torch.zeros_like(opacity)
            dc_bias[..., 0] = self.opacity_bias
            opacity = opacity + dc_bias
            feature = rearrange(feature, "b n (c d) -> b n d c", c=3).contiguous()
            opacity = rearrange(opacity, "b n (c d) -> b n d c", c=1).contiguous()
            dist = depth.sigmoid() * self.max_dist
            xyz = dist * rayd_gs + rayo_gs

            if not self.inference_mode:
                dirs = xyz[:, None, :, :] - t_c2w[..., :3, 3][..., None, :]  # (B, T, N, 3)
                opacity_broad = torch.broadcast_to(
                    opacity[..., None, :, :, :], (b, t, opacity.shape[1], -1, 1),
                )
                opacity_precompute = _spherical_harmonics(self.opacity_degree, dirs, opacity_broad)
            else:
                opacity_precompute = None

        return GaussianField(
            xyz=xyz,
            feature=feature,
            scale=scale,
            rotation=rotation,
            opacity=opacity,
            opacity_precompute=opacity_precompute,
        )

    def _run_stage1_blocks(self, x: torch.Tensor, info: dict | None) -> torch.Tensor:
        for block in self.stage1:
            x = block(x, False, 1, info)
        return x

    def _run_stage2_blocks(self, x: torch.Tensor, info: dict) -> torch.Tensor:
        g = self.group_size
        v = info["num_input_views"]
        for i, block in enumerate(self.stage2):
            if i % 2 == 0:
                x = rearrange(x, "(b g) (v l) d -> (b g v) l d", g=v // g, v=g)
                x = block(x, False, 2, info)
                x = rearrange(x, "(b g v) l d -> (b g) (v l) d", g=v // g, v=g)
            else:
                x = block(x, True, 2, info)
        return rearrange(x, "(b g) (v l) d -> (b g v) l d", g=v // g, v=g)

    def _run_stage3_blocks(self, x: torch.Tensor, info: dict) -> torch.Tensor:
        v = info["num_input_views"]
        for i, block in enumerate(self.stage3):
            if i % 2 == 0:
                x = rearrange(x, "b (v l) d -> (b v) l d", v=v)
                x = block(x, False, 3, info)
                x = rearrange(x, "(b v) l d -> b (v l) d", v=v)
            else:
                x = block(x, True, 3, info)
        return rearrange(x, "b (v l) d -> (b v) l d", v=v)


class TwoExpertModel(nn.Module):
    """Stage 1 model: wires the GeometryExpert and AppearanceExpert together."""

    def __init__(self, config: Any) -> None:
        """A ``config.inference`` section selects inference mode; its absence means training."""
        super().__init__()

        self.dim1 = config.model.dim1
        self.dim2 = config.model.dim2
        self.dim3 = config.model.dim3
        self.patch_size = config.model.patch_size
        self.num_register_tokens = config.model.num_register_tokens
        self.group_size = config.model.group_size
        self.dynamic_grouping = bool(config.model.get("dynamic_grouping", False))
        self.head_dim = config.model.head_dim
        self.inter_multi = config.model.inter_multi
        self.qk_norm = config.model.qk_norm
        self.in_channels = config.model.in_channels
        self.stage1_nlayer = config.model.stage1_nlayer
        self.stage2_nlayer = config.model.stage2_nlayer
        self.stage3_nlayer = config.model.stage3_nlayer
        self.sh_degree = config.model.gaussians.sh_degree
        self.opacity_degree = config.model.gaussians.opacity_degree
        self.near_plane = config.model.gaussians.near_plane
        self.far_plane = config.model.gaussians.far_plane
        self.scale_bias = config.model.gaussians.scale_bias
        self.scale_max = config.model.gaussians.scale_max
        self.opacity_bias = config.model.gaussians.opacity_bias
        self.max_dist = config.model.gaussians.max_dist
        self.da_model_name = config.model.da_model_name
        # training.grad_checkpoint_every: 0 disables DA3 gradient checkpointing, N checkpoints every N-th block.
        self.grad_ckpt_every = (
            config.training.get("grad_checkpoint_every", 0) if hasattr(config, "training") else 0
        )
        self.da_weights_path = getattr(config.model, "da_model_weights_path", None)
        self.mvp_weights_path = getattr(config.model, "mvp_weights_path", None)
        self.scene_scale = config.data.scene_scale
        self.width = config.data.resize_w
        self.height = config.data.resize_h
        self.inference_mode = hasattr(config, "inference")
        self.camera_mode = (
            getattr(config.inference, "camera_mode", "pred_pose_pred_intr")
            if self.inference_mode else None
        )
        # inference.epa_steps: Adam steps of EPA (evaluation-time pose alignment); default 200.
        epa_steps = getattr(config.inference, "epa_steps", None) if hasattr(config, "inference") else None
        self.epa_steps = 200 if epa_steps is None else int(epa_steps)
        self.use_pose_optimization = bool(
            getattr(config.inference, "pose_optimization", False)
            if self.inference_mode else False
        )

        self.color_dim = 3 * (self.sh_degree + 1) ** 2
        self.opacity_dim = 1 * (self.opacity_degree + 1) ** 2

        # Gaussians per patch as a gaussian_side x gaussian_side sub-grid; default one per pixel.
        self.gaussians_per_patch = getattr(
            config.model, "gaussians_per_patch", self.patch_size ** 2,
        )
        self.gaussian_side = int(round(self.gaussians_per_patch ** 0.5))
        assert self.gaussian_side ** 2 == self.gaussians_per_patch, \
            "gaussians_per_patch must be a perfect square"
        assert self.patch_size % self.gaussian_side == 0, \
            "patch_size must be divisible by sqrt(gaussians_per_patch)"
        self.gaussian_stride = self.patch_size // self.gaussian_side

        self._build_geometry_modules()
        self._build_appearance_modules()

        self.geometry_expert = GeometryExpert(
            pose_regressor=self.pose_regressor,
            scene_scale=self.scene_scale,
            inference_mode=self.inference_mode,
            camera_mode=self.camera_mode,
        )
        self.appearance_expert = AppearanceExpert(
            image_tokenizer=self.image_tokenizer,
            stage1=self.stage1,
            stage2=self.stage2,
            stage3=self.stage3,
            merge_block1=self.merge_block1,
            resize_block1=self.resize_block1,
            merge_block2=self.merge_block2,
            resize_block2=self.resize_block2,
            attention2=self.attention2,
            attention3=self.attention3,
            dpt_head=self.dpt_head,
            gaussian_decoder=self.gaussian_decoder,
            register_token_init=self.register_token_init,
            patch_size=self.patch_size,
            gaussian_side=self.gaussian_side,
            gaussian_stride=self.gaussian_stride,
            num_register_tokens=self.num_register_tokens,
            group_size=self.group_size,
            color_dim=self.color_dim,
            opacity_dim=self.opacity_dim,
            sh_degree=self.sh_degree,
            opacity_degree=self.opacity_degree,
            scale_bias=self.scale_bias,
            scale_max=self.scale_max,
            opacity_bias=self.opacity_bias,
            max_dist=self.max_dist,
            inference_mode=self.inference_mode,
            dynamic_grouping=self.dynamic_grouping,
        )

        if not self.inference_mode:
            from src.model.loss import LossComputer
            self.loss_computer = LossComputer(config)

    def _build_appearance_modules(self):
        """Build the appearance-expert modules (names must match checkpoint keys)."""
        # Marks a state dict as using the depth + image_offset layout.
        self.register_buffer("_image_offset_version", torch.tensor(1))

        self.image_tokenizer = self._create_patch_tokenizer(
            self.in_channels, self.patch_size, self.dim1,
        )
        self.gaussian_decoder = nn.Sequential(
            nn.LayerNorm(self.dim3, bias=False),
            nn.Linear(
                self.dim3,
                self.gaussians_per_patch * (1 + 2 + self.color_dim + 3 + 4 + self.opacity_dim),
                bias=False,
            ),
        )
        self.stage1 = self._build_transformer_stage(self.dim1, self.stage1_nlayer)
        self.stage2 = self._build_transformer_stage(self.dim2, self.stage2_nlayer)
        self.stage3 = self._build_transformer_stage(self.dim3, self.stage3_nlayer)

        self.register_token_init = nn.Parameter(
            torch.randn(1, 1, self.num_register_tokens, self.dim1),
        )
        nn.init.normal_(self.register_token_init, mean=0.0, std=0.02)

        self.attention2 = PropeDotProductAttention(
            head_dim=self.head_dim,
            patches_x=self.width // (self.patch_size * 2),
            patches_y=self.height // (self.patch_size * 2),
            image_width=self.width,
            image_height=self.height,
            num_register_tokens=self.num_register_tokens,
        )
        self.attention3 = PropeDotProductAttention(
            head_dim=self.head_dim,
            patches_x=self.width // (self.patch_size * 4),
            patches_y=self.height // (self.patch_size * 4),
            image_width=self.width,
            image_height=self.height,
            num_register_tokens=self.num_register_tokens,
        )
        self.merge_block1 = nn.Conv2d(
            self.dim1, self.dim2, kernel_size=2, stride=2,
            padding=0, bias=True, groups=self.dim1,
        )
        self.resize_block1 = nn.Linear(self.dim1, self.dim2)
        self.merge_block2 = nn.Conv2d(
            self.dim2, self.dim3, kernel_size=2, stride=2,
            padding=0, bias=True, groups=self.dim2,
        )
        self.resize_block2 = nn.Linear(self.dim2, self.dim3)
        self.dpt_head = DPTHead(
            dim_in=[self.dim1, self.dim2, self.dim3],
            features=self.dim3,
            out_channels=[self.dim1, self.dim2, self.dim3],
        )

        if self.mvp_weights_path != "":
            checkpoint = torch.load(self.mvp_weights_path, map_location="cpu", weights_only=True)
            state_dict = checkpoint["ema"] if "ema" in checkpoint else checkpoint
            result = self.load_state_dict(state_dict, strict=False)
            print(f"Loaded MVP appearance weights from {self.mvp_weights_path}")

    def _build_geometry_modules(self):
        self.pose_regressor = DepthAnything3(model_name=self.da_model_name)
        if self.da_weights_path != "":
            state_dict = load_file(self.da_weights_path)
            self.pose_regressor.load_state_dict(state_dict, strict=False)
            print(f"Loaded DA3 ({self.da_model_name}) pose regressor weights from {self.da_weights_path}")
        self.pose_regressor.model.backbone.pretrained.grad_ckpt_every = self.grad_ckpt_every
        self.pose_regressor.prune_layers()

    def _build_transformer_stage(self, dim: int, nlayer: int) -> nn.ModuleList:
        return nn.ModuleList([
            TransformerBlock(dim, False, self.head_dim, self.inter_multi, self.qk_norm)
            for _ in range(nlayer)
        ])

    @staticmethod
    def _create_patch_tokenizer(
        in_channels: int, patch_size: int, d_model: int,
    ) -> nn.Sequential:
        """Patch embedding: (B, V, C, H, W) -> (B, V*L, d_model)."""
        return nn.Sequential(
            Rearrange(
                "b v c (hh ph) (ww pw) -> b (v hh ww) (ph pw c)",
                ph=patch_size, pw=patch_size,
            ),
            nn.Linear(in_channels * (patch_size ** 2), d_model, bias=False),
            nn.LayerNorm(d_model, bias=False),
        )

    def train(self, mode: bool = True):
        """Keeps the loss modules permanently in eval mode."""
        super().train(mode)
        self.appearance_expert.training = mode
        if not self.inference_mode:
            self.loss_computer.eval()

    def _build_raymap_input(
        self,
        input_data_dict: dict,
        i_fxfycxcy: torch.Tensor,
        i_c2w: torch.Tensor,
    ):
        """Construct plucker ray-map images: [ray_origin, ray_dir, origin×dir, image]."""
        h, w = input_data_dict["image"].shape[-2:]
        with torch.autocast(device_type="cuda", enabled=False):
            ray_o, ray_d = compute_plucmap(i_fxfycxcy, i_c2w, h, w)
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)
            i_normalized_image = input_data_dict["image"] * 2.0 - 1.0
            i_raymap_images = torch.concat([ray_o, ray_d, o_cross_d, i_normalized_image], dim=2)
            Ks = fxfycxcy_to_K(i_fxfycxcy)
            i_w2c = invert_SE3(i_c2w)
        return i_raymap_images, Ks, i_w2c

    def _render_and_compute_loss(
        self,
        gaussians: GaussianField,
        cameras: CameraBundle,
        target_images: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Render Gaussians to all target views and compute photometric + pose losses."""
        h, w = target_images.shape[-2:]
        with torch.autocast(device_type="cuda", enabled=False):
            renderings = GaussianRenderer.apply(
                gaussians.xyz, gaussians.feature, gaussians.scale,
                gaussians.rotation, gaussians.opacity_precompute,
                cameras.pred_t_c2w, cameras.pred_t_fxfycxcy, w, h,
                self.sh_degree, self.near_plane, self.far_plane,
            )
        renderings = renderings.permute(0, 1, 4, 2, 3).contiguous()  # (B, V, 3, H, W)

        cam_info = {
            "pred_fxfycxcy": torch.cat([cameras.pred_i_fxfycxcy, cameras.pred_t_fxfycxcy], dim=1),
            "pred_c2w": torch.cat([cameras.pred_i_c2w, cameras.pred_t_c2w], dim=1),
            "gt_fxfycxcy": torch.cat([cameras.gt_i_fxfycxcy, cameras.gt_t_fxfycxcy], dim=1),
            "gt_c2w": torch.cat([cameras.gt_i_c2w, cameras.gt_t_c2w], dim=1),
        }
        loss_metrics = self.loss_computer(renderings, target_images, cam_info)

        return renderings, loss_metrics

    def _render_target_views(
        self,
        gaussians: GaussianField,
        t_c2w: torch.Tensor,
        t_fxfycxcy: torch.Tensor,
        w: int,
        h: int,
    ) -> torch.Tensor:
        """Render each target view sequentially without gradient tracking."""
        t = t_c2w.shape[1]
        xyz = gaussians.xyz[0]
        feature = gaussians.feature[0]
        scale = gaussians.scale[0]
        rotation = gaussians.rotation[0]
        opacity = gaussians.opacity[0]
        renderings = []
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
            for i in range(t):
                dir = xyz - t_c2w[0, i:i+1, :3, 3][None, ...]
                opacity_i = _spherical_harmonics(
                    self.opacity_degree, dir, opacity[None, ...],
                )[0]
                rendering = GaussianRenderer.render(
                    xyz, feature, scale, rotation, opacity_i,
                    t_c2w[0, i], t_fxfycxcy[0, i], w, h,
                    self.sh_degree, self.near_plane, self.far_plane,
                )
                renderings.append(rendering)
        renderings = torch.cat(renderings, dim=0)[None, ...]
        return renderings.permute(0, 1, 4, 2, 3).contiguous()  # (1, T, 3, H, W)

    def _training_forward(
        self,
        gaussians: GaussianField,
        cameras: CameraBundle,
        target_data_dict: dict,
        input_data_dict: dict,
    ) -> edict:
        """Render and compute training losses; returns edict(input, target, loss_metrics, render)."""
        renderings, loss_metrics = self._render_and_compute_loss(
            gaussians, cameras, target_data_dict["image"],
        )
        return edict(
            input=input_data_dict,
            target=target_data_dict,
            loss_metrics=loss_metrics,
            render=renderings,
        )

    def _inference_forward(
        self,
        gaussians: GaussianField,
        cameras: CameraBundle,
        target_data_dict: dict,
        input_data_dict: dict,
    ) -> edict:
        """Render target views (after EPA if enabled); returns edict(input, target, render, gaussians, cameras)."""
        h, w = target_data_dict["image"].shape[-2:]

        if self.use_pose_optimization:
            prev_grad_state = torch.is_grad_enabled()
            torch.set_grad_enabled(True)
            renderings, _ = self.pose_optimization(
                gaussians=gaussians,
                t_c2w=cameras.pred_t_c2w,
                t_fxfycxcy=cameras.pred_t_fxfycxcy,
                w=w, h=h,
                target_images=target_data_dict["image"],
            )
            torch.set_grad_enabled(prev_grad_state)
        else:
            renderings = self._render_target_views(
                gaussians, cameras.pred_t_c2w, cameras.pred_t_fxfycxcy, w, h,
            )

        # gaussians/cameras ride along for scene export; Gaussians live in the normalized camera frame.
        return edict(
            input=input_data_dict,
            target=target_data_dict,
            render=renderings,
            gaussians=gaussians,
            cameras=cameras,
        )

    def forward(self, input_data_dict: dict, target_data_dict: dict) -> edict:
        """Predict cameras -> build ray maps -> predict Gaussians -> render (mode-dependent edict)."""
        cameras = self.geometry_expert.predict_cameras(input_data_dict, target_data_dict)
        raymap_images, Ks, i_w2c = self._build_raymap_input(
            input_data_dict, cameras.pred_i_fxfycxcy, cameras.pred_i_c2w,
        )
        gaussians = self.appearance_expert.predict_gaussians(
            raymap_images, i_w2c, Ks,
            cameras.pred_i_fxfycxcy, cameras.pred_i_c2w, cameras.pred_t_c2w,
        )

        if not self.inference_mode:
            return self._training_forward(gaussians, cameras, target_data_dict, input_data_dict)
        return self._inference_forward(gaussians, cameras, target_data_dict, input_data_dict)

    def pose_optimization(
        self,
        gaussians: GaussianField,
        t_c2w: torch.Tensor,
        t_fxfycxcy: torch.Tensor,
        w: int,
        h: int,
        target_images: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """EPA (evaluation-time pose alignment): ``epa_steps`` Adam steps on the target
        quaternions + translations against an MSE loss, Gaussians frozen. Takes t_c2w (1, T, 4, 4),
        t_fxfycxcy (1, T, 4), target_images (1, T, 3, H, W); returns (renderings (1, T, 3, H, W), (c2w, fxfycxcy))."""
        num_target_views = t_c2w.shape[1]
        quats = mat_to_quat(t_c2w[..., :3, :3]).clone().detach().requires_grad_(True)
        trans = t_c2w[..., :3, 3].clone().detach().requires_grad_(True)
        t_fxfycxcy = t_fxfycxcy.clone().detach()
        xyz = gaussians.xyz[0].detach()
        feature = gaussians.feature[0].detach()
        scale = gaussians.scale[0].detach()
        rotation = gaussians.rotation[0].detach()
        opacity = gaussians.opacity[0].detach()
        optimizer = torch.optim.Adam([quats, trans], lr=1e-4)

        for _ in range(self.epa_steps):
            R = quat_to_mat(quats / quats.norm(dim=-1, keepdim=True))
            t_c2w_new = torch.zeros_like(t_c2w)
            t_c2w_new[..., :3, :3] = R
            t_c2w_new[..., :3, 3] = trans
            t_c2w_new[..., 3, 3] = 1.0

            optimizer.zero_grad()
            renderings = []
            with torch.autocast(device_type="cuda", enabled=False):
                for i in range(num_target_views):
                    dir = xyz - t_c2w_new[0, i:i+1, :3, 3][None, ...]
                    opacity_i = _spherical_harmonics(
                        self.opacity_degree, dir, opacity[None, ...],
                    )[0]
                    rendering = GaussianRenderer.render(
                        xyz, feature, scale, rotation, opacity_i,
                        t_c2w_new[0, i], t_fxfycxcy[0, i], w, h,
                        self.sh_degree, self.near_plane, self.far_plane,
                    )
                    renderings.append(rendering)
            renderings = torch.cat(renderings, dim=0)[None, ...]

            loss = F.mse_loss(renderings.permute(0, 1, 4, 2, 3).contiguous(), target_images)
            loss.backward()
            optimizer.step()

        refined_c2w = t_c2w_new.detach()
        refined_fxfycxcy = t_fxfycxcy.detach()
        return renderings.permute(0, 1, 4, 2, 3).contiguous(), (refined_c2w, refined_fxfycxcy)

    def _expand_gaussian_head(self, state_dict: dict) -> dict:
        """Convert a pretrained ``gaussian_decoder`` head to this model's layout: mean-pool a
        denser patch grid to this sub-grid, replace the xyz rows by depth (mean of xyz) + zero
        image_offset rows, and zero-pad SH0 to SH1. Converted checkpoints pass through."""
        key = "gaussian_decoder.1.weight"
        w = state_dict.get(key, None)
        if w is None:
            return state_dict
        # Pixel xyz and depth + image_offset both use three rows; the marker tells them apart.
        image_offset = "_image_offset_version" in state_dict
        if image_offset and w.shape == self.gaussian_decoder[1].weight.shape:
            return state_dict
        rows = w.shape[0]
        # Accept the target color layout or an SH0 head when upgrading to SH1.
        color_dims = (self.color_dim, 3) if self.color_dim == 12 else (self.color_dim,)
        for color_dim_old in color_dims:
            d_old = 3 + color_dim_old + 3 + 4 + self.opacity_dim
            p_old = rows // d_old
            side_old = int(round(p_old ** 0.5))
            if (rows % d_old == 0 and side_old > 0
                    and side_old ** 2 == p_old
                    and side_old % self.gaussian_side == 0):
                break
        else:
            raise ValueError(
                "Expected a pixel-aligned MVP checkpoint or a marked image-offset "
                "checkpoint with matching SH degree or SH0 for SH1 conversion; "
                "legacy world-space offsets require explicit migration."
            )
        k = side_old // self.gaussian_side

        # Rows are (ph, pw, d) over the pretrained grid; pool each k x k sub-block (k == 1: no-op).
        w = rearrange(
            w, "(ah kh aw kw d) D -> ah aw (kh kw) d D",
            ah=self.gaussian_side, kh=k,
            aw=self.gaussian_side, kw=k, d=d_old,
        ).mean(dim=2)
        position_w, color_w, rest_w = torch.split(
            w, [3, color_dim_old, 3 + 4 + self.opacity_dim], dim=-2,
        )
        if not image_offset:
            position_w = torch.cat(
                [position_w.mean(dim=-2, keepdim=True),
                 torch.zeros_like(position_w[..., :2, :])], dim=-2,
            )
        if color_dim_old != self.color_dim:
            # Forward stores coefficients per RGB channel: R[SH], G[SH], B[SH].
            color_w = rearrange(color_w, "ah aw (c d) D -> ah aw c d D", c=3)
            color_w = torch.cat(
                [color_w,
                 color_w.new_zeros(*color_w.shape[:-2],
                                   (self.color_dim - color_dim_old) // 3,
                                   color_w.shape[-1])], dim=-2,
            )
            color_w = rearrange(color_w, "ah aw c d D -> ah aw (c d) D")
        w = torch.cat([position_w, color_w, rest_w], dim=-2)
        # Copy, keeping the _metadata that nn.Module.load_state_dict reads.
        converted = OrderedDict(state_dict)
        converted._metadata = getattr(state_dict, "_metadata", OrderedDict())
        state_dict = converted
        state_dict[key] = rearrange(w, "ah aw d D -> (ah aw d) D")
        state_dict["_image_offset_version"] = self._image_offset_version.clone()
        print(f"Converted pretrained {key}: {p_old} -> {self.gaussians_per_patch} "
              f"gaussians/patch, depth + image_offset layout, "
              f"{color_dim_old} -> {self.color_dim} color channels")
        return state_dict

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Convert pixel-aligned checkpoints on the fly, then load normally."""
        return super().load_state_dict(
            self._expand_gaussian_head(state_dict), strict=strict, assign=assign,
        )

    @torch.no_grad()
    def load_ckpt(self, load_path: str) -> int | None:
        """Load the ``"ema"`` state dict from a .pt file or the last .pt in a directory
        (strict=False: DA3 weights are loaded separately). Returns 0, or None on failure."""
        if os.path.isdir(load_path):
            ckpt_names = sorted(f for f in os.listdir(load_path) if f.endswith(".pt"))
            ckpt_path = os.path.join(load_path, ckpt_names[-1])
        else:
            ckpt_path = load_path
        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except:
            traceback.print_exc()
            print(f"Failed to load {ckpt_path}")
            return None
        result = self.load_state_dict(checkpoint["ema"], strict=False)
        print(f"Loaded 2Xplat checkpoint in {ckpt_path}")
        return 0
