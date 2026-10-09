import torch
from typing import Any
from dataclasses import dataclass
from gsplat import rasterization

from src.utils.camera_utils import fxfycxcy_to_K


# ---------------------------------------------------------------------------
# Structured output types
# ---------------------------------------------------------------------------

@dataclass
class GaussianField:
    xyz:                torch.Tensor         # (B, N, 3)
    feature:            torch.Tensor         # (B, N, D, 3) — SH color coefficients
    scale:              torch.Tensor         # (B, N, 3)
    rotation:           torch.Tensor         # (B, N, 4)
    opacity:            torch.Tensor         # (B, N, D, 1) — SH opacity coefficients
    opacity_precompute: torch.Tensor | None  # (B, T, N, D, 1) — None during inference


# ---------------------------------------------------------------------------
# Differentiable Gaussian rasterizer
# ---------------------------------------------------------------------------

class GaussianRenderer(torch.autograd.Function):
    """Differentiable rasterizer for 3D Gaussian splats using gsplat.

    Implements a custom autograd Function so that gradients flow through
    gsplat's rasterization kernel per-view, enabling end-to-end training
    of Gaussian parameters from photometric losses.
    """

    @staticmethod
    def render(
        xyz: torch.Tensor,
        feature: torch.Tensor,
        scale: torch.Tensor,
        rotation: torch.Tensor,
        opacity: torch.Tensor,
        test_c2w: torch.Tensor,
        test_intr: torch.Tensor,
        W: int,
        H: int,
        sh_degree: int,
        near_plane: float,
        far_plane: float,
        render_mode: str = "RGB",
    ) -> torch.Tensor:
        """Rasterize a single view of a 3D Gaussian scene.

        Args:
            xyz: Gaussian center positions of shape (N, 3).
            feature: SH color coefficients of shape (N, D, 3).
            scale: Log-space Gaussian scales of shape (N, 3); exponentiated internally.
            rotation: Unit quaternions of shape (N, 4).
            opacity: Pre-sigmoid SH opacity coefficients of shape (N, D, 1);
                sigmoid-activated and squeezed internally.
            test_c2w: Camera-to-world transform of shape (4, 4).
            test_intr: Camera intrinsics as (fx, fy, cx, cy) of shape (4,).
            W: Output image width in pixels.
            H: Output image height in pixels.
            sh_degree: Maximum spherical harmonics degree for color evaluation.
            near_plane: Near clipping distance.
            far_plane: Far clipping distance.
            render_mode: gsplat render mode. "RGB" (default) or "RGB+ED", which
                appends the expected-depth channel used for occlusion tests.

        Returns:
            Rendered image of shape (1, H, W, 3), or (1, H, W, 4) for "RGB+ED".
        """
        opacity = opacity.sigmoid().squeeze(-1)
        scale = scale.exp()
        test_w2c = test_c2w.float().inverse().unsqueeze(0)
        test_intr_i = fxfycxcy_to_K(test_intr).unsqueeze(0)
        # backgrounds stays RGB-only: gsplat appends the depth channel itself.
        rendering, _, _ = rasterization(
            xyz, rotation, scale, opacity, feature,
            test_w2c, test_intr_i, W, H,
            sh_degree=sh_degree,
            near_plane=near_plane,
            far_plane=far_plane,
            packed=False,
            absgrad=False,
            sparse_grad=False,
            render_mode=render_mode,
            backgrounds=torch.ones(1, 3).to(test_intr.device),
            rasterize_mode='classic',
        )
        return rendering  # (1, H, W, 3), or (1, H, W, 4) with "RGB+ED"

    @staticmethod
    def forward(
        ctx: Any,
        xyz: torch.Tensor,
        feature: torch.Tensor,
        scale: torch.Tensor,
        rotation: torch.Tensor,
        opacity: torch.Tensor,
        test_c2ws: torch.Tensor,
        test_intr: torch.Tensor,
        W: int,
        H: int,
        sh_degree: int,
        near_plane: float,
        far_plane: float,
    ) -> torch.Tensor:
        """Render all target views for a batch of Gaussian scenes.

        Iterates over batch and view dimensions, calling render() for each
        (batch, view) pair under torch.no_grad(), then re-attaches gradients
        to the output tensor so that backward() can replay the computation.

        Args:
            ctx: Autograd context used to stash tensors for backward.
            xyz: Gaussian centers of shape (B, N, 3).
            feature: SH color coefficients of shape (B, N, D, 3).
            scale: Log-space scales of shape (B, N, 3).
            rotation: Unit quaternions of shape (B, N, 4).
            opacity: Pre-sigmoid SH opacity of shape (B, V, N, D, 1).
            test_c2ws: Camera-to-world transforms of shape (B, V, 4, 4).
            test_intr: Camera intrinsics (fx, fy, cx, cy) of shape (B, V, 4).
            W: Output image width in pixels.
            H: Output image height in pixels.
            sh_degree: Maximum spherical harmonics degree.
            near_plane: Near clipping distance.
            far_plane: Far clipping distance.

        Returns:
            Rendered RGB images of shape (B, V, H, W, 3).
        """
        ctx.save_for_backward(xyz, feature, scale, rotation, opacity, test_c2ws, test_intr)
        ctx.W = W
        ctx.H = H
        ctx.sh_degree = sh_degree
        ctx.near_plane = near_plane
        ctx.far_plane = far_plane
        with torch.no_grad():
            B, V, _ = test_intr.shape
            renderings = torch.zeros(B, V, H, W, 3).to(xyz.device)
            for ib in range(B):
                for iv in range(V):
                    renderings[ib, iv:iv+1] = GaussianRenderer.render(
                        xyz[ib], feature[ib], scale[ib], rotation[ib], opacity[ib, iv],
                        test_c2ws[ib, iv], test_intr[ib, iv],
                        W, H, sh_degree, near_plane, far_plane,
                    )
        renderings = renderings.requires_grad_()
        return renderings

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        """Recompute per-view renders under enable_grad to accumulate parameter gradients.

        Because gsplat's rasterization kernel does not natively support batched
        backward, we replay each (batch, view) render with gradients enabled and
        call backward() with the corresponding output gradient slice, letting
        PyTorch accumulate gradients into each leaf tensor.

        Args:
            ctx: Autograd context carrying saved tensors and scalar attributes.
            grad_output: Upstream gradient of shape (B, V, H, W, 3).

        Returns:
            Tuple of gradients matching the forward() signature:
            (grad_xyz, grad_feature, grad_scale, grad_rotation, grad_opacity,
            grad_test_c2ws, grad_test_intr, None, None, None, None, None).
        """
        xyz, feature, scale, rotation, opacity, test_c2ws, test_intr = ctx.saved_tensors
        xyz = xyz.detach().requires_grad_()
        feature = feature.detach().requires_grad_()
        scale = scale.detach().requires_grad_()
        rotation = rotation.detach().requires_grad_()
        opacity = opacity.detach().requires_grad_()
        test_c2ws = test_c2ws.detach().requires_grad_()
        test_intr = test_intr.detach().requires_grad_()
        W = ctx.W
        H = ctx.H
        sh_degree = ctx.sh_degree
        near_plane = ctx.near_plane
        far_plane = ctx.far_plane
        with torch.enable_grad():
            B, V, _ = test_intr.shape
            for ib in range(B):
                for iv in range(V):
                    rendering = GaussianRenderer.render(
                        xyz[ib], feature[ib], scale[ib], rotation[ib], opacity[ib, iv],
                        test_c2ws[ib, iv], test_intr[ib, iv],
                        W, H, sh_degree, near_plane, far_plane,
                    )
                    rendering.backward(grad_output[ib, iv:iv+1])
        return (
            xyz.grad, feature.grad, scale.grad, rotation.grad, opacity.grad,
            test_c2ws.grad, test_intr.grad,
            None, None, None, None, None,
        )
