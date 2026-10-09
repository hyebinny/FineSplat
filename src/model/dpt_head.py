import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from einops import rearrange


# ---------------------------------------------------------------------------
# Positional embedding utilities
# ---------------------------------------------------------------------------

def create_uv_grid(
    width: int, height: int, aspect_ratio: float = None, dtype: torch.dtype = None, device: torch.device = None
) -> torch.Tensor:
    """
    Create a normalized UV grid of shape (width, height, 2).

    The grid spans horizontally and vertically according to an aspect ratio,
    ensuring the top-left corner is at (-x_span, -y_span) and the bottom-right
    corner is at (x_span, y_span), normalized by the diagonal of the plane.

    Args:
        width (int): Number of points horizontally.
        height (int): Number of points vertically.
        aspect_ratio (float, optional): Width-to-height ratio. Defaults to width/height.
        dtype (torch.dtype, optional): Data type of the resulting tensor.
        device (torch.device, optional): Device on which the tensor is created.

    Returns:
        torch.Tensor: A (width, height, 2) tensor of UV coordinates.
    """
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)

    diag_factor = (aspect_ratio**2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor

    left_x = -span_x * (width - 1) / width
    right_x = span_x * (width - 1) / width
    top_y = -span_y * (height - 1) / height
    bottom_y = span_y * (height - 1) / height

    x_coords = torch.linspace(left_x, right_x, steps=width, dtype=dtype, device=device)
    y_coords = torch.linspace(top_y, bottom_y, steps=height, dtype=dtype, device=device)

    uu, vv = torch.meshgrid(x_coords, y_coords, indexing="xy")
    uv_grid = torch.stack((uu, vv), dim=-1)

    return uv_grid


def make_sincos_pos_embed(embed_dim: int, pos: torch.Tensor, omega_0: float = 100) -> torch.Tensor:
    """Generate a 1D sinusoidal positional embedding from a position tensor.

    Args:
        embed_dim (int): Output embedding dimension; must be even.
        pos (torch.Tensor): Position values of arbitrary shape, flattened to (M,).
        omega_0 (float): Base frequency scale. Higher values compress the frequency
            spectrum, giving finer-grained positional distinctions.

    Returns:
        torch.Tensor: Sinusoidal embeddings of shape (M, embed_dim) as float32.
    """
    assert embed_dim % 2 == 0
    device = pos.device
    omega = torch.arange(embed_dim // 2, dtype=torch.float32 if device.type == "mps" else torch.double, device=device)
    omega /= embed_dim / 2.0
    omega = 1.0 / omega_0**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb.float()


def position_grid_to_embed(pos_grid: torch.Tensor, embed_dim: int, omega_0: float = 100) -> torch.Tensor:
    """Convert a 2D position grid to sinusoidal embeddings.

    Args:
        pos_grid (torch.Tensor): UV coordinates of shape (H, W, 2).
        embed_dim (int): Total output embedding dimension; must be even so that
            half is allocated to each spatial axis.
        omega_0 (float): Base frequency scale forwarded to make_sincos_pos_embed.

    Returns:
        torch.Tensor: Sinusoidal embeddings of shape (H, W, embed_dim).
    """
    H, W, grid_dim = pos_grid.shape
    assert grid_dim == 2
    pos_flat = pos_grid.reshape(-1, grid_dim)  # (H*W, 2)

    emb_x = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 0], omega_0=omega_0)  # (H*W, D/2)
    emb_y = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 1], omega_0=omega_0)  # (H*W, D/2)

    emb = torch.cat([emb_x, emb_y], dim=-1)  # (H*W, D)
    return emb.view(H, W, embed_dim)  # (H, W, D)


# ---------------------------------------------------------------------------
# Interpolation helper
# ---------------------------------------------------------------------------

def custom_interpolate(
    x: torch.Tensor,
    size: Tuple[int, int] = None,
    scale_factor: float = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    """Chunked interpolation that avoids INT_MAX overflow in nn.functional.interpolate.

    When the total number of output elements exceeds INT_MAX, the batch dimension
    is split into smaller chunks that are each interpolated independently and then
    concatenated, sidestepping the 32-bit index limit in the underlying CUDA kernel.

    Args:
        x (torch.Tensor): Input feature map of shape (B, C, H_in, W_in).
        size (Tuple[int, int], optional): Target spatial size (H_out, W_out).
            Derived from scale_factor if not provided.
        scale_factor (float, optional): Multiplier for H and W when size is None.
        mode (str): Interpolation algorithm passed to nn.functional.interpolate.
        align_corners (bool): Passed to nn.functional.interpolate.

    Returns:
        torch.Tensor: Interpolated tensor of shape (B, C, H_out, W_out).
    """
    if size is None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))

    INT_MAX = 1610612736
    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]

    if input_elements > INT_MAX:
        chunks = torch.chunk(x, chunks=(input_elements // INT_MAX) + 1, dim=0)
        interpolated_chunks = [
            nn.functional.interpolate(chunk, size=size, mode=mode, align_corners=align_corners) for chunk in chunks
        ]
        x = torch.cat(interpolated_chunks, dim=0)
        return x.contiguous()
    else:
        return nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners)


# ---------------------------------------------------------------------------
# Fusion block modules
# ---------------------------------------------------------------------------

class ResidualConvUnit(nn.Module):
    """Two-layer residual convolution block."""

    def __init__(self, features, activation, bn, groups=1):
        """Initialize conv layers, optional batch norm, and the activation function."""
        super().__init__()

        self.bn = bn
        self.groups = groups
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        self.norm1 = None
        self.norm2 = None

        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Apply two conv layers with activation and add the input as a residual.

        Args:
            x (torch.Tensor): Input feature map of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output feature map of shape (B, C, H, W).
        """
        out = self.activation(x)
        out = self.conv1(out)
        if self.norm1 is not None:
            out = self.norm1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Fuse two feature maps with optional residual refinement and upsampling."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=None,
        has_residual=True,
        groups=1,
    ):
        """Initialize projection conv, optional residual unit, and refinement unit."""
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = groups
        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=self.groups
        )

        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, activation, bn, groups=self.groups)

        self.has_residual = has_residual
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn, groups=self.groups)

        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size=None):
        """Fuse input feature maps, optionally upsample, and apply output projection.

        Args:
            *xs: One or two feature maps. When has_residual is True, xs[1] is added
                to xs[0] after passing through resConfUnit1. Shape: (B, C, H, W).
            size (Tuple[int, int], optional): Target spatial size for bilinear
                upsampling applied before the output conv.

        Returns:
            torch.Tensor: Fused feature map of shape (B, out_features, H_out, W_out).
        """
        output = xs[0]

        if self.has_residual:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if size is not None:
            modifier = {"size": size}
            output = custom_interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)

        return output


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _make_fusion_block(features: int, size: int = None, has_residual: bool = True, groups: int = 1) -> nn.Module:
    """Construct a FeatureFusionBlock with standard settings (ReLU, no BN, no expand).

    Args:
        features (int): Number of channels throughout the fusion block.
        size (int, optional): Fixed target upsampling size passed to the block.
        has_residual (bool): Whether to include the first residual conv unit.
        groups (int): Number of groups for grouped convolutions.

    Returns:
        nn.Module: A configured FeatureFusionBlock instance.
    """
    return FeatureFusionBlock(
        features,
        nn.ReLU(inplace=True),
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=size,
        has_residual=has_residual,
        groups=groups,
    )


def _make_scratch(in_shape: List[int], out_shape: int, groups: int = 1) -> nn.Module:
    """Build an nn.Module holding three 3×3 projection convolutions for the scratch layers.

    Args:
        in_shape (List[int]): Input channel counts for each of the three layers.
        out_shape (int): Uniform output channel count for all three convolutions.
        groups (int): Number of groups for grouped convolutions.

    Returns:
        nn.Module: Module with attributes layer1_rn, layer2_rn, and layer3_rn.
    """
    scratch = nn.Module()
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    return scratch


# ---------------------------------------------------------------------------
# DPT Head
# ---------------------------------------------------------------------------

class DPTHead(nn.Module):
    """
    DPT Head for dense prediction tasks.

    Follows the architecture from "Vision Transformers for Dense Prediction"
    (https://arxiv.org/abs/2103.13413). Fuses multi-scale patch tokens from
    three transformer stages into a single dense feature map.

    Args:
        dim_in (List[int]): Input channel dimensions for each stage. Default is [256, 512, 1024].
        features (int): Channel width used throughout the fusion network. Default is 1024.
        out_channels (List[int]): Per-stage projection output channels. Default is [256, 512, 1024].
    """

    def __init__(
        self,
        dim_in: List[int] = [256, 512, 1024],
        features: int = 1024,
        out_channels: List[int] = [256, 512, 1024],
    ) -> None:
        """Initialize layer norms, scratch projections, refinenet blocks, and output conv."""
        super(DPTHead, self).__init__()

        self.norm1 = nn.LayerNorm(dim_in[0])
        self.norm2 = nn.LayerNorm(dim_in[1])
        self.norm3 = nn.LayerNorm(dim_in[2])

        self.scratch = _make_scratch(out_channels, features)

        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features, has_residual=False)

        self.scratch.output_conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        image_sizes: List[int],
        patch_size: int,
    ) -> torch.Tensor:
        """
        Reshape and fuse multi-scale tokens, then return dense token sequences.

        Args:
            aggregated_tokens_list (List[Tensor]): Patch tokens from each stage,
                shapes [(B, H1*W1, C1), (B, H2*W2, C2), (B, H3*W3, C3)].
            image_sizes (List[int]): [H, W] of the input image in pixels.
            patch_size (int): Patch size used during tokenization.

        Returns:
            Tensor: Fused features with shape (B, H1*W1, features).
        """
        H, W = image_sizes

        # Derive spatial grid sizes for each stage (each halved relative to the previous).
        hh1 = H // patch_size
        ww1 = W // patch_size
        hh2 = hh1 // 2
        ww2 = ww1 // 2
        hh3 = hh2 // 2
        ww3 = ww2 // 2

        # Reshape tokens to spatial feature maps and inject positional embeddings.
        out = []
        x1 = self.norm1(aggregated_tokens_list[0])
        x1 = x1.permute(0, 2, 1).reshape((x1.shape[0], x1.shape[-1], hh1, ww1))
        x1 = self._apply_pos_embed(x1, W, H)
        out.append(x1)

        x2 = self.norm2(aggregated_tokens_list[1])
        x2 = x2.permute(0, 2, 1).reshape((x2.shape[0], x2.shape[-1], hh2, ww2))
        x2 = self._apply_pos_embed(x2, W, H)
        out.append(x2)

        x3 = self.norm3(aggregated_tokens_list[2])
        x3 = x3.permute(0, 2, 1).reshape((x3.shape[0], x3.shape[-1], hh3, ww3))
        x3 = self._apply_pos_embed(x3, W, H)
        out.append(x3)

        out = self.scratch_forward(out)
        out = self._apply_pos_embed(out, W, H)

        return rearrange(out, "b c h w -> b (h w) c")

    def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        """Add a sinusoidal UV positional embedding scaled by `ratio`.

        Args:
            x (torch.Tensor): Feature map of shape (B, C, H_patch, W_patch).
            W (int): Original image width in pixels, used to compute aspect ratio.
            H (int): Original image height in pixels, used to compute aspect ratio.
            ratio (float): Scale factor applied to the positional embedding before
                addition, controlling the relative influence of position information.

        Returns:
            torch.Tensor: Feature map with positional embedding added, same shape as x.
        """
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed

    def scratch_forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Project and fuse three feature maps coarse-to-fine via refinenet blocks.

        Args:
            features (List[torch.Tensor]): Feature maps at three scales
                [fine, medium, coarse], each of shape (B, C_i, H_i, W_i).

        Returns:
            torch.Tensor: Fused feature map at the finest resolution,
                shape (B, features, H1, W1).
        """
        layer_1, layer_2, layer_3 = features

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)

        # Upsample and fuse from coarsest to finest resolution.
        out = self.scratch.refinenet3(layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3

        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2

        out = self.scratch.refinenet1(out, layer_1_rn)
        del layer_1_rn, layer_1

        out = self.scratch.output_conv1(out)
        return out
