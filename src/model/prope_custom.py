# Copyright (c) Authors of "Cameras as Relative Positional Encoding" https://arxiv.org/pdf/2507.10496

import torch
import torch.nn.functional as F
from functools import partial
from typing import Callable, Optional, Tuple, List

from src.utils.camera_utils import lift_K, invert_SE3, invert_K


# ---------------------------------------------------------------------------
# Low-level RoPE and projection matrix application functions
# ---------------------------------------------------------------------------

def _rope_precompute_coeffs(
    positions: torch.Tensor,  # (seqlen,)
    freq_base: float,
    freq_scale: float,
    feat_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute (cos, sin) RoPE coefficients for a sequence of positions."""
    assert len(positions.shape) == 1
    assert feat_dim % 2 == 0
    num_freqs = feat_dim // 2
    freqs = freq_scale * (
        freq_base
        ** (
            -torch.arange(num_freqs, device=positions.device)[None, None, None, :]
            / num_freqs
        )
    )
    angles = positions[None, None, :, None] * freqs
    # Shape: (batch, num_heads, seqlen, num_freqs) — broadcast across batch and heads.
    assert angles.shape == (1, 1, positions.shape[0], num_freqs)
    return torch.cos(angles), torch.sin(angles)


def _rope_apply_coeffs(
    feats: torch.Tensor,  # (batch, num_heads, seqlen, feat_dim)
    coeffs: Tuple[torch.Tensor, torch.Tensor],
    inverse: bool = False,
) -> torch.Tensor:
    """Apply RoPE coefficients to features using 'split' (not interleaved) ordering."""
    cos, sin = coeffs
    # Allow (cos, sin) to be per-image (seqlen_per_image) and tile to match feats.
    if cos.shape[2] != feats.shape[2]:
        n_repeats = feats.shape[2] // cos.shape[2]
        cos = cos.repeat(1, 1, n_repeats, 1)
        sin = sin.repeat(1, 1, n_repeats, 1)
    assert len(feats.shape) == len(cos.shape) == len(sin.shape) == 4
    assert cos.shape[-1] == sin.shape[-1] == feats.shape[-1] // 2

    x_in = feats[..., : feats.shape[-1] // 2]
    y_in = feats[..., feats.shape[-1] // 2 :]
    return torch.cat(
        (
            [cos * x_in + sin * y_in, -sin * x_in + cos * y_in]
            if not inverse
            else [cos * x_in - sin * y_in, sin * x_in + cos * y_in]
        ),
        dim=-1,
    )


def _apply_block_diagonal(
    feats: torch.Tensor,  # (..., dim)
    func_size_pairs: List[Tuple[Callable[[torch.Tensor], torch.Tensor], int]],
) -> torch.Tensor:
    """Apply a block-diagonal function to an input array.

    Each function is specified as a tuple with form:

        ((Tensor) -> Tensor, int)

    Where the integer is the size of the input to the function.
    """
    funcs, block_sizes = zip(*func_size_pairs)
    assert feats.shape[-1] == sum(block_sizes)
    x_blocks = torch.split(feats, block_sizes, dim=-1)
    out = torch.cat(
        [f(x_block) for f, x_block in zip(funcs, x_blocks)],
        dim=-1,
    )
    assert out.shape == feats.shape, "Input/output shapes should match."
    return out


def _apply_tiled_projmat(
    feats: torch.Tensor,  # (batch, num_heads, seqlen, feat_dim)
    matrix: torch.Tensor,  # (batch, cameras, D, D)
) -> torch.Tensor:
    """Apply a per-camera projection matrix to features tiled across patches."""
    # seqlen => (cameras, patches_x * patches_y)
    # feat_dim => (feat_dim // D, D)
    (batch, num_heads, seqlen, feat_dim) = feats.shape
    cameras = matrix.shape[1]
    assert seqlen > cameras and seqlen % cameras == 0
    D = matrix.shape[-1]
    assert matrix.shape == (batch, cameras, D, D)
    assert feat_dim % D == 0

    return torch.einsum(
        "bcij,bncpkj->bncpki",
        matrix,
        feats.reshape((batch, num_heads, cameras, -1, feat_dim // D, D)),
    ).reshape(feats.shape)


# ---------------------------------------------------------------------------
# PRoPE transform builders
# ---------------------------------------------------------------------------

def _prepare_apply_fns(
    head_dim: int,
    viewmats: Optional[torch.Tensor],   # (batch, cameras, 4, 4)
    Ks: Optional[torch.Tensor],         # (batch, cameras, 3, 3)
    patches_x: int,
    patches_y: int,
    image_width: int,
    image_height: int,
    coeffs_x: Optional[torch.Tensor] = None,
    coeffs_y: Optional[torch.Tensor] = None,
) -> Tuple[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
]:
    """Build block-diagonal transforms (q, kv, o) for PRoPE-style positional encoding."""
    device = viewmats.device
    (batch, cameras, _, _) = viewmats.shape

    if Ks is not None:
        # Normalize intrinsics to [-0.5, 0.5] × [-0.5, 0.5] image-plane coordinates.
        Ks_norm = torch.zeros_like(Ks)
        Ks_norm[..., 0, 0] = Ks[..., 0, 0] / image_width
        Ks_norm[..., 1, 1] = Ks[..., 1, 1] / image_height
        Ks_norm[..., 0, 2] = Ks[..., 0, 2] / image_width - 0.5
        Ks_norm[..., 1, 2] = Ks[..., 1, 2] / image_height - 0.5
        Ks_norm[..., 2, 2] = 1.0
        del Ks

        # Compute PRoPE projection matrices:
        #   K  = image←camera,  viewmats = camera←world
        #   P  = lift(K) @ viewmats  (image←world)
        P = torch.einsum("...ij,...jk->...ik", lift_K(Ks_norm), viewmats)
        P_T = P.transpose(-1, -2)
        P_inv = torch.einsum(
            "...ij,...jk->...ik",
            invert_SE3(viewmats),
            lift_K(invert_K(Ks_norm)),
        )
    else:
        # GTA formula: P is the camera←world transform.
        P = viewmats
        P_T = P.transpose(-1, -2)
        P_inv = invert_SE3(viewmats)

    assert P.shape == P_inv.shape == (batch, cameras, 4, 4)

    # Precompute RoPE cos/sin terms (row-major patch ordering).
    if coeffs_x is None:
        coeffs_x = _rope_precompute_coeffs(
            torch.tile(torch.arange(patches_x, device=device), (patches_y * cameras,)),
            freq_base=100.0,
            freq_scale=1.0,
            feat_dim=head_dim // 4,
        )
    if coeffs_y is None:
        coeffs_y = _rope_precompute_coeffs(
            torch.tile(
                torch.repeat_interleave(
                    torch.arange(patches_y, device=device), patches_x
                ),
                (cameras,),
            ),
            freq_base=100.0,
            freq_scale=1.0,
            feat_dim=head_dim // 4,
        )

    assert head_dim % 4 == 0
    transforms_q = [
        (partial(_apply_tiled_projmat, matrix=P_T), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x), head_dim // 4),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y), head_dim // 4),
    ]
    transforms_kv = [
        (partial(_apply_tiled_projmat, matrix=P_inv), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x), head_dim // 4),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y), head_dim // 4),
    ]
    transforms_o = [
        (partial(_apply_tiled_projmat, matrix=P), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x, inverse=True), head_dim // 4),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y, inverse=True), head_dim // 4),
    ]

    apply_fn_q = partial(_apply_block_diagonal, func_size_pairs=transforms_q)
    apply_fn_kv = partial(_apply_block_diagonal, func_size_pairs=transforms_kv)
    apply_fn_o = partial(_apply_block_diagonal, func_size_pairs=transforms_o)
    return apply_fn_q, apply_fn_kv, apply_fn_o


def _prepare_apply_fns_rope(
    head_dim: int,
    coeffs_x: Optional[torch.Tensor] = None,
    coeffs_y: Optional[torch.Tensor] = None,
) -> Tuple[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
]:
    """Build (q, kv) RoPE-only transforms (no camera projection — used when viewmats is None)."""
    assert head_dim % 2 == 0
    transforms_q = [
        (partial(_rope_apply_coeffs, coeffs=coeffs_x), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y), head_dim // 2),
    ]
    transforms_kv = [
        (partial(_rope_apply_coeffs, coeffs=coeffs_x), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y), head_dim // 2),
    ]

    apply_fn_q = partial(_apply_block_diagonal, func_size_pairs=transforms_q)
    apply_fn_kv = partial(_apply_block_diagonal, func_size_pairs=transforms_kv)
    return apply_fn_q, apply_fn_kv


# ---------------------------------------------------------------------------
# PRoPE attention — public API
# ---------------------------------------------------------------------------

def prope_dot_product_attention(
    q: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    k: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    v: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    *,
    viewmats: Optional[torch.Tensor],           # (batch, cameras, 4, 4)
    Ks: Optional[torch.Tensor],                 # (batch, cameras, 3, 3)
    patches_x: int,
    patches_y: int,
    image_width: int,
    image_height: int,
    num_register_tokens: int,
    coeffs_x: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    coeffs_y: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> torch.Tensor:
    """PRoPE-style scaled dot-product attention.

    Similar to ``torch.nn.functional.scaled_dot_product_attention`` but applies
    camera-conditioned positional encoding.  Sequence length must equal:

        cameras * patches_x * patches_y  (+  cameras * num_register_tokens  if applicable)

    Token ordering must allow the ``(seqlen,)`` axis to be reshaped into
    ``(cameras, patches_x, patches_y)``.
    """
    (batch, num_heads, seqlen, head_dim) = q.shape

    if viewmats is not None:
        cameras = viewmats.shape[1]
        assert q.shape == k.shape == v.shape
        assert viewmats.shape == (batch, cameras, 4, 4)
        assert Ks is None or Ks.shape == (batch, cameras, 3, 3)
        assert seqlen == cameras * patches_x * patches_y + cameras * num_register_tokens

        apply_fn_q, apply_fn_kv, apply_fn_o = _prepare_apply_fns(
            head_dim=head_dim,
            viewmats=viewmats,
            Ks=Ks,
            patches_x=patches_x,
            patches_y=patches_y,
            image_width=image_width,
            image_height=image_height,
            coeffs_x=coeffs_x,
            coeffs_y=coeffs_y,
        )
        out = F.scaled_dot_product_attention(
            query=apply_fn_q(q),
            key=apply_fn_kv(k),
            value=apply_fn_kv(v),
            **kwargs,
        )
        out = apply_fn_o(out)
    else:
        apply_fn_q, apply_fn_kv = _prepare_apply_fns_rope(
            head_dim=head_dim,
            coeffs_x=coeffs_x,
            coeffs_y=coeffs_y,
        )
        out = F.scaled_dot_product_attention(
            query=apply_fn_q(q),
            key=apply_fn_kv(k),
            value=v,
            **kwargs,
        )

    assert out.shape == (batch, num_heads, seqlen, head_dim)
    return out


class PropeDotProductAttention(torch.nn.Module):
    """PRoPE attention module with precomputed and cached RoPE coefficients."""

    coeffs_x_0: torch.Tensor
    coeffs_x_1: torch.Tensor
    coeffs_y_0: torch.Tensor
    coeffs_y_1: torch.Tensor

    def __init__(
        self,
        head_dim: int,
        patches_x: int,
        patches_y: int,
        image_width: int,
        image_height: int,
        num_register_tokens: int = 0,
        freq_base: float = 100.0,
        freq_scale: float = 1.0,
    ):
        """Precompute and register RoPE coefficient buffers for a fixed patch grid.

        Args:
            head_dim: Per-head feature dimension; must be divisible by 4.
            patches_x: Number of patches along the horizontal axis.
            patches_y: Number of patches along the vertical axis.
            image_width: Full image width in pixels (used to normalise intrinsics).
            image_height: Full image height in pixels.
            num_register_tokens: Number of register tokens prepended per view.
            freq_base: Base frequency for RoPE coefficient computation.
            freq_scale: Global scale applied to all RoPE frequencies.
        """
        super().__init__()
        self.head_dim = head_dim
        self.patches_x = patches_x
        self.patches_y = patches_y
        self.image_width = image_width
        self.image_height = image_height
        self.num_register_tokens = num_register_tokens

        coeffs_x_input = torch.tile(torch.arange(patches_x), (patches_y,))
        coeffs_y_input = torch.repeat_interleave(torch.arange(patches_y), patches_x)

        if num_register_tokens > 0:
            # Shift patch indices by 1; register tokens get position 0.
            coeffs_x_input = coeffs_x_input + 1
            coeffs_y_input = coeffs_y_input + 1
            pos_special_x = torch.zeros(num_register_tokens, dtype=coeffs_x_input.dtype)
            pos_special_y = torch.zeros(num_register_tokens, dtype=coeffs_y_input.dtype)
            coeffs_x_input = torch.cat([pos_special_x, coeffs_x_input])
            coeffs_y_input = torch.cat([pos_special_y, coeffs_y_input])

        # Coefficients for PRoPE (camera projection + RoPE, head_dim // 4 each).
        coeffs_x: Tuple[torch.Tensor, torch.Tensor] = _rope_precompute_coeffs(
            coeffs_x_input, freq_base=freq_base, freq_scale=freq_scale, feat_dim=head_dim // 4,
        )
        coeffs_y: Tuple[torch.Tensor, torch.Tensor] = _rope_precompute_coeffs(
            coeffs_y_input, freq_base=freq_base, freq_scale=freq_scale, feat_dim=head_dim // 4,
        )
        # Coefficients for RoPE-only path (no viewmats), head_dim // 2 each.
        coeffs_x_single: Tuple[torch.Tensor, torch.Tensor] = _rope_precompute_coeffs(
            coeffs_x_input, freq_base=freq_base, freq_scale=freq_scale, feat_dim=head_dim // 2,
        )
        coeffs_y_single: Tuple[torch.Tensor, torch.Tensor] = _rope_precompute_coeffs(
            coeffs_y_input, freq_base=freq_base, freq_scale=freq_scale, feat_dim=head_dim // 2,
        )

        # Do not save coeffs to checkpoint as `cameras` might change during testing.
        self.register_buffer("coeffs_x_0", coeffs_x[0], persistent=False)
        self.register_buffer("coeffs_x_1", coeffs_x[1], persistent=False)
        self.register_buffer("coeffs_y_0", coeffs_y[0], persistent=False)
        self.register_buffer("coeffs_y_1", coeffs_y[1], persistent=False)
        self.register_buffer("coeffs_x_single_0", coeffs_x_single[0], persistent=False)
        self.register_buffer("coeffs_x_single_1", coeffs_x_single[1], persistent=False)
        self.register_buffer("coeffs_y_single_0", coeffs_y_single[0], persistent=False)
        self.register_buffer("coeffs_y_single_1", coeffs_y_single[1], persistent=False)

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict, stripping precomputed RoPE buffers before delegation."""
        # Strip precomputed buffers — they are recomputed at init and must not be loaded
        # from checkpoints because the camera count can change between runs.
        for key in [
            "coeffs_x_0", "coeffs_x_1", "coeffs_y_0", "coeffs_y_1",
            "coeffs_x_single_0", "coeffs_x_single_1", "coeffs_y_single_0", "coeffs_y_single_1",
        ]:
            state_dict.pop(key, None)
        super().load_state_dict(state_dict, strict)

    def forward(
        self,
        q: torch.Tensor,                        # (batch, num_heads, seqlen, head_dim)
        k: torch.Tensor,                        # (batch, num_heads, seqlen, head_dim)
        v: torch.Tensor,                        # (batch, num_heads, seqlen, head_dim)
        viewmats: Optional[torch.Tensor],       # (batch, cameras, 4, 4)
        Ks: Optional[torch.Tensor],             # (batch, cameras, 3, 3)
        **kwargs,
    ) -> torch.Tensor:
        """Run PRoPE attention using cached RoPE coefficients.

        Dispatches to the full PRoPE path (camera projection + RoPE) when
        ``viewmats`` is provided, or falls back to RoPE-only when it is None.
        """
        if viewmats is None:
            return prope_dot_product_attention(
                q, k, v,
                viewmats=viewmats,
                Ks=Ks,
                patches_x=self.patches_x,
                patches_y=self.patches_y,
                image_width=self.image_width,
                image_height=self.image_height,
                num_register_tokens=self.num_register_tokens,
                coeffs_x=(self.coeffs_x_single_0, self.coeffs_x_single_1),
                coeffs_y=(self.coeffs_y_single_0, self.coeffs_y_single_1),
                **kwargs,
            )
        else:
            return prope_dot_product_attention(
                q, k, v,
                viewmats=viewmats,
                Ks=Ks,
                patches_x=self.patches_x,
                patches_y=self.patches_y,
                image_width=self.image_width,
                image_height=self.image_height,
                num_register_tokens=self.num_register_tokens,
                coeffs_x=(self.coeffs_x_0, self.coeffs_x_1),
                coeffs_y=(self.coeffs_y_0, self.coeffs_y_1),
                **kwargs,
            )

    def forward_with_kv_cache(
        self,
        q: torch.Tensor,                                # (batch, num_heads, seqlen_q, head_dim) — current chunk
        k: torch.Tensor,                                # (batch, num_heads, seqlen_q, head_dim) — current chunk
        v: torch.Tensor,                                # (batch, num_heads, seqlen_q, head_dim) — current chunk
        viewmats: torch.Tensor,                         # (batch, cameras_curr, 4, 4)
        Ks: Optional[torch.Tensor],                     # (batch, cameras_curr, 3, 3)
        cached_k_prope: Optional[torch.Tensor] = None,  # (batch, num_heads, seqlen_cached, head_dim) — already PropE-transformed
        cached_v_prope: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """PropE attention with streaming KV-cache.

        Applies PropE transforms to the current chunk's K/V only, then concatenates
        with pre-transformed cached K/V from previous chunks before running SDPA.
        Past K/V representations are frozen (causal streaming semantics).

        Returns:
            out: attention output for current Q tokens
            k_prope: post-PropE K for current chunk (store in cache)
            v_prope: post-PropE V for current chunk (store in cache)
        """
        apply_fn_q, apply_fn_kv, apply_fn_o = _prepare_apply_fns(
            head_dim=self.head_dim,
            viewmats=viewmats,
            Ks=Ks,
            patches_x=self.patches_x,
            patches_y=self.patches_y,
            image_width=self.image_width,
            image_height=self.image_height,
            coeffs_x=(self.coeffs_x_0, self.coeffs_x_1),
            coeffs_y=(self.coeffs_y_0, self.coeffs_y_1),
        )
        q_t = apply_fn_q(q)
        k_prope = apply_fn_kv(k)
        v_prope = apply_fn_kv(v)

        if cached_k_prope is not None:
            k_all = torch.cat([cached_k_prope, k_prope], dim=2)
            v_all = torch.cat([cached_v_prope, v_prope], dim=2)
        else:
            k_all, v_all = k_prope, v_prope

        out = F.scaled_dot_product_attention(query=q_t, key=k_all, value=v_all)
        out = apply_fn_o(out)  # P_{cam_q} transform — applies only to Q-length output
        return out, k_prope, v_prope

    def _precompute_and_cache_apply_fns(
        self, viewmats: torch.Tensor, Ks: Optional[torch.Tensor]
    ):
        """Compute and store PRoPE transform functions for a given camera set.

        Called once when the same cameras will be reused across multiple attention
        blocks. Results are stored as ``apply_fn_q``, ``apply_fn_kv``, and
        ``apply_fn_o`` instance attributes for use by ``_apply_to_q/kv/o``.

        Args:
            viewmats: Camera-to-world matrices (batch, cameras, 4, 4).
            Ks: Intrinsic matrices (batch, cameras, 3, 3), or None for GTA mode.
        """
        (batch, cameras, _, _) = viewmats.shape
        assert viewmats.shape == (batch, cameras, 4, 4)
        assert Ks is None or Ks.shape == (batch, cameras, 3, 3)
        self.cameras = cameras

        self.apply_fn_q, self.apply_fn_kv, self.apply_fn_o = _prepare_apply_fns(
            head_dim=self.head_dim,
            viewmats=viewmats,
            Ks=Ks,
            patches_x=self.patches_x,
            patches_y=self.patches_y,
            image_width=self.image_width,
            image_height=self.image_height,
            coeffs_x=(self.coeffs_x_0, self.coeffs_x_1),
            coeffs_y=(self.coeffs_y_0, self.coeffs_y_1),
        )

    def _apply_to_q(self, q: torch.Tensor) -> torch.Tensor:
        """Apply the cached PRoPE query transform to ``q``."""
        (_, _, seqlen, head_dim) = q.shape
        assert seqlen == self.cameras * self.patches_x * self.patches_y
        assert head_dim == self.head_dim
        assert self.apply_fn_q is not None
        return self.apply_fn_q(q)

    def _apply_to_kv(self, kv: torch.Tensor) -> torch.Tensor:
        """Apply the cached PRoPE key/value transform to ``kv``."""
        (_, _, seqlen, head_dim) = kv.shape
        assert seqlen == self.cameras * self.patches_x * self.patches_y
        assert head_dim == self.head_dim
        assert self.apply_fn_kv is not None
        return self.apply_fn_kv(kv)

    def _apply_to_o(self, o: torch.Tensor) -> torch.Tensor:
        """Apply the cached PRoPE output transform to ``o``."""
        (_, _, seqlen, head_dim) = o.shape
        assert seqlen == self.cameras * self.patches_x * self.patches_y
        assert head_dim == self.head_dim
        assert self.apply_fn_o is not None
        return self.apply_fn_o(o)
