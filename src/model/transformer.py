import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Normalization  
# (src: https://github.com/pytorch/benchmark/blob/main/torchbenchmark/models/llama/model.py#L28)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization without mean-centering."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        """Initializes RMSNorm with a learnable per-dimension scale.

        Args:
            dim: Feature dimension; sets the size of the learned weight vector.
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Applies RMS normalization to x without a learned scale.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Tensor of the same shape as x, normalized along the last dimension.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizes x and applies the learned per-dimension scale.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Normalized tensor of the same shape as x, scaled by self.weight.
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight.type_as(x)


# ---------------------------------------------------------------------------
# Feed-forward block
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Two-layer feed-forward block with GELU activation (no bias by default)."""

    def __init__(self, dim: int, inter_multi: float = 4, bias: bool = False) -> None:
        """Initializes the two-layer feed-forward block.

        Args:
            dim: Input and output feature dimension.
            inter_multi: Expansion factor for the hidden dimension.
            bias: Whether to add bias terms to the linear layers.
        """
        super().__init__()
        intermediate_dim = int(dim * inter_multi)
        self.c_fc = nn.Linear(dim, intermediate_dim, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(intermediate_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        """Projects x to an intermediate dimension, applies GELU, then projects back.

        Args:
            x: Input tensor of shape (B, L, dim).
            *args: Ignored extra arguments, accepted for interface compatibility.

        Returns:
            Tensor of shape (B, L, dim).
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


# ---------------------------------------------------------------------------
# Self-attention
# ---------------------------------------------------------------------------

class SelfAttention(nn.Module):
    """Self-attention layer with optional QK normalization and camera-aware attention.

    Reference: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L68-L92
    """

    def __init__(
        self,
        dim: int,
        head_dim: int,
        use_qk_norm: bool = True,
        causal: bool = False,
        bias: bool = False,
    ) -> None:
        """Initializes SelfAttention.

        Args:
            dim: Total embedding dimension; must be divisible by head_dim.
            head_dim: Dimension per attention head.
            use_qk_norm: Whether to apply RMSNorm to queries and keys before attention.
            causal: Whether to apply a causal mask (used in autoregressive settings).
            bias: Whether to add bias to linear projections.
        """
        super().__init__()
        assert dim % head_dim == 0
        self.dim = dim
        self.head_dim = head_dim

        self.to_qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.c_proj = nn.Linear(dim, dim, bias=bias)
        self.use_qk_norm = use_qk_norm

        if self.use_qk_norm:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)

        self.causal = causal

    def forward(self, x: torch.Tensor, prope: bool, stage: int, *args) -> torch.Tensor:
        """Runs self-attention, dispatching to stage-specific attention kernels.

        Stage 1 uses standard scaled-dot-product attention. Stages 2 and 3 delegate
        to camera-aware PropeDotProductAttention kernels (attn2/attn3 in args[0]);
        when prope is True the kernel receives camera extrinsics (w2c) and intrinsics
        (Ks) to apply positional encoding, otherwise those arguments are passed as None.

        Args:
            x: Input tensor of shape (B, L, dim).
            prope: Whether to enable camera-aware positional encoding in attention.
            stage: Transformer stage index (1, 2, or 3) that selects the attention kernel.
            *args: Optional dict at args[0] containing keys 'w2c', 'Ks', 'attn2', 'attn3'
                for stage-2/3 attention dispatch.

        Returns:
            Output tensor of shape (B, L, dim).
        """
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, "b l (qkv nh dh) -> qkv b nh l dh", qkv=3, dh=self.head_dim)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if stage == 1:
            x = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)

        elif stage == 2:
            if prope:
                w2c = args[0]["w2c"]
                Ks = args[0]["Ks"]
                attn_fn = args[0]["attn2"]

                x = attn_fn(
                    q, k, v,
                    viewmats=w2c,
                    Ks=Ks,
                )
            else:
                attn_fn = args[0]["attn2"]
                x = attn_fn(
                    q, k, v,
                    viewmats=None,
                    Ks=None,
                )
        elif stage == 3:
            if prope:
                w2c = args[0]["w2c"]
                Ks = args[0]["Ks"]
                attn_fn = args[0]["attn3"]
                x = attn_fn(
                    q, k, v,
                    viewmats=w2c,
                    Ks=Ks,
                )
            else:
                attn_fn = args[0]["attn3"]
                x = attn_fn(
                    q, k, v,
                    viewmats=None,
                    Ks=None,
                )

        x = rearrange(x, "b nh l dh -> b l (nh dh)")
        x = self.c_proj(x)
        return x


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-norm transformer block combining self-attention and an MLP feed-forward layer."""

    def __init__(self, dim: int, bias: bool, head_dim: int, inter_multi: float, use_qk_norm: bool) -> None:
        """Initializes TransformerBlock.

        Args:
            dim: Embedding dimension for all sub-layers.
            bias: Whether to include bias in LayerNorm and linear projections.
            head_dim: Dimension per attention head.
            inter_multi: Expansion factor for the MLP intermediate dimension.
            use_qk_norm: Whether to apply RMSNorm to attention queries and keys.
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(dim, bias=bias, eps=1e-5)
        self.attn = SelfAttention(dim=dim, bias=bias, head_dim=head_dim, use_qk_norm=use_qk_norm)

        self.ln2 = nn.LayerNorm(dim, bias=bias, eps=1e-5)
        self.mlp = MLP(dim=dim, bias=bias, inter_multi=inter_multi)

    def forward(self, x: torch.Tensor, prope: bool, stage: int, info: dict | None) -> torch.Tensor:
        """Applies pre-norm self-attention then pre-norm MLP with residual connections.

        Args:
            x: Input tensor of shape (B, L, dim).
            prope: Whether to enable camera-aware positional encoding in attention.
            stage: Transformer stage index passed through to SelfAttention.
            info: Optional dict with camera parameters and attention kernels for
                stage-2/3 attention; see SelfAttention.forward for expected keys.

        Returns:
            Output tensor of shape (B, L, dim).
        """
        x = x + self.attn(self.ln1(x), prope, stage, info)
        x = x + self.mlp(self.ln2(x))
        return x
