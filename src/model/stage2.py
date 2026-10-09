"""Stage 2 of FineSplat: error-conditioned correction of the Stage-1 tokens.

Stage 1 (2Xplat: DA3 + MVP) predicts one fused token per 4x4 patch, which its Gaussian
head decodes into 4 Gaussians. Stage 2 runs ``num_steps`` (4) correction steps on top. Each
step renders the current Gaussians into the context views, compares them with the inputs
(a per-view error map: frozen ResNet-18 feature difference + RGB difference, concatenated
with an embedding of the input view itself, at 1/stride resolution), lifts the map onto
the tokens by projecting every token into every view and averaging the bilinear samples
over the views it is visible in (its home view always counts, concatenated with the error
at its own patch), fuses token and error embeddings, and runs them through PTv3 blocks
(serialized window attention + FFN, no pooling, no decoder) over the whole token cloud. A
zero-initialised linear head predicts a token residual, which the Stage-1 head decodes
into the next step's Gaussians.

Ported from ZipSplat's ``splatfactory/models/networks/zipsplat_ptv3_refiner.py``, with the
differences the base model forces:
  - 18,816 tokens per scene at 6 views (6 x 56 x 56), against ZipSplat's 1,944. The flat
    design's premise - "a 1024 window already covers half the scene" - does not hold here;
    ``patch_size`` is left at 1024 regardless, so a window sees ~5% of the scene.
  - 4 Gaussians per token, against 32.
  - 2Xplat has no colour skip connection, so the whole token stream is corrected.
  - Opacity goes through 2Xplat's SH path per view. At the shipped ``opacity_degree: 0``
    that path is view-independent (the DC band is a constant), so the per-view evaluation
    is redundant; it is kept because the surrounding code is written for any degree.

    python -m src.model.stage2   # self-check (needs a GPU)
"""

import contextlib

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from torch import nn

from src.model.gaussians import GaussianRenderer
from src.model.ptv3.ptv3 import Block, SerializedAttention
from src.model.ptv3.structure import Point
from src.model.sh_eval import _spherical_harmonics
from src.utils.camera_utils import fxfycxcy_to_K

EPS = 1e-6
# Cap on elements materialised by one sparse-gather chunk in ErrorLifter (~512 MB in fp32).
_GATHER_CHUNK_ELEMS = 1 << 27
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

default_conf = {
    # "token":    one point per patch token, feature = the 1024-d token, the head predicts a
    #             token residual that the Stage-1 Gaussian head then decodes.
    # "gaussian": one point per Gaussian (4x as many), feature = that Gaussian's raw 23
    #             pre-activation parameters, the head predicts a residual on them. The
    #             Gaussian head is not re-run; only the activations and ray
    #             parameterisation are, so every bound they impose still holds.
    "mode": "token",
    # Stage-2 steps per forward. Each step re-renders the context views from the Gaussians
    # the previous step produced, so the error map it reads is the error that remains -
    # which is what a single-step model never sees, and why applying a single-step model
    # twice at test time degraded after the second application. ReSplat runs 2-4, MoGe3
    # runs 3. Tokens are detached between steps so each step learns to improve whatever
    # state it is handed, rather than backpropagating through the whole chain.
    "num_steps": 4,
    "network": {
        "width": 512,
        "depth": 4,
        "attn_channels": 256,  # q/k/v width (num_heads x 16); == width gives plain PTv3 attention
        "num_heads": 16,
        "patch_size": 1024,  # window = min(points per scene, patch_size)
        "order": ["z", "z-trans", "hilbert", "hilbert-trans"],
        "drop_path": 0.1,
        "grid_resolution": 384,  # serialization + xCPE voxel grid (SplatFormer's value)
    },
    "error": {
        "stride": 4,  # maps at 56x56 for 224x224 inputs
        "width": 128,  # per-view map = [error 128 | image 128]
        "lift_channels": 256,  # per-token error feature
        "occlusion_tol": 0.15,
        "reduce": "mean",  # how visible views are pooled per token: mean | max
        "lift": "mean+home",  # what reaches the token: mean+home | mean | home
        "use_rgb_diff": True,    # pixel RGB difference, summed into the error embedding
        "use_image_embed": True,  # separate embedding of the input image
        "backbone": "resnet18",  # error feature extractor: resnet18 | vgg16_lpips
    },
}


def _sample(img: torch.Tensor, pix: torch.Tensor, size: int) -> torch.Tensor:
    """Bilinear sample img [N,C,h,w] at full-resolution pixel coords pix [N,M,2] -> [N,M,C].

    Pixel centres sit at i+0.5 (COLMAP), which is grid_sample's align_corners=False convention.
    """
    grid = (pix / size * 2 - 1).unsqueeze(2)  # [N,M,1,2]
    out = F.grid_sample(img, grid.to(img.dtype), mode="bilinear", padding_mode="border", align_corners=False)
    return out[..., 0].transpose(1, 2)  # [N,M,C]


def _sample_sparse(imgs: torch.Tensor, pix: torch.Tensor, img_idx: torch.Tensor,
                   size: int) -> torch.Tensor:
    """Bilinear sample imgs [N,C,h,w] at pixel coords pix [P,2], each from image img_idx[p].

    Same convention as ``_sample`` (align_corners=False, padding_mode="border") written as a
    gather so that only the P (token, view) pairs that survive the frustum test are touched.
    Sampling every token in every view costs O(V * K) = O(V^2) work per step, while a token
    is visible in only a handful of views regardless of how many there are.
    """
    n, c, h, w = imgs.shape
    xs = (pix[:, 0] * (w / size) - 0.5).clamp(0, w - 1)
    ys = (pix[:, 1] * (h / size) - 0.5).clamp(0, h - 1)
    x0, y0 = xs.floor(), ys.floor()
    wx, wy = (xs - x0).to(imgs.dtype)[:, None], (ys - y0).to(imgs.dtype)[:, None]
    x0, y0 = x0.long(), y0.long()
    x1, y1 = (x0 + 1).clamp(max=w - 1), (y0 + 1).clamp(max=h - 1)
    flat = imgs.reshape(n, c, h * w)
    # advanced indices at dim 0 and 2 with a slice between them -> result is [P, C]
    top = flat[img_idx, :, y0 * w + x0] * (1 - wx) + flat[img_idx, :, y0 * w + x1] * wx
    bot = flat[img_idx, :, y1 * w + x0] * (1 - wx) + flat[img_idx, :, y1 * w + x1] * wx
    return top * (1 - wy) + bot * wy


class ErrorMapEncoder(nn.Module):
    """Per-view map at 1/stride resolution: [error | image], each `width` channels.

    ``use_rgb_diff`` and ``use_image_embed`` exist for ablations: the first drops the raw
    pixel-difference term that is summed into the error embedding, the second drops the
    separate embedding of the input image (halving the map's channels).

    ``backbone`` selects the frozen feature extractor for the error term:
      - "resnet18": conv1 / layer1 / layer2 (64+64+128 ch), raw feature difference.
      - "vgg16_lpips": VGG16 relu1_2 / relu2_2 / relu3_3 (64+128+256 ch), each layer
        channel-unit-normalised like LPIPS's normalize_tensor before the difference, so
        feat_proj plays the role of LPIPS's learned per-channel weights. The difference
        stays signed (LPIPS squares it) so downstream keeps the error's direction.
    """

    def __init__(self, image_size: int, stride: int, width: int,
                 use_rgb_diff: bool = True, use_image_embed: bool = True,
                 backbone: str = "resnet18", use_feat_diff: bool = True,
                 image_embed: str = "gt_rgb", mvp_dim: int = 256,
                 img_width: int | None = None):
        super().__init__()
        assert backbone in ("resnet18", "vgg16_lpips"), backbone
        assert image_embed in ("gt_rgb", "mvp_s1"), image_embed
        self.backbone = backbone
        self.use_feat_diff = use_feat_diff
        self.image_embed = image_embed if use_image_embed else None
        # Both error branches off is the MVP-only ablation: the map is the image half
        # alone, and the lifter still runs its home-sample and visibility-mean over it.
        self.use_err = use_feat_diff or use_rgb_diff
        assert self.use_err or use_image_embed, "no error branches and no image embed: empty map"
        if not use_feat_diff:
            # I_Res off: the frozen backbone is not built at all (it is the encoder's
            # whole cost); with use_rgb_diff the error is the raw pixel difference alone.
            self.stem = self.rest = self.layer2 = None
            self._stages = []
        elif backbone == "resnet18":
            from torchvision.models import ResNet18_Weights, resnet18

            # attribute names are load-bearing: existing checkpoints store the frozen
            # backbone under error.stem/rest/layer2 - do not rename.
            r = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.stem = nn.Sequential(r.conv1)
            self.rest = nn.Sequential(r.bn1, r.relu, r.maxpool, r.layer1)
            self.layer2 = r.layer2
            self._stages = [self.stem, self.rest, self.layer2]
            feat_channels = 64 + 64 + 128
        else:
            from torchvision.models import VGG16_Weights, vgg16

            vf = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
            self.slice1, self.slice2, self.slice3 = vf[:4], vf[4:9], vf[9:16]  # relu1_2/relu2_2/relu3_3
            self._stages = [self.slice1, self.slice2, self.slice3]
            feat_channels = 64 + 128 + 256
        for m in self._stages:
            m.requires_grad_(False)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

        assert image_size % stride == 0, (image_size, stride)
        self.stride = stride
        self.grid_hw = (image_size // stride, image_size // stride)
        # ``img_width`` defaults to ``width``, the square map both halves shared until the
        # image embedding became the base's own 256-d stage-1 tokens: projecting those down
        # to 128 throws half their rank away, and the two halves need not match.
        img_width = width if img_width is None else img_width
        self.channels = (width if self.use_err else 0) + (img_width if use_image_embed else 0)
        self.feat_proj = nn.Linear(feat_channels, width) if use_feat_diff else None
        self.rgb_proj = nn.Linear(3 * stride * stride, width) if use_rgb_diff else None
        self.err_norm = nn.LayerNorm(width) if self.use_err else None
        # "gt_rgb": our own 4x4 patch embedding of the context image. "mvp_s1": the frozen
        # base's stage-1 patch tokens for the same view, which are its own 4x4 patch
        # embedding already computed by Stage 1 - nothing extra is run.
        img_in = {"gt_rgb": 3 * stride * stride, "mvp_s1": mvp_dim}.get(self.image_embed)
        self.img_proj = (
            nn.Sequential(nn.Linear(img_in, img_width), nn.LayerNorm(img_width))
            if self.image_embed is not None else None
        )

    def train(self, mode: bool = True):
        super().train(mode)
        for m in self._stages:
            m.eval()  # frozen backbone: keep BatchNorm running stats fixed
        return self

    @torch.no_grad()
    def features(self, img: torch.Tensor) -> torch.Tensor | None:
        """img [N,3,H,W] in [0,1] -> [N,C,gh,gw] (C = 256 resnet / 448 vgg)."""
        if not self.use_feat_diff:
            return None
        x = (img.float() - self.mean) / self.std
        feats = []
        for m in self._stages:
            x = m(x)
            feats.append(x)
        if self.backbone == "vgg16_lpips":
            feats = [f / f.pow(2).sum(dim=1, keepdim=True).sqrt().clamp(min=1e-10) for f in feats]
        return torch.cat(
            [F.interpolate(f, size=self.grid_hw, mode="bilinear", align_corners=True) for f in feats],
            dim=1,
        )

    def image_map(self, gt: torch.Tensor, mvp_feat: torch.Tensor | None = None):
        """The image half of the map, [N, gh, gw, img_width].

        Neither the input image nor the base's stage-1 tokens change between Stage-2
        steps, so this is computed once per forward and handed to every step.
        """
        if self.img_proj is None:
            return None
        if self.image_embed == "mvp_s1":
            if mvp_feat.shape[-2:] != self.grid_hw:
                mvp_feat = F.interpolate(mvp_feat, size=self.grid_hw, mode="bilinear", align_corners=True)
            return self.img_proj(mvp_feat.permute(0, 2, 3, 1))
        return self.img_proj(F.pixel_unshuffle(gt, self.stride).permute(0, 2, 3, 1))

    def forward(self, render: torch.Tensor, gt: torch.Tensor,
                gt_feat: torch.Tensor | None = None,
                mvp_feat: torch.Tensor | None = None,
                img: torch.Tensor | None = None) -> torch.Tensor:
        """render, gt [N,3,H,W] -> [N, width + img_width, gh, gw].

        ``gt_feat`` is the backbone's features for ``gt``, which do not change between
        Stage-2 steps. Passing them in keeps the ResNet at one call per step instead of
        two; ReSplat caches the same thing (``assert i == 0`` on its first pass).
        ``mvp_feat`` [N, mvp_dim, gh, gw] is the frozen base's stage-1 patch tokens, read
        only when ``image_embed == "mvp_s1"``.
        """
        if not self.use_err:
            img = self.image_map(gt, mvp_feat) if img is None else img
            return img.permute(0, 3, 1, 2)
        err = None
        if self.use_feat_diff:
            if gt_feat is None:
                gt_feat = self.features(gt)
            err = self.feat_proj((self.features(render) - gt_feat).permute(0, 2, 3, 1))
        if self.rgb_proj is not None:
            rgb = self.rgb_proj(F.pixel_unshuffle(render - gt, self.stride).permute(0, 2, 3, 1))
            err = rgb if err is None else err + rgb
        err = self.err_norm(err)
        if self.img_proj is None:
            return err.permute(0, 3, 1, 2)
        if img is None:
            img = self.image_map(gt, mvp_feat)
        return torch.cat([err, img.to(err.dtype)], dim=-1).permute(0, 3, 1, 2)


class _GlobalBlock(nn.Module):
    """Plain pre-norm transformer block with full self-attention."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class ErrorGlobalAttn(nn.Module):
    """Exchange information between the per-view error maps, ReSplat-style.

    The maps are narrowed to ``attn_channels`` and pixel-unshuffled by ``down`` before
    attention, so the tokens that attend are a factor ``down`` coarser than the map. At
    224px with error stride 4 and down 4 that is 6 x 14 x 14 = 1176 tokens of 64 channels -
    negligible next to the 18,816 points Stage 2 itself runs on.

    Optional: the world-grid Stage 2 already mixes views inside its serialized windows and
    through the projected error lifting, so this module is off in the best configuration
    and used by the ReSplat port, where it is the only cross-view path.
    """

    def __init__(self, channels: int, down: int = 4, attn_channels: int = 64,
                 num_heads: int = 1, depth: int = 1):
        super().__init__()
        self.down = down
        self.proj0 = nn.Linear(channels, attn_channels)
        self.proj1 = nn.Linear(attn_channels * down ** 2, attn_channels)
        self.norm1 = nn.LayerNorm(attn_channels)
        self.blocks = nn.ModuleList([_GlobalBlock(attn_channels, num_heads) for _ in range(depth)])
        self.proj2 = nn.Linear(attn_channels, attn_channels * down ** 2)
        self.norm2 = nn.LayerNorm(attn_channels * down ** 2)
        self.conv = nn.Conv2d(attn_channels, attn_channels, 3, 1, 1)
        self.proj3 = nn.Linear(attn_channels, channels)
        nn.init.zeros_(self.proj3.weight)  # starts as identity on the error maps
        nn.init.zeros_(self.proj3.bias)

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        """E [B, V, C, h, w] -> same shape."""
        b, v, _, h, w = E.shape
        x = rearrange(E, "b v c h w -> b (v h w) c")
        x = self.proj0(x)
        x = rearrange(x, "b (v h w) c -> (b v) c h w", v=v, h=h, w=w)
        x = F.pixel_unshuffle(x, self.down)
        x = rearrange(x, "(b v) c h w -> b (v h w) c", v=v)
        x = self.norm1(self.proj1(x))
        for blk in self.blocks:
            x = blk(x)
        x = self.norm2(self.proj2(x))
        x = rearrange(x, "b (v h w) c -> (b v) c h w", v=v, h=h // self.down, w=w // self.down)
        x = self.conv(F.pixel_shuffle(x, self.down))
        x = rearrange(x, "(b v) c h w -> b (v h w) c", v=v)
        x = self.proj3(x)
        return E + rearrange(x, "b (v h w) c -> b v c h w", v=v, h=h, w=w)


class ErrorLifter(nn.Module):
    """Per-view maps -> one feature per token: mean (or channel-wise max) over the views the
    token is visible in (its home view always counts) concatenated with the map at the
    token's own patch."""

    def __init__(self, in_channels: int, out_channels: int, occlusion_tol: float,
                 reduce: str = "mean", lift: str = "mean+home"):
        super().__init__()
        assert reduce in ("mean", "max"), reduce
        assert lift in ("mean+home", "mean", "home"), lift
        self.tol = occlusion_tol
        self.reduce = reduce
        self.lift = lift
        n_in = 2 if lift == "mean+home" else 1
        self.fuse = nn.Sequential(nn.Linear(n_in * in_channels, out_channels), nn.LayerNorm(out_channels))

    def forward(self, E, depth, coord, i_c2w, i_fxfycxcy, patch: int, side: int, size: int):
        """E [B,V,C,gh,gw], depth [B,V,H,W] (render depth), coord [B,K,3] world points,
        i_c2w [B,V,4,4], i_fxfycxcy [B,V,4], tokens ordered (v, i, j) -> [B,K,C'], [B,K]."""
        B, V, C = E.shape[:3]
        K = coord.shape[1]
        dev = E.device
        home_view = torch.arange(K, device=dev) // (side * side)  # token k comes from view k // side^2
        if self.lift == "home":
            # the pooled term is never read, so skip the projection entirely
            cnt = torch.ones(B, K, 1, device=dev, dtype=E.dtype)
            pooled = None
        else:
            # Project every token into every view. This part is a 3x3 matmul per pair and
            # stays O(V*K); only the two samplings below are culled down to the pairs that
            # land inside a frustum, which is what keeps the cost linear in V.
            w2c = i_c2w.float().inverse()  # [B,V,4,4]
            p_cam = (torch.einsum("bkj,bvij->bvki", coord.float(), w2c[:, :, :3, :3])
                     + w2c[:, :, None, :3, 3])
            z = p_cam[..., 2]
            k_mat = fxfycxcy_to_K(i_fxfycxcy.reshape(B * V, 4)).reshape(B, V, 3, 3)
            uvw = torch.einsum("bvkj,bvij->bvki", p_cam, k_mat)
            pix = uvw[..., :2] / uvw[..., 2:3].clamp(min=EPS)
            in_frustum = (z > EPS) & (pix[..., 0] >= 0) & (pix[..., 0] < size) \
                & (pix[..., 1] >= 0) & (pix[..., 1] < size)
            is_home = home_view[None, :] == torch.arange(V, device=dev)[:, None]  # [V,K]
            b_i, v_i, k_i = (in_frustum | is_home).nonzero(as_tuple=True)
            img_i = b_i * V + v_i
            pix_c = pix[b_i, v_i, k_i]

            d_render = _sample_sparse(
                depth.reshape(B * V, 1, *depth.shape[-2:]).float(), pix_c, img_i, size)[:, 0]
            z_c = z[b_i, v_i, k_i]
            # visible: in the frustum and not behind the rendered surface; the home view always counts
            vis = (in_frustum[b_i, v_i, k_i]
                   & ((d_render <= EPS) | (z_c <= d_render * (1 + self.tol)))) | is_home[v_i, k_i]
            sel = vis.nonzero(as_tuple=True)[0]
            flat_idx = b_i[sel] * K + k_i[sel]
            cnt = torch.zeros(B * K, 1, device=dev, dtype=E.dtype).index_add(
                0, flat_idx, torch.ones(sel.numel(), 1, device=dev, dtype=E.dtype)).view(B, K, 1)

            # Sample the visible pairs in chunks: their number grows with the view count and a
            # single flat [P, C] buffer would dominate memory at high V (20 GB at 128 views).
            e_flat = E.reshape(B * V, C, *E.shape[-2:])
            acc = (torch.full((B * K, C), -torch.inf, device=dev, dtype=E.dtype)
                   if self.reduce == "max" else torch.zeros(B * K, C, device=dev, dtype=E.dtype))
            step = max(1, _GATHER_CHUNK_ELEMS // C)
            for s in range(0, sel.numel(), step):
                part, idx = sel[s : s + step], flat_idx[s : s + step]
                e_s = _sample_sparse(e_flat, pix_c[part], img_i[part], size)
                acc = (acc.scatter_reduce(0, idx[:, None].expand(-1, C), e_s, reduce="amax")
                       if self.reduce == "max" else acc.index_add(0, idx, e_s))
            pooled = acc.view(B, K, C)
            if self.reduce != "max":
                pooled = pooled / cnt.clamp(min=1)  # cnt >= 1 (home view)

        # home view: token (v, i, j) <-> patch centre ((j+0.5)p, (i+0.5)p) of view v
        ij = torch.stack(torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij"), -1)
        centres = ((ij.flip(-1).float() + 0.5) * patch).to(E.device)  # [side, side, 2] as (x, y)
        centres = centres.view(1, side * side, 2).expand(B * V, -1, -1)
        home = _sample(rearrange(E, "b v c h w -> (b v) c h w"), centres, size)
        home = rearrange(home, "(b v) n c -> b (v n) c", b=B)
        assert home.shape[1] == K, (home.shape, K)
        feats = {"mean+home": lambda: torch.cat([pooled, home], dim=-1),
                 "mean": lambda: pooled,
                 "home": lambda: home}[self.lift]()
        return self.fuse(feats), cnt[..., 0]


def _knn_indices(coord: torch.Tensor, offset: torch.Tensor, k: int) -> torch.Tensor:
    """[K,3] coords, scene-end offsets -> [K,k] neighbour indices (self included)."""
    out = torch.empty(coord.shape[0], k, dtype=torch.long, device=coord.device)
    start = 0
    for end in offset.tolist():
        pts = coord[start:end]
        for q0 in range(0, pts.shape[0], 8192):
            q = pts[q0:q0 + 8192]
            d = torch.cdist(q, pts)
            out[start + q0:start + q0 + q.shape[0]] = start + d.topk(k, largest=False).indices
        start = end
    return out


def _knn_block_forward(blk, feat: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """One PTv3 block evaluated with kNN attention instead of serialized windows.

    Reuses the block's own weights - cpe (the Linear+LN that replaces xCPE), norm1,
    attn.qkv / attn.proj, norm2, mlp - and mirrors Block.forward's residual layout
    exactly. Every point attends to its ``idx`` neighbours (itself included), so the
    receptive field is a true metric neighbourhood rather than a serialized window.
    Inference-time only: nothing here is trained.
    """
    n, k = idx.shape
    a = blk.attn
    heads = a.num_heads
    shortcut = feat
    # cpe residual first, as in Block.forward
    for m in blk.cpe._modules.values():
        feat = m(feat)
    feat = shortcut + feat
    shortcut = feat
    x = feat
    for m in blk.norm1._modules.values():
        x = m(x)
    qkv = a.qkv(x)                                   # [N, 3C']
    cp = qkv.shape[-1] // 3
    q, kk, v = qkv.split(cp, dim=-1)
    hd = cp // heads
    q = q.view(n, heads, hd)
    kn = kk[idx.reshape(-1)].view(n, k, heads, hd)   # neighbour keys
    vn = v[idx.reshape(-1)].view(n, k, heads, hd)
    attn = torch.einsum("nhd,nkhd->nhk", q * a.scale, kn)
    attn = attn.softmax(dim=-1)
    out = torch.einsum("nhk,nkhd->nhd", attn, vn).reshape(n, cp)
    out = a.proj(out)
    # Block.forward applies drop path to both residual branches; mirror it so a model
    # trained through this path sees the same regularization as the serialized one.
    for m in blk.drop_path._modules.values():
        out = m(out)
    feat = shortcut + out
    shortcut = feat
    x = feat
    for m in blk.norm2._modules.values():
        x = m(x)
    for m in blk.mlp._modules.values():
        x = m(x)
    for m in blk.drop_path._modules.values():
        x = m(x)
    return shortcut + x


def _grid_neighbors(mode: str, B: int, V: int, side: int, device, coord=None, c2w=None,
                    fxfycxcy=None, depth=None, patch: int = 1, size: int = 1,
                    tol: float = 0.15) -> torch.Tensor:
    """Image-space neighbour indices for tokens laid out (b, v, i, j) -> [B*V*side^2, n].

    "inframe": the 3x3 block around the token in its own view (9, self included).
    "xframe":  the token and its 4-neighbourhood in its own view, plus the token its 3D
               position projects to in the partner view of its MVP view pair (v ^ 1) and
               that token's 4-neighbourhood (10). The partner half falls back to self when
               the projection leaves the frustum or fails the depth test against the
               rendered depth, the same visibility rule the error lifting uses.
    Out-of-image neighbours are replaced by the token itself.
    """
    S2 = side * side
    K = V * S2
    ar = torch.arange(side, device=device)
    ii, jj = torch.meshgrid(ar, ar, indexing="ij")
    ii, jj = ii.reshape(-1), jj.reshape(-1)                      # [S2]
    base = (torch.arange(B, device=device)[:, None, None] * K
            + torch.arange(V, device=device)[None, :, None] * S2)  # [B,V,1]
    self_idx = (base + torch.arange(S2, device=device)).reshape(-1)   # [B*K]

    def offsets(ci, cj, offs):
        cols = []
        for di, dj in offs:
            ni, nj = ci + di, cj + dj
            ok = (ni >= 0) & (ni < side) & (nj >= 0) & (nj < side)
            cols.append((ni.clamp(0, side - 1) * side + nj.clamp(0, side - 1), ok))
        return cols

    if mode == "inframe":
        offs = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)]
        cols = [(base + c).reshape(-1) for c, _ in offsets(ii, jj, offs)]
        oks = [ok.expand(B, V, -1).reshape(-1) for _, ok in offsets(ii, jj, offs)]
        idx = torch.stack([torch.where(o, c, self_idx) for c, o in zip(cols, oks)], 1)
        return idx

    assert mode == "xframe", mode
    cross = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
    own = [torch.where(ok.expand(B, V, -1).reshape(-1), (base + c).reshape(-1), self_idx)
           for c, ok in offsets(ii, jj, cross)]
    # project every token into its partner view
    partner = torch.arange(V, device=device) ^ 1                  # [V]
    pc2w = c2w[:, partner].float()                                # [B,V,4,4] partner cams
    w2c = pc2w.inverse()
    pts = coord.float().view(B, V, S2, 3)
    p_cam = torch.einsum("bvkj,bvij->bvki", pts, w2c[..., :3, :3]) + w2c[..., None, :3, 3]
    z = p_cam[..., 2]
    kmat = fxfycxcy_to_K(fxfycxcy[:, partner].reshape(B * V, 4).float()).reshape(B, V, 3, 3)
    uvw = torch.einsum("bvkj,bvij->bvki", p_cam, kmat)
    pix = uvw[..., :2] / uvw[..., 2:3].clamp(min=EPS)             # [B,V,S2,2] (x, y)
    inside = (z > EPS) & (pix[..., 0] >= 0) & (pix[..., 0] < size) & (pix[..., 1] >= 0) & (pix[..., 1] < size)
    px = pix[..., 0].floor().long().clamp(0, size - 1)
    py = pix[..., 1].floor().long().clamp(0, size - 1)
    dpart = depth.float()[:, partner]                             # [B,V,H,W]
    d = torch.gather(dpart.reshape(B, V, -1), 2, py * dpart.shape[-1] + px)
    visible = inside & ((d <= EPS) | (z <= d * (1 + tol)))        # [B,V,S2]
    ci, cj = (py // patch).clamp(0, side - 1), (px // patch).clamp(0, side - 1)
    pbase = (torch.arange(B, device=device)[:, None, None] * K + partner[None, :, None] * S2)
    theirs = []
    for di, dj in cross:
        ni, nj = ci + di, cj + dj
        ok = visible & (ni >= 0) & (ni < side) & (nj >= 0) & (nj < side)
        c = pbase + ni.clamp(0, side - 1) * side + nj.clamp(0, side - 1)
        theirs.append(torch.where(ok.reshape(-1), c.reshape(-1), self_idx))
    return torch.stack(own + theirs, 1)


def _block(channels, attn_channels, num_heads, patch_size, drop_path, order_index):
    """PTv3 block (xCPE -> window attention -> FFN, pre-norm LayerNorm) whose attention runs
    in `attn_channels` (ReSplat-style low-rank q/k/v)."""
    blk = Block(
        channels=channels, num_heads=num_heads, patch_size=patch_size, mlp_ratio=4, qkv_bias=True,
        drop_path=drop_path, norm_layer=nn.LayerNorm, act_layer=nn.GELU, pre_norm=True, order_index=order_index,
        cpe_indice_key="stage0", enable_rpe=False, enable_flash=False, upcast_attention=False, upcast_softmax=False,
    )
    if attn_channels != channels:
        attn = SerializedAttention(
            channels=attn_channels, num_heads=num_heads, patch_size=patch_size, qkv_bias=True,
            order_index=order_index, enable_rpe=False, enable_flash=False, upcast_attention=False,
            upcast_softmax=False,
        )
        attn.qkv = nn.Linear(channels, 3 * attn_channels, bias=True)
        attn.proj = nn.Linear(attn_channels, channels)
        blk.attn = attn
    return blk


class Stage2(nn.Module):
    """Token + error embeddings -> `depth` PTv3 blocks over the whole token cloud -> token residual."""

    def __init__(self, conf, token_dim: int, error_channels: int, z_dim: int,
                 aux_channels: int = 0, raw_input: bool = False):
        super().__init__()
        self.conf = conf
        self.z_dim = z_dim
        self.order = list(conf["order"])
        W = conf["width"]
        # raw_input: the point feature is a small heterogeneous vector (the 23 raw Gaussian
        # parameters) that goes into the fuse unprojected, normalised only - the ReSplat
        # recipe. aux is a per-point conditioning stream (the parent token, broadcast).
        if raw_input:
            self.token_proj = nn.LayerNorm(token_dim)
            fuse_in = token_dim + error_channels + aux_channels
        else:
            self.token_proj = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, W))
            fuse_in = W + error_channels + aux_channels
        self.fuse = nn.Sequential(nn.Linear(fuse_in, W), nn.LayerNorm(W), nn.GELU())
        dp = torch.linspace(0, conf["drop_path"], conf["depth"]).tolist()
        self.blocks = nn.ModuleList(
            [_block(W, conf["attn_channels"], conf["num_heads"], conf["patch_size"], dp[i], i % len(self.order))
             for i in range(conf["depth"])]
        )
        self.norm = nn.LayerNorm(W)
        # z_dim=None: no head, forward returns the normalised backbone features instead of a
        # residual - the caller owns the output projection (gstoken_DeltaZ groups the four
        # features of a token before its own zero-init head).
        # Inference-time attention swap: "knn" evaluates every block over each point's
        # knn_k nearest neighbours (weights unchanged) instead of serialized windows.
        self.attn_mode = conf.get("attn_mode", "serialized")
        self.knn_k = conf.get("knn_k", 8)
        self.head = nn.Linear(W, z_dim) if z_dim is not None else None
        if self.head is not None:
            nn.init.zeros_(self.head.weight)  # zero-init -> step 0 reproduces the frozen model
            nn.init.zeros_(self.head.bias)
        self.window = 0  # attention window actually used (logged once)

    def _point(self, grid_coord, offset, feat):
        """Serialize `feat` under one grid. Rebuilt whenever the grid changes between blocks."""
        grid = grid_coord.reshape(-1, 3).int()
        point = Point({
            "coord": grid.float(),
            "grid_coord": grid,
            "grid_size": torch.ones(3, device=grid.device),
            "feat": feat,
            "offset": offset,
        })
        point.serialization(order=self.order, shuffle_orders=self.training)
        point.sparsify()
        return point

    def forward(self, tokens, err, grids, aux=None):
        """tokens [B,K,D], err [B,K,E] -> [B,K,z_dim].

        ``grids`` is a list of (grid_coord, offset); the blocks cycle over it. A single
        entry (the world grid) means every block shares one serialization. ``offset`` marks
        the end of each independent point cloud - one per scene here - and thereby which
        points may attend to each other.
        """
        B, K = tokens.shape[:2]
        parts = [self.token_proj(tokens), err.to(tokens.dtype)]
        if aux is not None:
            parts.append(aux.to(tokens.dtype))
        feat = self.fuse(torch.cat(parts, dim=-1))
        feat = feat.reshape(B * K, -1)

        if self.attn_mode in ("knn", "inframe", "xframe"):
            if self.attn_mode == "knn":
                grid_coord, offset = grids[0]
                idx = _knn_indices(grid_coord.reshape(-1, 3).float(), offset, self.knn_k)
            else:
                idx = grids[0][1]  # precomputed image-space neighbours ("idx", tensor)
            for blk in self.blocks:
                feat = _knn_block_forward(blk, feat, idx)
            self.window = -idx.shape[1]  # logged once; negative marks gather attention
            feat = self.norm(feat)
            if self.head is None:
                return feat.view(B, K, -1)
            return self.head(feat).view(B, K, self.z_dim)

        point, active = None, None
        for i, blk in enumerate(self.blocks):
            grid_coord, offset = grids[i % len(grids)]
            if active is not grid_coord:
                point = self._point(grid_coord, offset, feat)
                active = grid_coord
            else:
                point["feat"] = feat
            point = blk(point)
            feat = point.feat
        self.window = self.blocks[0].attn.patch_size
        feat = self.norm(feat)
        if self.head is None:
            return feat.view(B, K, -1)
        return self.head(feat).view(B, K, self.z_dim)


class FineSplat(nn.Module):
    """FineSplat: Stage 1 (2Xplat) + Stage 2 (error-conditioned PTv3 token correction).

    ``forward`` returns the target-view renderings after every Stage-2 step, with
    gradients, plus the Stage-1 Gaussians and rendering for reference.
    """

    def __init__(self, model, conf: dict):
        super().__init__()
        # e2e: true unfreezes the whole Stage 1 (DA3 + MVP + GS head) and trains it
        # jointly with Stage 2, as in 2Xplat's own update stage. Default: Stage 1 is
        # frozen and only Stage 2 learns.
        self.e2e = conf.get("e2e", False)
        # grad_ckpt: recompute each step's Stage-2 forward in backward instead of keeping
        # its activations (every step's graph lives until the single backward).
        self.grad_ckpt = conf.get("grad_ckpt", False)
        self.model = model  # TwoExpertModel (frozen unless e2e)
        if not self.e2e:
            self.model.requires_grad_(False)
            self.model.eval()

        self.conf = conf
        self.mode = conf.get("mode", "token")
        assert self.mode in ("token", "gaussian"), self.mode
        assert model.width == model.height, "square images assumed (side, home-patch lookup)"
        self.image_size = model.width
        expert = model.appearance_expert
        self.G = expert.gaussian_side ** 2

        if self.mode == "token":
            # One point per patch token; its home pixel is the patch centre.
            self.patch = expert.patch_size
        else:
            # One point per Gaussian. The unfold order is (v hh ph ww pw), so the Gaussians
            # of a view form a plain gh x gw grid of sub-cells and the same home-patch
            # lookup works with the sub-cell stride.
            self.patch = expert.gaussian_stride
        self.side = self.image_size // self.patch

        self.num_steps = conf.get("num_steps", 4)
        # Context views to re-render with gradients so the loss can score them. The error
        # map already shows Stage 2 its context error; supervising only the held-out
        # views leaves it to infer that closing that error helps them. Test-time
        # optimisation closes it directly and gains +1.76 PSNR on the held-out views, so
        # the signal transfers - it is simply not in the objective. 0 keeps the shipped
        # behaviour. The views are drawn once per forward; rendering all of them would cost
        # a backward through V renders, which the 32-view bucket cannot afford.
        # train_finesplat overwrites this from train.loss_views.context and pools the
        # per-step renders with the target views into a single loss.
        self.context_loss_views = conf.get("context_loss_views", 0)
        # Set by evaluation code that re-renders each step (EPA); off during training so
        # the Gaussians of earlier steps are freed as soon as their render is taken.
        self.keep_step_gaussians = False

        err = conf["error"]
        self.error = ErrorMapEncoder(
            self.image_size, err["stride"], err["width"],
            use_rgb_diff=err.get("use_rgb_diff", True),
            use_image_embed=err.get("use_image_embed", True),
            backbone=err.get("backbone", "resnet18"),
            use_feat_diff=err.get("use_feat_diff", True),
            image_embed=err.get("image_embed", "gt_rgb"),
            mvp_dim=model.dim1,
            img_width=err.get("img_width"),
        )
        self.lifter = ErrorLifter(
            self.error.channels, err["lift_channels"], err["occlusion_tol"],
            err["reduce"], err.get("lift", "mean+home"),
        )
        self.error_global = (
            ErrorGlobalAttn(
                self.error.channels,
                down=err.get("global_attn_down", 4),
                attn_channels=err.get("global_attn_channels", 64),
                num_heads=err.get("global_attn_heads", 1),
                depth=err.get("global_attn_blocks", 1),
            ) if err.get("global_attn", False) else None
        )
        # B': one point per Gaussian carrying (g raw 23 | parent token 512 | error 256),
        # fused directly - the ReSplat feature recipe on the serialized world backbone.
        self.gaussian_token_feat = self.mode == "gaussian" and conf.get("gaussian_token_feat", False)
        # B'' (gstoken_DeltaZ): same per-Gaussian points and features as B', but the output
        # is a token residual - the four backbone features of a token are concatenated (the
        # (ph, pw) slot order is fixed by the unfold) and mapped to delta-token 1024, which
        # 2Xplat's frozen head decodes. Gaussian-level receptive field, token-level output.
        self.gaussian_delta_z = self.gaussian_token_feat and conf.get("gaussian_delta_z", False)
        point_dim = model.dim3 if self.mode == "token" else self.raw_dim(model)
        if self.gaussian_token_feat:
            self.token_broadcast = nn.Sequential(
                nn.LayerNorm(model.dim3), nn.Linear(model.dim3, conf.get("token_feat_channels", 512)))
            self.stage2 = Stage2(
                conf["network"], point_dim, err["lift_channels"],
                None if self.gaussian_delta_z else point_dim,
                aux_channels=conf.get("token_feat_channels", 512), raw_input=True)
            if self.gaussian_delta_z:
                self.token_head = nn.Linear(self.G * conf["network"]["width"], model.dim3)
                nn.init.zeros_(self.token_head.weight)  # step 0 reproduces the frozen model
                nn.init.zeros_(self.token_head.bias)
        else:
            self.stage2 = Stage2(conf["network"], point_dim, err["lift_channels"], point_dim)
        self._logged = False

    def _to_gaussians(self, x: torch.Tensor, v: int) -> torch.Tensor:
        """Broadcast a per-token tensor (B, K, C) onto its Gaussians in unfold order.

        A token's four Gaussians are unfolded as (v hh ph ww pw) - ph sits between hh and
        ww, so they are not adjacent and a plain repeat would pair most Gaussians with a
        neighbouring token.
        """
        side = self.image_size // self.model.appearance_expert.patch_size
        g = self.model.appearance_expert.gaussian_side
        x = rearrange(x, "b (v hh ww) c -> b v hh ww c", v=v, hh=side, ww=side)
        return rearrange(x.unsqueeze(3).unsqueeze(5).expand(-1, -1, -1, g, -1, g, -1),
                         "b v hh ph ww pw c -> b (v hh ph ww pw) c")

    def _gs_to_token(self, x: torch.Tensor, v: int) -> torch.Tensor:
        """Inverse grouping of ``_to_gaussians``: per-Gaussian (B, N, C) -> per-token
        (B, K, G*C), the G features concatenated in their fixed (ph, pw) slot order."""
        side = self.image_size // self.model.appearance_expert.patch_size
        g = self.model.appearance_expert.gaussian_side
        return rearrange(x, "b (v hh ph ww pw) c -> b (v hh ww) (ph pw c)",
                         v=v, hh=side, ww=side, ph=g, pw=g)

    @staticmethod
    def raw_dim(model) -> int:
        """Width of one Gaussian's raw parameter vector: depth + offset + SH + scale + quat + opacity."""
        e = model.appearance_expert
        return 1 + 2 + e.color_dim + 3 + 4 + e.opacity_dim

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.e2e:
            self.model.eval()  # frozen: never leave eval
        # in e2e the base follows `mode`; TwoExpertModel.train() itself keeps its loss
        # modules in eval.
        return self

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def _render_views(self, gaussians, c2w, fxfycxcy, with_grad: bool, render_mode="RGB",
                      size: int | None = None):
        """Render every view. Returns (B,V,3,H,W) and, for "RGB+ED", (B,V,H,W) depth.

        ``size`` overrides the training resolution. Gaussians carry detail a 224-pixel frame
        discards, so visualisation renders larger; ``fxfycxcy`` has to be scaled to match, as
        this only changes the raster size.
        """
        model = self.model
        h = w = self.image_size if size is None else size
        b, v = c2w.shape[:2]
        if with_grad:
            dirs = gaussians.xyz[:, None, :, :] - c2w[..., :3, 3][..., None, :]
            opacity = torch.broadcast_to(
                gaussians.opacity[..., None, :, :, :], (b, v, gaussians.opacity.shape[1], -1, 1),
            )
            opacity = _spherical_harmonics(model.opacity_degree, dirs, opacity)
            out = GaussianRenderer.apply(
                gaussians.xyz, gaussians.feature, gaussians.scale, gaussians.rotation, opacity,
                c2w, fxfycxcy, w, h, model.sh_degree, model.near_plane, model.far_plane,
            )
            return out.permute(0, 1, 4, 2, 3).contiguous(), None

        # Sequential: gsplat rasterizes one view at a time, and "RGB+ED" is not exposed
        # through GaussianRenderer.apply.
        rows = []
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
            for ib in range(b):
                xyz, feature = gaussians.xyz[ib], gaussians.feature[ib]
                scale, rotation = gaussians.scale[ib], gaussians.rotation[ib]
                opac = gaussians.opacity[ib]
                outs = []
                for i in range(v):
                    d = xyz - c2w[ib, i:i + 1, :3, 3][None, ...]
                    opacity_i = _spherical_harmonics(model.opacity_degree, d, opac[None, ...])[0]
                    outs.append(GaussianRenderer.render(
                        xyz, feature, scale, rotation, opacity_i, c2w[ib, i], fxfycxcy[ib, i],
                        w, h, model.sh_degree, model.near_plane, model.far_plane, render_mode,
                    ))
                rows.append(torch.cat(outs, dim=0))
        out = torch.stack(rows, dim=0)  # (B, V, H, W, C)
        rgb = out[..., :3].permute(0, 1, 4, 2, 3).contiguous()
        depth = out[..., 3] if out.shape[-1] == 4 else None
        return rgb, depth

    def forward(self, input_data: dict, target_data: dict) -> dict:
        model, expert = self.model, self.model.appearance_expert
        h = w = self.image_size

        # e2e training needs gradients through the base encode (and through the predicted
        # cameras into DA3); otherwise the base stays under no_grad as before.
        base_ctx = (contextlib.nullcontext() if (self.e2e and self.training)
                    else torch.no_grad())
        with base_ctx:
            cameras = model.geometry_expert.predict_cameras(input_data, target_data)
            raymap_images, Ks, i_w2c = model._build_raymap_input(
                input_data, cameras.pred_i_fxfycxcy, cameras.pred_i_c2w,
            )
            x, s1 = expert._stage1_tokenize_and_encode(raymap_images)
            x, s2 = expert._stage2_cross_view_and_downsample(x, i_w2c, Ks, h, w)
            s3 = expert._stage3_global_cross_view(x, i_w2c, Ks)
            v = raymap_images.shape[1]
            tokens = expert.fuse_tokens(s1, s2, s3, v, h, w).float()
            raw = expert.raw_from_tokens(tokens, v, h, w)
            g0 = expert.gaussians_from_raw(
                raw, cameras.pred_i_fxfycxcy, cameras.pred_i_c2w, cameras.pred_t_c2w, h, w,
            )
        render0 = None
        if self.e2e and self.training:
            # 2Xplat's own objective on the Stage-1 prediction (step 0), so Stage 1 keeps
            # being supervised directly and not only through the Stage-2 steps.
            with torch.autocast(device_type="cuda", enabled=False):
                render0, _ = self._render_views(
                    g0, cameras.pred_t_c2w, cameras.pred_t_fxfycxcy, with_grad=True,
                )

        # K is the number of corrected points: tokens in token mode, Gaussians in gaussian
        # mode. self.side is set per mode in __init__ (56 vs 112), so the check holds in both.
        B = tokens.shape[0]
        K = tokens.shape[1] if self.mode == "token" else raw.shape[1]
        assert K == v * self.side ** 2, (K, v, self.side)
        gt_ctx = rearrange(input_data["image"], "b v c h w -> (b v) c h w").float()
        gt_feat = self.error.features(gt_ctx)  # constant across steps
        # image_embed="mvp_s1": reuse the frozen base's stage-1 patch tokens (its own 4x4
        # patch embedding of this view) instead of embedding the raw image ourselves.
        mvp_feat = None
        if self.error.image_embed == "mvp_s1":
            gh = self.image_size // expert.patch_size
            mvp_feat = rearrange(s1.detach(), "n (hh ww) d -> n d hh ww", hh=gh, ww=gh)
        # The image half of the error map reads only the context views, which do not change
        # between steps, so it is projected once here rather than inside every step.
        img_map = self.error.image_map(gt_ctx, mvp_feat)

        # In gaussian mode every Gaussian is its own point, so the coordinate is exact
        # rather than the mean of the four Gaussians a token decodes into.
        points = tokens if self.mode == "token" else raw.float()
        # The parent token does not change across steps (only the Gaussians move), so its
        # broadcast is computed once. Detached like ReSplat's prev-gaussian inputs. In
        # delta-z mode the tokens DO change every step, so aux is rebuilt inside the loop.
        aux = (self._to_gaussians(self.token_broadcast(tokens), v)
               if self.gaussian_token_feat and not self.gaussian_delta_z else None)
        g, renders, n_vis = g0, [], None
        # Per-step Gaussians are only kept when something downstream needs to re-render
        # them - EPA scores each step by realigning the target poses against that step's
        # own Gaussians, which the rendered images alone cannot support.
        gaussians_all = []
        ctx_renders, ctx_indices = [], None
        # An int applies to every view-count bucket; a dict maps the bucket's view count to
        # its own quota (train_finesplat builds it from loss_views.context as a list).
        k_ctx = (self.context_loss_views if isinstance(self.context_loss_views, int)
                 else self.context_loss_views.get(v, 0))
        if k_ctx and self.training:
            ctx_indices = torch.randperm(v, device=gt_ctx.device)[:min(k_ctx, v)]
        for _ in range(self.num_steps):
            with torch.no_grad():
                ctx_rgb, ctx_depth = self._render_views(
                    g, cameras.pred_i_c2w, cameras.pred_i_fxfycxcy,
                    with_grad=False, render_mode="RGB+ED",
                )
            if ctx_indices is not None:
                # A second, differentiable render of the sampled context views. The one
                # above feeds the error map and stays detached: letting the loss reach the
                # Gaussians through the map as well would supervise the conditioning path
                # rather than the prediction.
                with torch.autocast(device_type="cuda", enabled=False):
                    sub, _ = self._render_views(
                        g, cameras.pred_i_c2w[:, ctx_indices],
                        cameras.pred_i_fxfycxcy[:, ctx_indices], with_grad=True,
                    )
                ctx_renders.append(sub)
            E = self.error(rearrange(ctx_rgb, "b v c h w -> (b v) c h w").float(), gt_ctx,
                           gt_feat, mvp_feat, img=img_map)
            E = rearrange(E, "(b v) c h w -> b v c h w", b=B)
            if self.error_global is not None:
                E = self.error_global(E)

            coord = self._token_coord(g.xyz, v) if self.mode == "token" else g.xyz
            err, n_vis = self.lifter(
                E, ctx_depth, coord, cameras.pred_i_c2w, cameras.pred_i_fxfycxcy,
                self.patch, self.side, self.image_size,
            )
            if self.stage2.attn_mode in ("inframe", "xframe"):
                grids = [("idx", _grid_neighbors(
                    self.stage2.attn_mode, B, v, self.side, coord.device, coord=coord,
                    c2w=cameras.pred_i_c2w, fxfycxcy=cameras.pred_i_fxfycxcy,
                    depth=ctx_depth.view(B, v, *ctx_depth.shape[-2:]), patch=self.patch,
                    size=self.image_size, tol=self.conf["error"]["occlusion_tol"]))]
            else:
                grids = self._grids(coord, g, B, K)
            if self.gaussian_delta_z:
                aux = self._to_gaussians(self.token_broadcast(tokens), v)
                feat = self._run_stage2(points.detach(), err, grids, aux=aux)
                tokens = tokens + self.token_head(self._gs_to_token(feat, v))
            elif self.gaussian_token_feat:
                points = points + self._run_stage2(points.detach(), err, grids, aux=aux)
            else:
                points = points + self._run_stage2(points, err, grids)

            # Decode under the ambient autocast, exactly as src/commands/inference.py does,
            # so that delta = 0 reproduces 2Xplat's own numbers bit for bit. ZipSplat forces
            # its head to fp32; matching 2Xplat's evaluation protocol matters more here,
            # since the fp32 head shifted PSNR by +0.06 - a seventh of Stage 2's gain.
            if self.mode == "token":
                g = expert.decode_from_tokens(
                    points, cameras.pred_i_fxfycxcy, cameras.pred_i_c2w,
                    cameras.pred_t_c2w, h, w,
                )
            else:
                # Residual on the pre-activation parameters: the Gaussian head is not re-run,
                # only the activations, so the offset stays in its sub-cell, depth stays
                # inside (0, max_dist) and scale stays clamped. In delta-z mode the residual
                # landed on the tokens instead, so the raw parameters (which are also the
                # next step's point features) are re-decoded from them through the frozen head.
                if self.gaussian_delta_z:
                    points = expert.raw_from_tokens(tokens, v, h, w).float()
                g = expert.gaussians_from_raw(
                    points, cameras.pred_i_fxfycxcy, cameras.pred_i_c2w,
                    cameras.pred_t_c2w, h, w,
                )
            with torch.autocast(device_type="cuda", enabled=False):
                render, _ = self._render_views(
                    g, cameras.pred_t_c2w, cameras.pred_t_fxfycxcy, with_grad=True,
                )
            renders.append(render)
            if self.keep_step_gaussians:
                gaussians_all.append(g)
            # Each step is supervised on its own render and starts from a detached state, so
            # it learns to improve whatever it is handed instead of relying on gradients
            # reaching back through every earlier step.
            points = points.detach()
            if self.gaussian_delta_z:
                tokens = tokens.detach()

        if not self._logged:
            print(f"flat PTv3 ({self.mode}): {K} points/scene, attention window "
                  f"{self.stage2.window}, {self.num_steps} step(s) (batch of {B}); "
                  f"tokens visible in {n_vis.mean():.2f} views on average", flush=True)
            self._logged = True
        # "points" is what the head decodes: the corrected tokens in token mode, the raw
        # Gaussian parameters in gaussian mode. Exposed so the displacement it represents
        # can be compared against the one test-time optimisation finds for the same scene.
        return {"render": renders[-1], "renders": renders, "gaussians": g,
                "gaussians_all": gaussians_all, "gaussians_base": g0,
                "render0": render0,
                "tokens_final": points, "tokens_base": tokens,
                "context_renders": ctx_renders, "context_indices": ctx_indices,
                "cameras": cameras, "error_views": n_vis}

    def _run_stage2(self, *args, **kwargs):
        if self.grad_ckpt and self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(self.stage2, *args, use_reentrant=False, **kwargs)
        return self.stage2(*args, **kwargs)

    def _token_coord(self, xyz: torch.Tensor, v: int) -> torch.Tensor:
        """Mean position of each token's Gaussians, (B, N, 3) -> (B, K, 3).

        2Xplat unfolds a token's Gaussians as ``(v hh ph ww pw)``, so ph sits between hh and
        ww: a token's four Gaussians are ww apart, not adjacent. Reshaping to (K, G) and
        averaging would mix each token with its neighbour along ww.
        """
        side = self.image_size // self.model.appearance_expert.patch_size
        g = self.model.appearance_expert.gaussian_side
        xyz = rearrange(xyz, "b (v hh ph ww pw) c -> b (v hh ww) (ph pw) c",
                        v=v, hh=side, ww=side, ph=g, pw=g)
        return xyz.mean(dim=2)

    def _grids(self, coord, g0, b: int, k: int):
        """World-space grid for the serialization: the scene bounding box of the frozen
        Gaussians, split ``grid_resolution`` ways per axis. One point cloud per scene, so
        serialized windows mix points from every view - each token is visible in 3-4 views
        on average and their world positions nearly coincide, which is where cross-view
        coordination happens. Returned as a single-entry list (the blocks cycle over it).

        A per-view frustum grid and an alternating frustum/world schedule were tried and
        retired; both measured at or below this (frustum +0.86, alternate +1.01, world
        +1.16 on the test watchlist).
        """
        lo = g0.xyz.amin(dim=-2, keepdim=True)
        span = (g0.xyz.amax(dim=-2, keepdim=True) - lo).clamp(min=EPS)
        res = self.conf["network"]["grid_resolution"]
        grid = (((coord - lo) / span) * res).floor().int().clamp(0, res - 1)
        offset = torch.arange(1, b + 1, device=coord.device) * k
        return [(grid, offset)]

    @torch.no_grad()
    def render_baseline_targets(self, gaussians_base, cameras) -> torch.Tensor:
        """Render the frozen Gaussians at the target views, for side-by-side logging."""
        rgb, _ = self._render_views(
            gaussians_base, cameras.pred_t_c2w, cameras.pred_t_fxfycxcy, with_grad=False,
        )
        return rgb
