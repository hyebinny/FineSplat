"""Stage 2 of FineSplat: error-conditioned correction of the Stage-1 tokens.

Stage 1 (2Xplat) predicts one token per 4x4 patch, decoded into 4 Gaussians. Each Stage-2
step renders the current Gaussians into the context views, builds a per-view error map,
lifts it onto the tokens by projection and visibility pooling, runs PTv3 serialized-attention
blocks over the token cloud, and predicts a token residual the frozen Stage-1 head decodes.

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
    # "token": one point per patch token, residual on the token (decoded by the Stage-1 head).
    # "gaussian": one point per Gaussian, residual on its raw pre-activation parameters.
    "mode": "token",
    # Stage-2 steps per forward; each re-renders the context views from the previous step.
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
        "lift_channels": 256,
        "occlusion_tol": 0.15,
        "reduce": "mean",  # how visible views are pooled per token: mean | max
        "lift": "mean+home",  # what reaches the token: mean+home | mean | home
        "use_rgb_diff": True,
        "use_image_embed": True,
        "backbone": "resnet18",  # frozen error feature extractor: resnet18 | vgg16_lpips
    },
}


def _sample(img: torch.Tensor, pix: torch.Tensor, size: int) -> torch.Tensor:
    """Bilinear sample img [N,C,h,w] at pixel coords pix [N,M,2] -> [N,M,C]; centres at i+0.5 (align_corners=False)."""
    grid = (pix / size * 2 - 1).unsqueeze(2)
    out = F.grid_sample(img, grid.to(img.dtype), mode="bilinear", padding_mode="border", align_corners=False)
    return out[..., 0].transpose(1, 2)  # [N,M,C]


def _sample_sparse(imgs: torch.Tensor, pix: torch.Tensor, img_idx: torch.Tensor,
                   size: int) -> torch.Tensor:
    """Gather form of ``_sample``: imgs [N,C,h,w], pix [P,2] from image img_idx[p] -> [P,C]."""
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
    """Per-view error map at 1/stride resolution: [error `width` | image `img_width`] channels."""

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
        self.use_err = use_feat_diff or use_rgb_diff
        assert self.use_err or use_image_embed, "no error branches and no image embed: empty map"
        if not use_feat_diff:
            self.stem = self.rest = self.layer2 = None
            self._stages = []
        elif backbone == "resnet18":
            from torchvision.models import ResNet18_Weights, resnet18

            # checkpoints store the frozen backbone under error.stem/rest/layer2 - do not rename
            r = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.stem = nn.Sequential(r.conv1)
            self.rest = nn.Sequential(r.bn1, r.relu, r.maxpool, r.layer1)
            self.layer2 = r.layer2
            self._stages = [self.stem, self.rest, self.layer2]
            feat_channels = 64 + 64 + 128
        else:
            from torchvision.models import VGG16_Weights, vgg16

            vf = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
            self.slice1, self.slice2, self.slice3 = vf[:4], vf[4:9], vf[9:16]
            self._stages = [self.slice1, self.slice2, self.slice3]
            feat_channels = 64 + 128 + 256
        for m in self._stages:
            m.requires_grad_(False)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

        assert image_size % stride == 0, (image_size, stride)
        self.stride = stride
        self.grid_hw = (image_size // stride, image_size // stride)
        img_width = width if img_width is None else img_width
        self.channels = (width if self.use_err else 0) + (img_width if use_image_embed else 0)
        self.feat_proj = nn.Linear(feat_channels, width) if use_feat_diff else None
        self.rgb_proj = nn.Linear(3 * stride * stride, width) if use_rgb_diff else None
        self.err_norm = nn.LayerNorm(width) if self.use_err else None
        # "gt_rgb": pixel-unshuffled context image; "mvp_s1": Stage 1's own patch tokens for the view
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
        """The image half of the map, [N, gh, gw, img_width]; constant across steps, computed once per forward."""
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
        """render, gt [N,3,H,W] -> [N, width + img_width, gh, gw]; ``gt_feat``/``img`` may be precomputed."""
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
    """Optional cross-view attention over the per-view error maps, at 1/``down`` resolution (off by default)."""

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
        nn.init.zeros_(self.proj3.weight)
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
    """Per-view maps -> one feature per token: pooled over the views the token is visible in
    (home view always counts), concatenated with the map at the token's own patch."""

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
        home_view = torch.arange(K, device=dev) // (side * side)
        if self.lift == "home":
            cnt = torch.ones(B, K, 1, device=dev, dtype=E.dtype)
            pooled = None
        else:
            # project every token into every view; only in-frustum pairs are sampled below
            w2c = i_c2w.float().inverse()
            p_cam = (torch.einsum("bkj,bvij->bvki", coord.float(), w2c[:, :, :3, :3])
                     + w2c[:, :, None, :3, 3])
            z = p_cam[..., 2]
            k_mat = fxfycxcy_to_K(i_fxfycxcy.reshape(B * V, 4)).reshape(B, V, 3, 3)
            uvw = torch.einsum("bvkj,bvij->bvki", p_cam, k_mat)
            pix = uvw[..., :2] / uvw[..., 2:3].clamp(min=EPS)
            in_frustum = (z > EPS) & (pix[..., 0] >= 0) & (pix[..., 0] < size) \
                & (pix[..., 1] >= 0) & (pix[..., 1] < size)
            is_home = home_view[None, :] == torch.arange(V, device=dev)[:, None]
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

            # chunked: a single flat [P, C] buffer would dominate memory at high view counts
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
                pooled = pooled / cnt.clamp(min=1)

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
    """One PTv3 block evaluated with gather attention over ``idx`` [N,k] neighbours instead of
    serialized windows; reuses the block's weights and mirrors Block.forward's residual layout."""
    n, k = idx.shape
    a = blk.attn
    heads = a.num_heads
    shortcut = feat
    for m in blk.cpe._modules.values():
        feat = m(feat)
    feat = shortcut + feat
    shortcut = feat
    x = feat
    for m in blk.norm1._modules.values():
        x = m(x)
    qkv = a.qkv(x)
    cp = qkv.shape[-1] // 3
    q, kk, v = qkv.split(cp, dim=-1)
    hd = cp // heads
    q = q.view(n, heads, hd)
    kn = kk[idx.reshape(-1)].view(n, k, heads, hd)
    vn = v[idx.reshape(-1)].view(n, k, heads, hd)
    attn = torch.einsum("nhd,nkhd->nhk", q * a.scale, kn)
    attn = attn.softmax(dim=-1)
    out = torch.einsum("nhk,nkhd->nhd", attn, vn).reshape(n, cp)
    out = a.proj(out)
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

    "inframe": 3x3 block in the token's own view (9). "xframe": 4-neighbourhood in its own view
    plus the 4-neighbourhood of its projection into the partner view v ^ 1 (10), falling back
    to self when the projection is out of frustum or fails the depth test.
    """
    S2 = side * side
    K = V * S2
    ar = torch.arange(side, device=device)
    ii, jj = torch.meshgrid(ar, ar, indexing="ij")
    ii, jj = ii.reshape(-1), jj.reshape(-1)
    base = (torch.arange(B, device=device)[:, None, None] * K
            + torch.arange(V, device=device)[None, :, None] * S2)
    self_idx = (base + torch.arange(S2, device=device)).reshape(-1)

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
    partner = torch.arange(V, device=device) ^ 1
    pc2w = c2w[:, partner].float()
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
    dpart = depth.float()[:, partner]
    d = torch.gather(dpart.reshape(B, V, -1), 2, py * dpart.shape[-1] + px)
    visible = inside & ((d <= EPS) | (z <= d * (1 + tol)))
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
    """PTv3 block (xCPE -> window attention -> FFN, pre-norm) with q/k/v in `attn_channels`."""
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
        # raw_input: raw Gaussian parameters enter the fuse normalised, unprojected; aux: per-point conditioning
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
        # z_dim=None: no head, forward returns the normalised backbone features.
        # attn_mode: "serialized" (trained) or an inference-time gather swap: knn | inframe | xframe.
        self.attn_mode = conf.get("attn_mode", "serialized")
        self.knn_k = conf.get("knn_k", 8)
        self.head = nn.Linear(W, z_dim) if z_dim is not None else None
        if self.head is not None:
            nn.init.zeros_(self.head.weight)  # zero-init -> step 0 reproduces the frozen model
            nn.init.zeros_(self.head.bias)
        self.window = 0

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
        """tokens [B,K,D], err [B,K,E], grids [(grid_coord, offset)] cycled over by the blocks -> [B,K,z_dim]."""
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
                idx = grids[0][1]
            for blk in self.blocks:
                feat = _knn_block_forward(blk, feat, idx)
            self.window = -idx.shape[1]  # negative marks gather attention in the log line
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
    """FineSplat: Stage 1 (2Xplat) + Stage 2; ``forward`` returns the target-view render after every step."""

    def __init__(self, model, conf: dict):
        super().__init__()
        self.e2e = conf.get("e2e", False)  # true: train Stage 1 jointly; default: Stage 1 frozen
        self.grad_ckpt = conf.get("grad_ckpt", False)  # recompute each step's Stage-2 forward in backward
        self.model = model
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
            self.patch = expert.patch_size
        else:
            # one point per Gaussian; the (v hh ph ww pw) unfold makes a view's Gaussians a
            # plain grid of sub-cells, so the same home-patch lookup works with the sub-cell stride
            self.patch = expert.gaussian_stride
        self.side = self.image_size // self.patch

        self.num_steps = conf.get("num_steps", 4)
        # Context views re-rendered with gradients per forward so the loss can score them
        # (int for every view count, or dict view_count -> quota); 0 disables.
        self.context_loss_views = conf.get("context_loss_views", 0)
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
        # gaussian_token_feat: per-Gaussian points carry (raw params | parent token | error);
        # gaussian_delta_z: same points, output is a token residual (G features concatenated in (ph, pw) order)
        self.gaussian_token_feat = self.mode == "gaussian" and conf.get("gaussian_token_feat", False)
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
                nn.init.zeros_(self.token_head.weight)
                nn.init.zeros_(self.token_head.bias)
        else:
            self.stage2 = Stage2(conf["network"], point_dim, err["lift_channels"], point_dim)
        self._logged = False

    def _to_gaussians(self, x: torch.Tensor, v: int) -> torch.Tensor:
        """Broadcast a per-token tensor (B, K, C) onto its Gaussians in the (v hh ph ww pw) unfold
        order: a token's Gaussians are not adjacent, so a plain repeat would be wrong."""
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
            self.model.eval()
        return self

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def _render_views(self, gaussians, c2w, fxfycxcy, with_grad: bool, render_mode="RGB",
                      size: int | None = None):
        """Render every view -> (B,V,3,H,W) rgb and, for "RGB+ED", (B,V,H,W) depth; ``size`` sets the raster size only."""
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

        # "RGB+ED" is not exposed through GaussianRenderer.apply: rasterize one view at a time
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
        out = torch.stack(rows, dim=0)
        rgb = out[..., :3].permute(0, 1, 4, 2, 3).contiguous()
        depth = out[..., 3] if out.shape[-1] == 4 else None
        return rgb, depth

    def forward(self, input_data: dict, target_data: dict) -> dict:
        model, expert = self.model, self.model.appearance_expert
        h = w = self.image_size

        base_ctx = (contextlib.nullcontext() if (self.e2e and self.training)
                    else torch.no_grad())
        with base_ctx:
            cameras = model.geometry_expert.predict_cameras(input_data, target_data)
            raymap_images, Ks, i_w2c = model._build_raymap_input(
                input_data, cameras.pred_i_fxfycxcy, cameras.pred_i_c2w,
            )
            v = raymap_images.shape[1]
            # model.dynamic_grouping: Stage 1 sees a sampled view-group size while it is being
            # trained (end-to-end); frozen or at inference it keeps model.group_size.
            with expert.grouping(v, enabled=self.e2e and self.training):
                x, s1 = expert._stage1_tokenize_and_encode(raymap_images)
                x, s2 = expert._stage2_cross_view_and_downsample(x, i_w2c, Ks, h, w)
                s3 = expert._stage3_global_cross_view(x, i_w2c, Ks)
            tokens = expert.fuse_tokens(s1, s2, s3, v, h, w).float()
            raw = expert.raw_from_tokens(tokens, v, h, w)
            g0 = expert.gaussians_from_raw(
                raw, cameras.pred_i_fxfycxcy, cameras.pred_i_c2w, cameras.pred_t_c2w, h, w,
            )
        render0 = None
        if self.e2e and self.training:
            with torch.autocast(device_type="cuda", enabled=False):
                render0, _ = self._render_views(
                    g0, cameras.pred_t_c2w, cameras.pred_t_fxfycxcy, with_grad=True,
                )

        B = tokens.shape[0]
        K = tokens.shape[1] if self.mode == "token" else raw.shape[1]
        assert K == v * self.side ** 2, (K, v, self.side)
        gt_ctx = rearrange(input_data["image"], "b v c h w -> (b v) c h w").float()
        gt_feat = self.error.features(gt_ctx)
        mvp_feat = None
        if self.error.image_embed == "mvp_s1":
            gh = self.image_size // expert.patch_size
            mvp_feat = rearrange(s1.detach(), "n (hh ww) d -> n d hh ww", hh=gh, ww=gh)
        img_map = self.error.image_map(gt_ctx, mvp_feat)

        points = tokens if self.mode == "token" else raw.float()
        aux =(self._to_gaussians(self.token_broadcast(tokens), v)
               if self.gaussian_token_feat and not self.gaussian_delta_z else None)
        g, renders, n_vis = g0, [], None
        gaussians_all = []
        ctx_renders, ctx_indices = [], None
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
                # separate differentiable render: the error-map render above must stay detached
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

            # Decode under the ambient autocast (as inference does) so a zero residual
            # reproduces Stage 1 bit for bit.
            if self.mode == "token":
                g = expert.decode_from_tokens(
                    points, cameras.pred_i_fxfycxcy, cameras.pred_i_c2w,
                    cameras.pred_t_c2w, h, w,
                )
            else:
                # residual on the pre-activation parameters: only the activations are re-run,
                # so every bound they impose still holds; delta-z re-decodes raw from the tokens
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
            # each step is supervised on its own render and starts from a detached state
            points = points.detach()
            if self.gaussian_delta_z:
                tokens = tokens.detach()

        if not self._logged:
            print(f"flat PTv3 ({self.mode}): {K} points/scene, attention window "
                  f"{self.stage2.window}, {self.num_steps} step(s) (batch of {B}); "
                  f"tokens visible in {n_vis.mean():.2f} views on average", flush=True)
            self._logged = True
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
        """Mean position of each token's Gaussians, (B, N, 3) -> (B, K, 3), honouring the
        (v hh ph ww pw) unfold order (a plain (K, G) reshape would mix neighbouring tokens)."""
        side = self.image_size // self.model.appearance_expert.patch_size
        g = self.model.appearance_expert.gaussian_side
        xyz = rearrange(xyz, "b (v hh ph ww pw) c -> b (v hh ww) (ph pw) c",
                        v=v, hh=side, ww=side, ph=g, pw=g)
        return xyz.mean(dim=2)

    def _grids(self, coord, g0, b: int, k: int):
        """World-space serialization grid: the Stage-1 Gaussians' bounding box split
        ``grid_resolution`` ways per axis, one point cloud per scene -> [(grid, offset)]."""
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
