import torch
import numpy as np
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# General-purpose linear algebra utilities for camera matrices
# ---------------------------------------------------------------------------

def invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
    """Invert a batch of 4×4 SE(3) matrices analytically."""
    assert transforms.shape[-2:] == (4, 4)
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


# ---------------------------------------------------------------------------
# Ray generation related functions from camera parameters
# ---------------------------------------------------------------------------

def compute_plucmap(fxfycxcy, c2w, h, w):
    """Compute per-pixel Plucker ray maps from camera intrinsics and extrinsics.

    Args:
        fxfycxcy (torch.Tensor): Intrinsics [b, v, 4] as [fx, fy, cx, cy].
        c2w (torch.Tensor): Camera-to-world matrices [b, v, 4, 4].
        h (int): Image height in pixels.
        w (int): Image width in pixels.

    Returns:
        ray_o (torch.Tensor): Ray origins with shape (b, v, 3, h, w).
        ray_d (torch.Tensor): Normalized ray directions with shape (b, v, 3, h, w).
    """
    b, v = fxfycxcy.size(0), fxfycxcy.size(1)

    # Efficient meshgrid equivalent using broadcasting
    idx_x = torch.arange(w, device=c2w.device)[None, :].expand(h, -1)  # [h, w]
    idx_y = torch.arange(h, device=c2w.device)[:, None].expand(-1, w)  # [h, w]

    idx_x = idx_x.flatten().expand(b * v, -1)           # [b*v, h*w]
    idx_y = idx_y.flatten().expand(b * v, -1)           # [b*v, h*w]

    fxfycxcy = fxfycxcy.reshape(b * v, 4)               # [b*v, 4]
    c2w = c2w.reshape(b * v, 4, 4)                      # [b*v, 4, 4]

    x = (idx_x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]     # [b*v, h*w]
    y = (idx_y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]     # [b*v, h*w]
    z = torch.ones_like(x)                                        # [b*v, h*w]

    ray_d = torch.stack([x, y, z], dim=1)                         # [b*v, 3, h*w]
    ray_d = torch.bmm(c2w[:, :3, :3], ray_d)                      # [b*v, 3, h*w]
    ray_d = ray_d / torch.norm(ray_d, dim=1, keepdim=True)        # [b*v, 3, h*w]

    ray_o = c2w[:, :3, 3:4].expand(b * v, -1, h*w)                # [b*v, 3, h*w]

    ray_o = ray_o.reshape(b, v, 3, h, w)                          # [b, v, 3, h, w]
    ray_d = ray_d.reshape(b, v, 3, h, w)                          # [b, v, 3, h, w]

    return ray_o, ray_d


def compute_rays(fxfycxcy, c2w, h, w):
    """Compute per-pixel ray origins and directions, flattened across views and pixels.

    Args:
        fxfycxcy (torch.Tensor): Intrinsics [b, v, 4] as [fx, fy, cx, cy].
        c2w (torch.Tensor): Camera-to-world matrices [b, v, 4, 4].
        h (int): Image height in pixels.
        w (int): Image width in pixels.

    Returns:
        ray_o (torch.Tensor): Ray origins with shape (b, v*h*w, 3).
        ray_d (torch.Tensor): Normalized ray directions with shape (b, v*h*w, 3).
    """
    b, v = fxfycxcy.size(0), fxfycxcy.size(1)

    # Efficient meshgrid equivalent using broadcasting
    idx_x = torch.arange(w, device=c2w.device)[None, :].expand(h, -1)  # [h, w]
    idx_y = torch.arange(h, device=c2w.device)[:, None].expand(-1, w)  # [h, w]

    idx_x = idx_x.flatten().expand(b * v, -1)           # [b*v, h*w]
    idx_y = idx_y.flatten().expand(b * v, -1)           # [b*v, h*w]

    fxfycxcy = fxfycxcy.reshape(b * v, 4)               # [b*v, 4]
    c2w = c2w.reshape(b * v, 4, 4)                      # [b*v, 4, 4]

    x = (idx_x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]     # [b*v, h*w]
    y = (idx_y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]     # [b*v, h*w]
    z = torch.ones_like(x)                                        # [b*v, h*w]

    ray_d = torch.stack([x, y, z], dim=1)                         # [b*v, 3, h*w]
    ray_d = torch.bmm(c2w[:, :3, :3], ray_d)                      # [b*v, 3, h*w]
    ray_d = ray_d / torch.norm(ray_d, dim=1, keepdim=True)        # [b*v, 3, h*w]

    ray_o = c2w[:, :3, 3:4].expand(b * v, -1, h*w)                # [b*v, 3, h*w]

    ray_o = ray_o.reshape(b, v, 3, h, w)                          # [b, v, 3, h, w]
    ray_d = ray_d.reshape(b, v, 3, h, w)                          # [b, v, 3, h, w]

    ray_o = rearrange(ray_o, 'b v c h w -> b (v h w) c')
    ray_d = rearrange(ray_d, 'b v c h w -> b (v h w) c')

    return ray_o, ray_d


# ---------------------------------------------------------------------------
# Camera matrix (intrinsics, extrinsics) conversions
# ---------------------------------------------------------------------------

def fxfycxcy_to_K(fxfycxcy: torch.Tensor) -> torch.Tensor:
    """Build a 3x3 intrinsic matrix from a [..., 4] vector [fx, fy, cx, cy]."""
    K = torch.zeros(*fxfycxcy.shape[:-1], 3, 3, dtype=fxfycxcy.dtype, device=fxfycxcy.device)
    K[..., 0, 0] = fxfycxcy[..., 0]
    K[..., 1, 1] = fxfycxcy[..., 1]
    K[..., 0, 2] = fxfycxcy[..., 2]
    K[..., 1, 2] = fxfycxcy[..., 3]
    K[..., 2, 2] = 1.0
    return K


def lift_K(Ks: torch.Tensor) -> torch.Tensor:
    """Embed 3×3 intrinsics matrices into homogeneous 4×4 matrices."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros(Ks.shape[:-2] + (4, 4), device=Ks.device)
    out[..., :3, :3] = Ks
    out[..., 3, 3] = 1.0
    return out


def invert_K(Ks: torch.Tensor) -> torch.Tensor:
    """Invert 3×3 camera intrinsics matrices (assumes no skew)."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros_like(Ks)
    out[..., 0, 0] = 1.0 / Ks[..., 0, 0]
    out[..., 1, 1] = 1.0 / Ks[..., 1, 1]
    out[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
    out[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
    out[..., 2, 2] = 1.0
    return out


def extrinsics_to_44(ext: torch.Tensor) -> torch.Tensor:
    """Pad a (B, V, 3, 4) extrinsics tensor to (B, V, 4, 4) with a [0,0,0,1] bottom row.

    Args:
        ext (torch.Tensor): Extrinsics with shape (B, V, 3, 4) or (B, V, 4, 4).

    Returns:
        torch.Tensor: Extrinsics with shape (B, V, 4, 4).
    """
    if ext.shape[-2] == 3:
        B, V = ext.shape[:2]

        out = torch.eye(
            4,
            device=ext.device,
            dtype=ext.dtype
        ).expand(B, V, 4, 4).clone()  # (B, V, 4, 4)

        out[:, :, :3, :4] = ext
        return out

    return ext


def quat_to_mat(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as quaternions to rotation matrices.

    Quaternion order is XYZW (scalar-last / ijkr convention).

    Args:
        quaternions (torch.Tensor): Quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        torch.Tensor: Rotation matrices as tensor of shape (..., 3, 3).
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def mat_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix (torch.Tensor): Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        torch.Tensor: Quaternions with real part last, as tensor of shape (..., 4).
            Quaternion order is XYZW (scalar-last / ijkr convention).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(batch_dim + (4,))

    # Convert from rijk to ijkr
    out = out[..., [1, 2, 3, 0]]

    out = standardize_quaternion(out)

    return out


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """Return sqrt(max(0, x)) with a zero subgradient where x is 0."""
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert a unit quaternion to a standard form where the real part is non-negative.

    Args:
        quaternions (torch.Tensor): Quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        torch.Tensor: Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)
