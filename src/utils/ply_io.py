"""Per-scene Gaussian export in the reference gaussian-splatting .ply layout (x y z, zero
normals, f_dc / channel-major f_rest, pre-sigmoid opacity, log scale, wxyz rotation).
Gaussians are in 2Xplat's normalized camera frame; ``cameras.json`` stores those cameras.
"""

import json
from pathlib import Path

import numpy as np
import torch

from src.model.gaussians import GaussianField

# 2Xplat stores opacity as a degree-0 SH coefficient; the .ply stores the logit C0 * coeff.
SH_C0 = 0.28209479177387814

PROPS_FIXED = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]


def write_ply(path, xyz, feature, scale, rotation, opacity) -> None:
    """xyz (N,3), feature (N,K,3), scale (N,3) log, rotation (N,4) wxyz, opacity (N,) pre-sigmoid."""
    n, k = xyz.shape[0], feature.shape[1]
    cols = [xyz, np.zeros_like(xyz), feature[:, 0, :]]
    names = list(PROPS_FIXED)
    if k > 1:
        rest = feature[:, 1:, :].transpose(0, 2, 1).reshape(n, -1)
        cols.append(rest)
        names += [f"f_rest_{i}" for i in range(rest.shape[1])]
    cols += [opacity[:, None], scale, rotation]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    data = np.concatenate(cols, axis=1).astype(np.float32)
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              + "".join(f"property float {p}\n" for p in names)
              + "end_header\n")
    path = Path(path)
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(data.tobytes())


def read_ply(path) -> dict[str, np.ndarray]:
    """The inverse of write_ply: property name -> (N,) float32 column."""
    with open(path, "rb") as f:
        names = []
        while True:
            line = f.readline().decode().strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property float"):
                names.append(line.split()[-1])
            elif line == "end_header":
                break
        data = np.frombuffer(f.read(n * len(names) * 4), dtype=np.float32).reshape(n, len(names))
    return {k: data[:, i] for i, k in enumerate(names)}


def gaussians_to_ply(path, g: GaussianField, batch: int = 0) -> int:
    """Write one batch element of a GaussianField. Returns the Gaussian count."""
    n = g.xyz.shape[1]
    op = g.opacity[batch].float().reshape(n, -1)
    if op.shape[1] != 1:
        raise ValueError(f"opacity_degree > 0 has no reference-3DGS equivalent (D={op.shape[1]})")
    write_ply(
        path,
        g.xyz[batch].float().cpu().numpy(),
        g.feature[batch].float().cpu().numpy(),
        g.scale[batch].float().cpu().numpy(),
        g.rotation[batch].float().cpu().numpy(),
        (op[:, 0] * SH_C0).cpu().numpy(),
    )
    return n


def gaussians_from_ply(path, device="cpu") -> GaussianField:
    """Rebuild a batch-1 GaussianField from a .ply written by gaussians_to_ply."""
    p = read_ply(path)
    n = len(p["x"])
    xyz = np.stack([p["x"], p["y"], p["z"]], 1)
    n_rest = sum(1 for k in p if k.startswith("f_rest_"))
    k = 1 + n_rest // 3
    feat = np.zeros((n, k, 3), dtype=np.float32)
    feat[:, 0] = np.stack([p["f_dc_0"], p["f_dc_1"], p["f_dc_2"]], 1)
    if n_rest:
        rest = np.stack([p[f"f_rest_{i}"] for i in range(n_rest)], 1)
        feat[:, 1:] = rest.reshape(n, 3, k - 1).transpose(0, 2, 1)   # undo channel-major
    scale = np.stack([p[f"scale_{i}"] for i in range(3)], 1)
    rot = np.stack([p[f"rot_{i}"] for i in range(4)], 1)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float()[None].to(device)
    return GaussianField(
        xyz=t(xyz), feature=t(feat), scale=t(scale), rotation=t(rot),
        opacity=t((p["opacity"] / SH_C0)[:, None, None]), opacity_precompute=None,
    )


def save_scene_cameras(path, cameras, input_index, target_index, image_size: int) -> None:
    """The normalized cameras a saved scene's Gaussians live in, context and target."""
    def block(c2w, intr, index):
        return {
            "frame_index": [int(i) for i in np.asarray(index).reshape(-1)],
            "c2w": c2w[0].float().cpu().numpy().tolist(),
            "fxfycxcy": intr[0].float().cpu().numpy().tolist(),
        }
    payload = {
        "convention": "OpenCV camera-to-world, poses normalized to 2Xplat's scene scale; "
                      "intrinsics in pixels at image_size",
        "image_size": image_size,
        "context": block(cameras.pred_i_c2w, cameras.pred_i_fxfycxcy, input_index),
        "target": block(cameras.pred_t_c2w, cameras.pred_t_fxfycxcy, target_index),
    }
    Path(path).write_text(json.dumps(payload))
