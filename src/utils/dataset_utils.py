"""
Helpers used by both InferenceDataset and DynamicInputDataset.
"""

import os
import numpy as np
import PIL.Image as Image
import torch
import torch.nn.functional as F
from einops import repeat


# ---------------------------------------------------------------------------
# Image + intrinsics helpers
# ---------------------------------------------------------------------------

def _pil_to_chw_float(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a CHW float tensor in [0, 1]."""
    return torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0


def _resize_to_cover(
    image: Image.Image,
    target_h: int,
    target_w: int,
    resize_mode: str = "lanczos",
) -> tuple[torch.Tensor, int, int, int, int]:
    """Scale image so both target dimensions are covered, preserving aspect ratio.

    "lanczos" resamples the uint8 PIL image; "bilinear_antialias" resamples the float
    tensor with torch. The two are not interchangeable - they differ in kernel and in
    whether the source is quantised - and the 2Xplat evaluation protocol names which one
    a number was produced under, so the choice has to be stated alongside any metric.

    Args:
        image (Image.Image): Source image.
        target_h (int): Target height to cover.
        target_w (int): Target width to cover.
        resize_mode (str): 'lanczos' or 'bilinear_antialias'.

    Returns:
        tuple: (CHW float tensor in [0, 1], orig_w, orig_h, scaled_w, scaled_h).
    """
    orig_w, orig_h = image.size
    scale = max(target_h / orig_h, target_w / orig_w)
    scaled_h = round(orig_h * scale)
    scaled_w = round(orig_w * scale)

    if resize_mode == "lanczos":
        image_tensor = _pil_to_chw_float(image.resize((scaled_w, scaled_h), Image.LANCZOS))
    elif resize_mode == "bilinear_antialias":
        image_tensor = F.interpolate(
            _pil_to_chw_float(image).unsqueeze(0),
            size=(scaled_h, scaled_w),
            mode="bilinear",
            antialias=True,
            align_corners=False,
        ).squeeze(0)
    else:
        raise NotImplementedError(f"Unknown resize_mode: {resize_mode}")

    return image_tensor, orig_w, orig_h, scaled_w, scaled_h


def _center_square_crop(
    image: torch.Tensor,
    intrinsics: np.ndarray,
    crop_size: int,
) -> tuple[torch.Tensor, np.ndarray]:
    """Crop a square from the center and shift cx/cy accordingly."""
    h, w = image.shape[-2:]
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    image = image[..., start_h:start_h + crop_size, start_w:start_w + crop_size]
    adjusted = intrinsics.copy()
    adjusted[2] -= start_w
    adjusted[3] -= start_h
    return image, adjusted


def _prepare_image_and_intrinsics(
    image: Image.Image,
    raw_intrinsics: np.ndarray,
    target_h: int,
    target_w: int,
    square_crop: bool,
    resize_mode: str = "lanczos",
) -> tuple[torch.Tensor, np.ndarray]:
    """Resize/crop an already-loaded image and rescale its pixel intrinsics to match."""
    image_tensor, orig_w, orig_h, scaled_w, scaled_h = _resize_to_cover(
        image, target_h, target_w, resize_mode
    )

    intrinsics = raw_intrinsics * np.array(
        [scaled_w / orig_w, scaled_h / orig_h, scaled_w / orig_w, scaled_h / orig_h]
    )

    if square_crop:
        crop_size = min(target_h, target_w)
        image_tensor, intrinsics = _center_square_crop(image_tensor, intrinsics, crop_size)

    return image_tensor, intrinsics


def _load_frame_image_and_intrinsics(
    frame: dict,
    image_base_dir: str,
    target_h: int,
    target_w: int,
    square_crop: bool,
    resize_mode: str = "lanczos",
) -> tuple[torch.Tensor, np.ndarray]:
    """Load one frame's image and return it with intrinsics adjusted for resize/crop."""
    image_path = os.path.join(image_base_dir, frame["file_path"])
    raw_intrinsics = np.array([frame["fx"], frame["fy"], frame["cx"], frame["cy"]])
    return _prepare_image_and_intrinsics(
        Image.open(image_path), raw_intrinsics, target_h, target_w, square_crop, resize_mode
    )


# ---------------------------------------------------------------------------
# Pose helpers
# ---------------------------------------------------------------------------

def _w2c_stack_to_c2w(frames: list[dict]) -> torch.Tensor:
    """Invert per-frame w2c matrices and pin the last row to [0,0,0,1] for numerical safety."""
    w2c_stack = np.stack([np.array(frame["w2c"]) for frame in frames])
    c2ws = np.linalg.inv(w2c_stack)
    # Pinning the last row avoids floating-point drift from inversion
    c2w_out = repeat(torch.eye(4, dtype=torch.float32), 'h w -> b h w', b=len(frames)).clone()
    c2w_out[:, :3] = torch.from_numpy(c2ws).float()[:, :3]
    return c2w_out


def _validate_poses(c2ws: torch.Tensor, label: str) -> None:
    """Raise if poses have extreme translations, NaN determinants, or non-unit rotation scale."""
    if (c2ws[:, :3, 3] > 1e3).any():
        raise ValueError(f"Large translation in {label} poses: {c2ws[:, :3, 3].max()}")
    dets = torch.det(c2ws[:, :3, :3])
    if torch.isnan(dets).any():
        raise ValueError(f"NaN in {label} pose determinants")
    if not torch.allclose(dets, dets.new_ones(dets.shape)):
        raise ValueError(f"Det of {label} poses not equal to 1")


def _build_avg_camera_frame(input_c2ws: torch.Tensor) -> torch.Tensor:
    """Construct an SE(3) frame from the mean input camera orientation and position.

    The axes are built via Gram-Schmidt so they stay orthonormal even when
    individual forward vectors don't average to a unit vector.
    """
    position_avg = input_c2ws[:, :3, 3].mean(0)
    forward_avg = F.normalize(input_c2ws[:, :3, 2].mean(0), dim=0)
    down_raw = input_c2ws[:, :3, 1].mean(0)
    down_avg = F.normalize(down_raw - down_raw.dot(forward_avg) * forward_avg, dim=0)
    right_avg = torch.linalg.cross(down_avg, forward_avg)
    return torch.cat([
        torch.stack([right_avg, down_avg, forward_avg, position_avg], dim=1),
        torch.tensor([[0, 0, 0, 1]], dtype=torch.float32),
    ], dim=0)


def _normalize_poses_to_input_frame(
    input_c2ws: torch.Tensor,
    target_c2ws: torch.Tensor,
    scene_scale_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-express all poses in the average input camera frame and unit-scale translations.

    Centering on the mean input view makes the model's coordinate system
    consistent across scenes regardless of global world-frame placement.

    Args:
        input_c2ws (torch.Tensor): Input camera-to-world matrices.
        target_c2ws (torch.Tensor): Target camera-to-world matrices.
        scene_scale_factor (float): Scene-level scale divisor applied before
            normalizing by the maximum translation magnitude.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Normalized (input_c2ws, target_c2ws).
    """
    avg_frame = _build_avg_camera_frame(input_c2ws)
    world_to_avg = torch.inverse(avg_frame)

    input_c2ws = torch.matmul(world_to_avg.unsqueeze(0), input_c2ws)
    target_c2ws = torch.matmul(world_to_avg.unsqueeze(0), target_c2ws)

    translation_scale = 1.0 / (scene_scale_factor * input_c2ws[:, :3, 3].abs().max())
    input_c2ws[:, :3, 3] *= translation_scale
    target_c2ws[:, :3, 3] *= translation_scale

    return input_c2ws, target_c2ws


# ---------------------------------------------------------------------------
# Frame window + ordering helpers
# ---------------------------------------------------------------------------

def _pick_frame_window(
    num_frames_total: int,
    min_dist: int,
    max_dist: int,
) -> list[int]:
    """Randomly sample a contiguous frame window within the given distance bounds."""
    frame_dist = np.random.randint(min_dist, max_dist + 1)
    start = np.random.randint(0, num_frames_total - frame_dist)
    return list(range(start, start + frame_dist + 1))


def _select_input_frames(
    candidate_indices: list[int],
    selection_type: str,
    num_frames: int,
    scene_eval_info: dict | None = None,
) -> list[int]:
    """Return input frame indices from candidates using the specified selection strategy.

    Args:
        candidate_indices (list[int]): Pool of frame indices to select from.
        selection_type (str): One of 'random', 'uniform', 'kmeans', 'json_context'.
        num_frames (int): Number of input frames to select.
        scene_eval_info (dict | None): Per-scene metadata required by some strategies.

    Returns:
        list[int]: Selected input frame indices.
    """
    if selection_type == 'random':
        return list(np.random.choice(candidate_indices, num_frames, replace=False))
    elif selection_type == 'uniform':
        linspace_idx = np.linspace(0, len(candidate_indices) - 1, num_frames, dtype=int)
        return [candidate_indices[i] for i in linspace_idx]
    elif selection_type == 'kmeans':
        # k-means cluster centers precomputed offline for reproducible diverse coverage
        kmeans_key = f"fold_8_kmeans_{num_frames}_input"
        if not isinstance(scene_eval_info, dict) or kmeans_key not in scene_eval_info:
            raise KeyError(f"input_frame_select_type=kmeans requires key '{kmeans_key}' in scene index entry")
        return list(scene_eval_info[kmeans_key])
    elif selection_type == 'json_context':
        if not isinstance(scene_eval_info, dict) or 'context' not in scene_eval_info:
            raise KeyError("input_frame_select_type=json_context requires scene entry to have 'context' list")
        return list(scene_eval_info['context'])
    else:
        raise NotImplementedError(f"Unknown input_frame_select_type: {selection_type}")


def _select_target_frames(
    candidate_indices: list[int],
    selection_type: str,
    num_frames: int,
    uniform_every: int,
    scene_eval_info: dict | None = None,
) -> list[int]:
    """Return target frame indices from candidates using the specified selection strategy.

    Args:
        candidate_indices (list[int]): Pool of frame indices to select from.
        selection_type (str): One of 'random', 'uniform', 'uniform_every', 'json_target'.
        num_frames (int): Number of target frames to select (unused for 'uniform_every').
        uniform_every (int): Step size used by the 'uniform_every' strategy.
        scene_eval_info (dict | None): Per-scene metadata required by some strategies.

    Returns:
        list[int]: Selected target frame indices.
    """
    start, end = candidate_indices[0], candidate_indices[-1]

    if selection_type == 'random':
        return list(np.random.choice(candidate_indices, num_frames, replace=False))
    elif selection_type == 'uniform':
        return list(np.linspace(start, end, num_frames, dtype=int))
    elif selection_type == 'uniform_every':
        return list(range(start, end + 1, uniform_every))
    elif selection_type == 'json_target':
        if not isinstance(scene_eval_info, dict) or 'target' not in scene_eval_info:
            raise KeyError("target_frame_select_type=json_target requires scene entry to have 'target' list")
        return list(scene_eval_info['target'])
    else:
        raise NotImplementedError(f"Unknown target_frame_select_type: {selection_type}")


def _apply_input_ordering(
    indices: list[int],
    reverse: bool,
    shuffle: bool,
) -> list[int]:
    """Optionally reverse and/or shuffle input frame order for training-time augmentation."""
    if reverse:
        indices = indices[::-1]
    if shuffle:
        np.random.shuffle(indices)
    return indices
