import os
import json
import torch
import numpy as np

from src.datasets.scene_lists import resolve_scene_paths
from src.datasets.base_dataset import BaseDataset
from src.utils.dataset_utils import (
    _load_frame_image_and_intrinsics,
    _w2c_stack_to_c2w,
    _validate_poses,
    _normalize_poses_to_input_frame,
    _pick_frame_window,
    _select_input_frames,
    _select_target_frames,
    _apply_input_ordering,
)


# ---------------------------------------------------------------------------
# Frame distance bounds (training-specific, keyed by input view count)
# ---------------------------------------------------------------------------

# Empirically determined bounds that ensure sufficient visual overlap per view count.
_FRAME_DIST_BY_NUM_INPUTS: dict[int, tuple[int, int]] = {
    2:  (30,  50),
    6:  (30,  50),
    12: (50,  100),
    24: (100, 150),
    32: (150, 200),
}


def _frame_dist_bounds(num_input_views: int) -> tuple[int, int]:
    """Return (min_dist, max_dist) frame window bounds for the given input view count.

    Args:
        num_input_views: The number of input views to be sampled.

    Returns:
        A (min_dist, max_dist) tuple of frame index distance bounds.

    Raises:
        NotImplementedError: If no bounds are defined for num_input_views.
    """
    if num_input_views not in _FRAME_DIST_BY_NUM_INPUTS:
        raise NotImplementedError(
            f"No frame distance bounds defined for num_input_views={num_input_views}. "
            "Add an entry to _FRAME_DIST_BY_NUM_INPUTS."
        )
    return _FRAME_DIST_BY_NUM_INPUTS[num_input_views]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DynamicInputDataset(BaseDataset):
    """Training dataset that supports dynamic multi-resolution batching."""

    def __init__(self, config):
        """
        Args:
            config: OmegaConf config containing training and data settings.
        """
        super().__init__(config)
        self.config = config
        self.evaluation = config.get("evaluation", False)
        self.data_path = resolve_scene_paths(config.data)
        self._scene_len()

    def _scene_len(self):
        """Set num_of_scenes from the loaded data_path list."""
        self.num_of_scenes = len(self.data_path)


    def process_frames(
        self,
        frames: list[dict],
        image_base_dir: str,
        target_h: int,
        target_w: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load images and camera data for a list of frames, applying configured resize/crop.

        Args:
            frames: List of frame dicts from the scene JSON, each containing image path
                and camera parameters.
            image_base_dir: Directory used to resolve relative image paths.
            target_h: Target image height after resizing.
            target_w: Target image width after resizing.

        Returns:
            A tuple of (images, intrinsics, c2ws) where images is (N, C, H, W),
            intrinsics is (N, 4), and c2ws is (N, 4, 4).
        """
        square_crop = self.config.data.square_crop
        resize_mode = self.config.data.get("resize_mode", "lanczos")

        image_tensors, intrinsics_list = [], []
        for frame in frames:
            image_tensor, intrinsics = _load_frame_image_and_intrinsics(
                frame, image_base_dir, target_h, target_w, square_crop, resize_mode
            )
            image_tensors.append(image_tensor)
            intrinsics_list.append(torch.from_numpy(intrinsics).float())

        images = torch.stack(image_tensors, dim=0)
        intrinsics = torch.stack(intrinsics_list, dim=0)
        c2ws = _w2c_stack_to_c2w(frames)
        return images, intrinsics, c2ws

    def _get_views(
        self,
        sampled_idx: int,
        resolution: tuple[int, int],
        num_views_to_input: int,
        num_views_to_target: int,
    ) -> dict:
        """Load a scene and select input/target frames at the specified resolution.

        Args:
            sampled_idx: Index into data_path for the scene to load.
            resolution: (height, width) tuple for image loading.
            num_views_to_input: Number of input frames to select.
            num_views_to_target: Number of target frames to select.

        Returns:
            A dict with separate input/target image, intrinsic, pose, and index tensors,
            plus the scene_name string.

        Raises:
            ValueError: If poses contain NaN or Inf after normalization.
        """
        scene_path = self.data_path[sampled_idx]
        with open(scene_path, 'r') as f:
            data_json = json.load(f)

        scene_name = data_json['scene_name']
        frames = data_json['frames']
        image_base_dir = scene_path.rsplit('/', 1)[0]
        target_h, target_w = int(resolution[0]), int(resolution[1])

        input_select_type = self.config.data.input_frame_select_type
        target_select_type = self.config.data.target_frame_select_type
        target_has_input = self.config.data.target_has_input
        scene_scale_factor = self.config.data.get("scene_scale", 1.0)
        uniform_every = self.config.data.get("target_uniform_every", 8)

        min_dist, max_dist = _frame_dist_bounds(num_views_to_input)
        min_dist = min(min_dist, len(frames) - 1)
        max_dist = min(max_dist, len(frames) - 1)
        assert min_dist <= max_dist
        min_required = (
            max(num_views_to_input, num_views_to_target) - 1
            if target_has_input
            else num_views_to_input + num_views_to_target - 1
        )
        assert min_dist >= min_required

        candidate_indices = _pick_frame_window(len(frames), min_dist, max_dist)

        target_frame_idx = sorted(_select_target_frames(
            candidate_indices, target_select_type, num_views_to_target, uniform_every,
        ))

        if not target_has_input:
            candidate_indices = [x for x in candidate_indices if x not in target_frame_idx]

        input_frame_idx = sorted(_select_input_frames(
            candidate_indices, input_select_type, num_views_to_input,
        ))
        input_frame_idx = _apply_input_ordering(
            input_frame_idx,
            reverse=np.random.rand() < self.config.data.get("reverse_input_prob", 0.0),
            shuffle=np.random.rand() < self.config.data.get("shuffle_input_prob", 0.0),
        )

        target_images, target_intr, target_c2ws = self.process_frames(
            [frames[i] for i in target_frame_idx], image_base_dir, target_h, target_w
        )
        input_images, input_intr, input_c2ws = self.process_frames(
            [frames[i] for i in input_frame_idx], image_base_dir, target_h, target_w
        )

        _validate_poses(target_c2ws, "target")
        _validate_poses(input_c2ws, "input")

        input_c2ws, target_c2ws = _normalize_poses_to_input_frame(
            input_c2ws, target_c2ws, scene_scale_factor
        )

        for label, c2ws in [("input", input_c2ws), ("target", target_c2ws)]:
            if torch.isnan(c2ws).any() or torch.isinf(c2ws).any():
                raise ValueError(f"NaN or Inf in {label} poses after normalization")

        return {
            "input_image": input_images,
            "input_fxfycxcy": input_intr,
            "input_c2w": input_c2ws,
            "target_image": target_images,
            "target_fxfycxcy": target_intr,
            "target_c2w": target_c2ws,
            "input_indices": torch.tensor(input_frame_idx).long().unsqueeze(-1),
            "target_indices": torch.tensor(target_frame_idx).long().unsqueeze(-1),
            "scene_name": scene_name,
        }
