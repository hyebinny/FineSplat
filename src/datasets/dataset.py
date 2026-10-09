import os
import torch
import json
import random
import traceback
import numpy as np
from dataclasses import dataclass
from torch.utils.data import Dataset

from src.datasets.scene_lists import resolve_scene_paths

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


@dataclass
class FrameSelectionConfig:
    """Parameters governing how input and target frames are selected from a scene."""

    input_select_type: str
    target_select_type: str
    num_input: int
    num_target: int
    target_has_input: bool
    min_dist: int
    max_dist: int
    shuffle_input_prob: float
    reverse_input_prob: float
    target_uniform_every: int


def _validate_frame_selection_config(cfg: FrameSelectionConfig) -> None:
    """Assert that a FrameSelectionConfig is internally consistent.

    Args:
        cfg: The frame selection config to validate.
    """
    assert cfg.min_dist <= cfg.max_dist
    if cfg.num_target == 0:
        assert cfg.target_select_type in ['uniform_every', 'json_target']
    min_required = (
        max(cfg.num_input, cfg.num_target) - 1
        if cfg.target_has_input
        else cfg.num_input + cfg.num_target - 1
    )
    assert cfg.min_dist >= min_required


class InferenceDataset(Dataset):
    """Dataset for inference over scenes listed in a text file."""

    def __init__(self, config):
        """
        Args:
            config: OmegaConf config containing data settings for inference.
        """
        self.config = config
        self.scene_json_paths = resolve_scene_paths(config.data)
        self.scene_dict = self._load_scene_eval_index(config.data.get("json_path", None))


    @staticmethod
    def _load_scene_eval_index(json_path: str) -> dict:
        """Load the eval metadata file and index it by scene name for O(1) lookup.

        Args:
            json_path: Path to a JSON file containing eval metadata as a list or dict.

        Returns:
            A dict mapping scene_name strings to their metadata entries.
        """
        with open(json_path, "r") as f:
            eval_data = json.load(f)
        if isinstance(eval_data, list):
            return {item["scene_name"]: item for item in eval_data}
        if isinstance(eval_data, dict):
            return eval_data
        raise ValueError(f"Unsupported eval index format: {type(eval_data)}")

    def __len__(self) -> int:
        """Return the number of scenes in the dataset."""
        return len(self.scene_json_paths)

    def process_frames(
        self,
        frames: list[dict],
        image_base_dir: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load images and camera data for a list of frames, applying configured resize/crop.

        Args:
            frames: List of frame dicts from the scene JSON, each containing image path
                and camera parameters.
            image_base_dir: Directory used to resolve relative image paths.

        Returns:
            A tuple of (images, intrinsics, c2ws) where images is (N, C, H, W),
            intrinsics is (N, 4), and c2ws is (N, 4, 4).
        """
        target_h = self.config.data.get("resize_h", -1)
        target_w = self.config.data.get("resize_w", -1)
        square_crop = self.config.data.square_crop
        resize_mode = self.config.data.get("resize_mode", "lanczos")

        image_tensors, intrinsics_list = [], []
        for frame in frames:
            image_tensor, intrinsics = _load_frame_image_and_intrinsics(
                frame, image_base_dir, target_h, target_w, square_crop, resize_mode
            )
            image_tensors.append(image_tensor)
            intrinsics_list.append(intrinsics)

        images = torch.stack(image_tensors, dim=0)
        intrinsics = torch.from_numpy(np.array(intrinsics_list, dtype=np.float32))
        c2ws = _w2c_stack_to_c2w(frames)
        return images, intrinsics, c2ws

    def _read_frame_selection_config(self, num_frames_total: int) -> FrameSelectionConfig:
        """Extract frame selection parameters from config, clamping distances to available frames.

        Args:
            num_frames_total: Total number of frames in the current scene.

        Returns:
            A FrameSelectionConfig populated from the data config.
        """
        data_cfg = self.config.data
        min_dist = data_cfg.get("min_frame_dist", "all")
        max_dist = data_cfg.get("max_frame_dist", 256)

        if min_dist == "all":
            min_dist = max_dist = num_frames_total - 1

        return FrameSelectionConfig(
            input_select_type=data_cfg.input_frame_select_type,
            target_select_type=data_cfg.target_frame_select_type,
            num_input=data_cfg.num_input_frames,
            num_target=data_cfg.get("num_target_frames", 0),
            target_has_input=data_cfg.get("target_has_input", False),
            min_dist=min(min_dist, num_frames_total - 1),
            max_dist=min(max_dist, num_frames_total - 1),
            shuffle_input_prob=data_cfg.get("shuffle_input_prob", 0.0),
            reverse_input_prob=data_cfg.get("reverse_input_prob", 0.0),
            target_uniform_every=data_cfg.get("target_uniform_every", 8),
        )

    def __getitem__(self, idx: int) -> dict:
        """Return a scene sample, falling back to a random scene on error.

        Args:
            idx: Index into scene_json_paths.

        Returns:
            A dict with image, intrinsic, pose, index, and scene_name tensors.
        """
        try:
            return self._load_scene(idx)
        except Exception:
            traceback.print_exc()
            print(f"error loading data: {self.scene_json_paths[idx]}")
            return self.__getitem__(random.randint(0, len(self) - 1))

    def _load_scene(self, idx: int) -> dict:
        """Load and process a single scene at the given index.

        Args:
            idx: Index into scene_json_paths.

        Returns:
            A dict with concatenated input+target image, intrinsic, pose, and index
            tensors, plus the scene_name string.

        Raises:
            KeyError: If the scene is not present in the eval index.
            ValueError: If poses contain NaN or Inf after normalization.
        """
        scene_path = self.scene_json_paths[idx]
        with open(scene_path, 'r') as f:
            data_json = json.load(f)

        scene_name = data_json['scene_name']
        frames = data_json['frames']
        image_base_dir = scene_path.rsplit('/', 1)[0]

        if scene_name not in self.scene_dict:
            raise KeyError(f"Scene '{scene_name}' not found in eval index")
        scene_eval_info = self.scene_dict[scene_name]

        sel_cfg = self._read_frame_selection_config(len(frames))
        _validate_frame_selection_config(sel_cfg)

        candidate_indices = _pick_frame_window(len(frames), sel_cfg.min_dist, sel_cfg.max_dist)

        target_frame_idx = sorted(_select_target_frames(
            candidate_indices, sel_cfg.target_select_type, sel_cfg.num_target,
            sel_cfg.target_uniform_every, scene_eval_info,
        ))

        if not sel_cfg.target_has_input:
            candidate_indices = [x for x in candidate_indices if x not in target_frame_idx]

        input_frame_idx = sorted(_select_input_frames(
            candidate_indices, sel_cfg.input_select_type, sel_cfg.num_input, scene_eval_info,
        ))
        input_frame_idx = _apply_input_ordering(
            input_frame_idx,
            reverse=np.random.rand() < sel_cfg.reverse_input_prob,
            shuffle=np.random.rand() < sel_cfg.shuffle_input_prob,
        )

        target_images, target_intr, target_c2ws = self.process_frames(
            [frames[i] for i in target_frame_idx], image_base_dir
        )
        input_images, input_intr, input_c2ws = self.process_frames(
            [frames[i] for i in input_frame_idx], image_base_dir
        )

        _validate_poses(target_c2ws, "target")
        _validate_poses(input_c2ws, "input")

        scene_scale_factor = self.config.data.get("scene_scale", 1.0)
        input_c2ws, target_c2ws = _normalize_poses_to_input_frame(
            input_c2ws, target_c2ws, scene_scale_factor
        )

        for label, c2ws in [("input", input_c2ws), ("target", target_c2ws)]:
            if torch.isnan(c2ws).any() or torch.isinf(c2ws).any():
                raise ValueError(f"NaN or Inf in {label} poses after normalization")

        return {
            "image": torch.cat([input_images, target_images], dim=0),
            "fxfycxcy": torch.cat([input_intr, target_intr], dim=0),
            "c2w": torch.cat([input_c2ws, target_c2ws], dim=0),
            "index": torch.tensor(input_frame_idx + target_frame_idx).long().unsqueeze(-1),
            "scene_name": scene_name,
        }
