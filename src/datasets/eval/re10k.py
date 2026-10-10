"""RealEstate10K evaluation over WAI-converted per-scene directories.

Each scene directory holds ``images/`` and a ``scene_meta.json`` whose frames carry an
OpenCV camera-to-world ``transform_matrix``, pixel-space ``fl_x/fl_y/cx/cy`` and an
absolute ``file_path``. Directories are named '<split>_<scene_id>'; the eval index
(src/data/re10k_ctx_*v_tgt_*v.json) is keyed by the bare scene_id.
"""

import json
import random
import traceback

import numpy as np
import torch
from PIL import Image
from einops import repeat
from torch.utils.data import Dataset

from src.utils.dataset_utils import (
    _prepare_image_and_intrinsics,
    _validate_poses,
    _normalize_poses_to_input_frame,
    _select_input_frames,
    _select_target_frames,
    _apply_input_ordering,
)


class RE10KEvalDataset(Dataset):
    """Dataset for inference over RealEstate10K scenes stored as WAI scene_meta.json."""

    def __init__(self, config):
        """``data.data_path``: text file of scene_meta.json paths; ``data.json_path``: eval index."""
        self.config = config
        with open(config.data.data_path, "r") as f:
            self.scene_json_paths = [line.strip() for line in f if line.strip()]
        if not self.scene_json_paths:
            raise FileNotFoundError(f"No scenes listed in {config.data.data_path}")
        with open(config.data.json_path, "r") as f:
            self.scene_dict = json.load(f)

    def __len__(self) -> int:
        return len(self.scene_json_paths)

    def process_frames(
        self,
        frames: list[dict],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load and resize/crop frames. Returns images (N,C,H,W), intrinsics (N,4) fxfycxcy, c2ws (N,4,4)."""
        target_h = self.config.data.get("resize_h", -1)
        target_w = self.config.data.get("resize_w", -1)
        square_crop = self.config.data.square_crop

        image_tensors, intrinsics_list = [], []
        for frame in frames:
            raw_intrinsics = np.array([frame["fl_x"], frame["fl_y"], frame["cx"], frame["cy"]])
            image_tensor, intrinsics = _prepare_image_and_intrinsics(
                Image.open(frame["file_path"]), raw_intrinsics,
                target_h, target_w, square_crop,
                resize_mode="lanczos",
            )
            image_tensors.append(image_tensor)
            intrinsics_list.append(intrinsics)

        images = torch.stack(image_tensors, dim=0)
        intrinsics = torch.from_numpy(np.array(intrinsics_list, dtype=np.float32))

        # transform_matrix is already c2w; pin the last row against stored round-off.
        c2ws = repeat(torch.eye(4, dtype=torch.float32), 'h w -> b h w', b=len(frames)).clone()
        c2ws[:, :3] = torch.tensor(
            np.array([frame["transform_matrix"] for frame in frames]), dtype=torch.float32
        )[:, :3]
        return images, intrinsics, c2ws

    def __getitem__(self, idx: int) -> dict:
        """Return a scene sample, falling back to a random scene on error."""
        try:
            return self._load_scene(idx)
        except Exception:
            traceback.print_exc()
            print(f"error loading data: {self.scene_json_paths[idx]}")
            return self.__getitem__(random.randint(0, len(self) - 1))

    def _load_scene(self, idx: int) -> dict:
        """Load one scene; image/fxfycxcy/c2w/index hold input frames then target frames."""
        with open(self.scene_json_paths[idx], "r") as f:
            scene_meta = json.load(f)

        scene_name = scene_meta["scene_name"]
        frames = scene_meta["frames"]
        # Directories are '<split>_<scene_id>'; the eval index is keyed by scene_id.
        scene_key = scene_name.split("_", 1)[-1]

        if scene_key not in self.scene_dict:
            raise KeyError(f"Scene '{scene_key}' not found in eval index")
        scene_eval_info = self.scene_dict[scene_key]

        data_cfg = self.config.data
        candidate_indices = list(range(len(frames)))

        target_frame_idx = sorted(_select_target_frames(
            candidate_indices,
            data_cfg.target_frame_select_type,
            data_cfg.get("num_target_frames", 0),
            data_cfg.get("target_uniform_every", 8),
            scene_eval_info,
        ))

        if not data_cfg.get("target_has_input", False):
            candidate_indices = [x for x in candidate_indices if x not in target_frame_idx]

        input_frame_idx = sorted(_select_input_frames(
            candidate_indices,
            data_cfg.input_frame_select_type,
            data_cfg.num_input_frames,
            scene_eval_info,
        ))
        input_frame_idx = _apply_input_ordering(
            input_frame_idx,
            reverse=np.random.rand() < data_cfg.get("reverse_input_prob", 0.0),
            shuffle=np.random.rand() < data_cfg.get("shuffle_input_prob", 0.0),
        )

        target_images, target_intr, target_c2ws = self.process_frames(
            [frames[i] for i in target_frame_idx]
        )
        input_images, input_intr, input_c2ws = self.process_frames(
            [frames[i] for i in input_frame_idx]
        )

        _validate_poses(target_c2ws, "target")
        _validate_poses(input_c2ws, "input")

        input_c2ws, target_c2ws = _normalize_poses_to_input_frame(
            input_c2ws, target_c2ws, data_cfg.get("scene_scale", 1.0)
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
