"""ScanNet++ v2 iPhone evaluation over pixelSplat-style .torch chunks.

The eval indices in src/data/scannetpp_ctx_*v_tgt_16v.json come from YoNoSplat and are
defined over the iPhone captures of the 50 ScanNet++ v2 `sem_test` scenes, keyed
'<scene>_iphone'. The matching pixels ship as one .torch chunk per scene, holding
JPEG-encoded frames plus a (num_frames, 18) camera tensor: normalized fx/fy/cx/cy, the
source width/height, and a flattened 3x4 OpenCV w2c.

Preprocessing (resize/crop, pose normalization) is shared with the DL3DV inference path so
the two eval sets stay comparable.
"""

import glob
import json
import os
import random
import traceback
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.utils.dataset_utils import (
    _prepare_image_and_intrinsics,
    _w2c_stack_to_c2w,
    _validate_poses,
    _normalize_poses_to_input_frame,
    _select_input_frames,
    _select_target_frames,
    _apply_input_ordering,
)


class ScanNetppEvalDataset(Dataset):
    """Dataset for inference over ScanNet++ v2 iPhone scenes stored as .torch chunks."""

    def __init__(self, config):
        """
        Args:
            config: Config whose data.data_path points at the directory of per-scene
                .torch chunks and data.json_path at the eval index naming the context and
                target views for each scene.
        """
        self.config = config
        self.chunk_paths = sorted(glob.glob(os.path.join(config.data.data_path, "*.torch")))
        if not self.chunk_paths:
            raise FileNotFoundError(f"No .torch chunks found under {config.data.data_path}")
        with open(config.data.json_path, "r") as f:
            self.scene_dict = json.load(f)

    def __len__(self) -> int:
        """Return the number of scenes in the dataset."""
        return len(self.chunk_paths)

    def process_frames(
        self,
        example: dict,
        frame_indices: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode the given frames of a chunk and return images, intrinsics, and poses.

        Args:
            example: The scene entry loaded from a .torch chunk.
            frame_indices: Positions of the frames to decode.

        Returns:
            A tuple of (images, intrinsics, c2ws) where images is (N, C, H, W),
            intrinsics is (N, 4), and c2ws is (N, 4, 4).
        """
        target_h = self.config.data.get("resize_h", -1)
        target_w = self.config.data.get("resize_w", -1)
        square_crop = self.config.data.square_crop
        cameras = example["cameras"]

        image_tensors, intrinsics_list = [], []
        for frame_idx in frame_indices:
            image = Image.open(BytesIO(example["images"][frame_idx].numpy().tobytes()))
            # Chunk intrinsics are normalized by the source resolution.
            orig_w, orig_h = image.size
            fx, fy, cx, cy = cameras[frame_idx, :4].tolist()
            raw_intrinsics = np.array([fx * orig_w, fy * orig_h, cx * orig_w, cy * orig_h])

            image_tensor, intrinsics = _prepare_image_and_intrinsics(
                image, raw_intrinsics, target_h, target_w, square_crop,
                self.config.data.get("resize_mode", "lanczos"),
            )
            image_tensors.append(image_tensor)
            intrinsics_list.append(intrinsics)

        images = torch.stack(image_tensors, dim=0)
        intrinsics = torch.from_numpy(np.array(intrinsics_list, dtype=np.float32))

        w2cs = np.tile(np.eye(4), (len(frame_indices), 1, 1))
        w2cs[:, :3] = cameras[frame_indices, 6:].reshape(-1, 3, 4).numpy()
        c2ws = _w2c_stack_to_c2w([{"w2c": w2c} for w2c in w2cs])
        return images, intrinsics, c2ws

    def __getitem__(self, idx: int) -> dict:
        """Return a scene sample, falling back to a random scene on error.

        Args:
            idx: Index into chunk_paths.

        Returns:
            A dict with image, intrinsic, pose, index, and scene_name tensors.
        """
        try:
            return self._load_scene(idx)
        except Exception:
            traceback.print_exc()
            print(f"error loading data: {self.chunk_paths[idx]}")
            return self.__getitem__(random.randint(0, len(self) - 1))

    def _load_scene(self, idx: int) -> dict:
        """Load and process the scene held by the chunk at the given index.

        Args:
            idx: Index into chunk_paths.

        Returns:
            A dict with concatenated input+target image, intrinsic, pose, and index
            tensors, plus the scene_name string.

        Raises:
            KeyError: If the scene is not present in the eval index.
            ValueError: If poses contain NaN or Inf after normalization.
        """
        example = torch.load(self.chunk_paths[idx], weights_only=False)[0]
        scene_name = example["key"]

        if scene_name not in self.scene_dict:
            raise KeyError(f"Scene '{scene_name}' not found in eval index")
        scene_eval_info = self.scene_dict[scene_name]

        data_cfg = self.config.data
        # Chunks hold one scene each, so every frame is a selection candidate.
        candidate_indices = list(range(len(example["images"])))

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

        target_images, target_intr, target_c2ws = self.process_frames(example, target_frame_idx)
        input_images, input_intr, input_c2ws = self.process_frames(example, input_frame_idx)

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
