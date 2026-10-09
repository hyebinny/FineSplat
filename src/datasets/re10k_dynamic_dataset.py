"""RealEstate10K training dataset with dynamic multi-resolution batching.

RE10K scenes are stored as WAI-converted per-scene directories holding `images/`
plus a `scene_meta.json`. Their frame entries differ from the DL3DV ones consumed by
DynamicInputDataset: intrinsics are named `fl_x/fl_y/cx/cy`, `file_path` is absolute,
and the pose is an OpenCV camera-to-world `transform_matrix` rather than a `w2c`.

Only frame loading differs, so window selection, pose normalization and the dynamic
batching machinery are inherited unchanged. Frame loading matches
src/datasets/eval/re10k.py so train and eval preprocessing stay comparable.
"""

import numpy as np
import torch
from PIL import Image
from einops import repeat

from src.datasets.dynamic_dataset import DynamicInputDataset
from src.utils.dataset_utils import _prepare_image_and_intrinsics


class RE10KDynamicInputDataset(DynamicInputDataset):
    """Training dataset over RE10K scene_meta.json scenes with dynamic batching."""

    def frame_dist_bounds(self, num_input_views: int, num_frames: int) -> tuple[int, int]:
        """Return the frame window bounds, using one fixed range for every view count.

        RE10K clips are short handheld pans, so the DL3DV policy of widening the window
        with the input view count does not transfer. This matches the reference RE10K
        loader instead: a single [min_frame_dist, max_frame_dist] range shared by all
        view counts, with only the upper bound shrunk to fit the scene. Scenes too short
        to span min_frame_dist are rejected rather than collapsed to their full extent,
        so every sample keeps a minimum camera baseline.

        Args:
            num_input_views: Unused; the range does not vary with view count.
            num_frames: Total number of frames in the scene.

        Returns:
            A (min_dist, max_dist) tuple of frame index distance bounds.

        Raises:
            ValueError: If the scene is too short, which BaseDataset retries on.
        """
        view_cfg = self.config.data.get("view_selector", {})
        min_dist = view_cfg.get("min_frame_dist", 45)
        max_dist = min(num_frames - 1, view_cfg.get("max_frame_dist", 192))
        # Reference rejects on `max <= min`; scenes under 20 frames are subsumed by this.
        if max_dist <= min_dist:
            raise ValueError(
                f"Scene with {num_frames} frames is too short for "
                f"min_frame_dist={min_dist}"
            )
        return min_dist, max_dist

    def process_frames(
        self,
        frames: list[dict],
        image_base_dir: str,
        target_h: int,
        target_w: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load images and camera data for a list of RE10K frames.

        Args:
            frames: Frame dicts from a scene_meta.json, each carrying `file_path`,
                `fl_x`/`fl_y`/`cx`/`cy`, and a c2w `transform_matrix`.
            image_base_dir: Unused; RE10K frames store absolute image paths.
            target_h: Target image height after resizing.
            target_w: Target image width after resizing.

        Returns:
            A tuple of (images, intrinsics, c2ws) where images is (N, C, H, W),
            intrinsics is (N, 4), and c2ws is (N, 4, 4).
        """
        square_crop = self.config.data.square_crop

        image_tensors, intrinsics_list = [], []
        for frame in frames:
            raw_intrinsics = np.array([frame["fl_x"], frame["fl_y"], frame["cx"], frame["cy"]])
            image_tensor, intrinsics = _prepare_image_and_intrinsics(
                Image.open(frame["file_path"]), raw_intrinsics,
                target_h, target_w, square_crop,
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
