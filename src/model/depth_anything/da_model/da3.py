# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.transforms as T
from addict import Dict
from omegaconf import OmegaConf
from huggingface_hub import PyTorchModelHubMixin

from src.model.depth_anything.da_utils.cfg import create_object, load_config
from src.model.depth_anything.da_model.utils.transform import pose_encoding_to_extri_intri
from src.model.depth_anything.da_utils.registry import MODEL_REGISTRY
from src.utils.camera_utils import extrinsics_to_44


def _wrap_cfg(cfg_obj):
    return OmegaConf.create(cfg_obj)


# ---------------------------------------------------------------------------
# Revised-DepthAnything3Net to be used as Geometry Expert
# ---------------------------------------------------------------------------

class DepthAnything3Net(nn.Module):
    """
    Depth Anything 3 network for depth and camera pose estimation.

    Args:
        preset: Configuration preset containing network dimensions and settings

    Returns:
        Dictionary containing extrinsics, fxfycxcy, pose_T, pose_quat, pose_fov_h, pose_fov_w
    """

    PATCH_SIZE = 14

    def __init__(self, net, head, cam_dec=None, cam_enc=None, gs_head=None, gs_adapter=None):
        super().__init__()
        self.backbone = net if isinstance(net, nn.Module) else create_object(_wrap_cfg(net))
        self.head = head if isinstance(head, nn.Module) else create_object(_wrap_cfg(head))
        self.cam_dec, self.cam_enc = None, None
        if cam_dec is not None:
            self.cam_dec = (
                cam_dec if isinstance(cam_dec, nn.Module) else create_object(_wrap_cfg(cam_dec))
            )
            self.cam_enc = (
                cam_enc if isinstance(cam_enc, nn.Module) else create_object(_wrap_cfg(cam_enc))
            )
        self.gs_adapter, self.gs_head = None, None
        if gs_head is not None and gs_adapter is not None:
            self.gs_adapter = (
                gs_adapter
                if isinstance(gs_adapter, nn.Module)
                else create_object(_wrap_cfg(gs_adapter))
            )
            gs_out_dim = self.gs_adapter.d_in + 1
            if isinstance(gs_head, nn.Module):
                assert (
                    gs_head.out_dim == gs_out_dim
                ), f"gs_head.out_dim should be {gs_out_dim}, got {gs_head.out_dim}"
                self.gs_head = gs_head
            else:
                assert (
                    gs_head["output_dim"] == gs_out_dim
                ), f"gs_head output_dim should set to {gs_out_dim}, got {gs_head['output_dim']}"
                self.gs_head = create_object(_wrap_cfg(gs_head))

    def forward(
        self,
        x: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        export_feat_layers: list[int] | None = [],
        ref_view_strategy: str = "middle",
    ) -> Dict[str, torch.Tensor]:
        if extrinsics is not None:
            with torch.autocast(device_type=x.device.type, enabled=False):
                cam_token = self.cam_enc(extrinsics, intrinsics, x.shape[-2:])
        else:
            cam_token = None

        feats, _ = self.backbone(
            x, cam_token=cam_token, export_feat_layers=export_feat_layers, ref_view_strategy=ref_view_strategy
        )
        H, W = x.shape[-2], x.shape[-1]

        with torch.autocast(device_type=x.device.type, enabled=False):
            output = self._process_camera_estimation_ft(feats, H, W)

        return output

    def _process_camera_estimation_ft(
        self, feats: list[torch.Tensor], H: int, W: int
    ) -> Dict[str, torch.Tensor]:
        if self.cam_dec is not None:
            pose_enc = self.cam_dec(feats[-1][1])
            c2w, fxfycxcy, T, quat, fov_h, fov_w = pose_encoding_to_extri_intri(pose_enc, (H, W))

        return Dict(
            extrinsics=c2w,
            fxfycxcy=fxfycxcy,
            pose_T=T,
            pose_quat=quat,
            pose_fov_h=fov_h,
            pose_fov_w=fov_w,
        )
    

# ---------------------------------------------------------------------------
# Revised-DepthAnything3Net model wrapper for PyTorch Hub
# ---------------------------------------------------------------------------

class DepthAnything3(nn.Module, PyTorchModelHubMixin):
    _commit_hash: str | None = None

    def __init__(self, model_name: str = "da3-giant", **kwargs) -> None:
        super().__init__()
        self.model_name = model_name
        self.config = load_config(MODEL_REGISTRY[self.model_name])
        self.model = create_object(self.config)
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x):
        x = self.normalize(x)
        output = self.model(x)
        output["extrinsics"] = extrinsics_to_44(output["extrinsics"])
        return output

    def prune_layers(self):
        if hasattr(self.model, 'head'):
            del self.model.head
        if hasattr(self.model, 'cam_enc'):
            del self.model.cam_enc
