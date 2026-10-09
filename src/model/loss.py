# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import os
import torch
import lpips
import scipy.io
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from pathlib import Path
from easydict import EasyDict as edict
from torchvision.models import vgg19

from src.utils.camera_utils import invert_SE3


# ---------------------------------------------------------------------------
# Camera loss helpers
# ---------------------------------------------------------------------------

def rot_ang_loss(R: torch.Tensor, Rgt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Mean geodesic rotation error between predicted and GT rotation matrices.

    Args:
        R:   Predicted rotation matrices (B, 3, 3).
        Rgt: Ground-truth rotation matrices (B, 3, 3).

    Returns:
        Scalar mean angular error in radians, in [0, π].
    """
    residual = torch.matmul(R.transpose(1, 2), Rgt)
    trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
    cosine = (trace - 1) / 2
    R_err = torch.acos(torch.clamp(cosine, -1.0 + eps, 1.0 - eps))
    return R_err.mean()


def compute_relative_camera_loss(cam_info: dict) -> dict:
    """Huber translation loss + geodesic rotation loss on all pairwise relative poses."""
    pred_pose = cam_info["pred_c2w"]
    gt_pose = cam_info["gt_c2w"]

    _, N, _, _ = pred_pose.shape

    pred_w2c = invert_SE3(pred_pose)
    gt_w2c = invert_SE3(gt_pose)

    pred_w2c_exp = pred_w2c.unsqueeze(2)    # (B, V, 1, 4, 4)
    pred_pose_exp = pred_pose.unsqueeze(1)  # (B, 1, V, 4, 4)
    gt_w2c_exp = gt_w2c.unsqueeze(2)        # (B, V, 1, 4, 4)
    gt_pose_exp = gt_pose.unsqueeze(1)      # (B, 1, V, 4, 4)

    pred_rel_all = torch.matmul(pred_w2c_exp, pred_pose_exp)  # (B, V, V, 4, 4)
    gt_rel_all = torch.matmul(gt_w2c_exp, gt_pose_exp)        # (B, V, V, 4, 4)

    # Exclude self-relative poses (diagonal).
    mask = ~torch.eye(N, dtype=torch.bool, device=pred_pose.device)

    t_pred = pred_rel_all[..., :3, 3][:, mask, ...]
    R_pred = pred_rel_all[..., :3, :3][:, mask, ...]
    t_gt = gt_rel_all[..., :3, 3][:, mask, ...]
    R_gt = gt_rel_all[..., :3, :3][:, mask, ...]

    trans_loss = F.huber_loss(t_pred, t_gt, reduction="mean", delta=0.1)
    rot_loss = rot_ang_loss(R_pred.reshape(-1, 3, 3), R_gt.reshape(-1, 3, 3))

    return dict(trans_loss=trans_loss, rot_loss=rot_loss)


def compute_intrinsics_loss(cam_info: dict) -> torch.Tensor:
    """MSE loss on predicted vs. GT focal lengths, normalised by image width."""
    pred_fxfy = cam_info["pred_fxfycxcy"][..., :2]
    gt_fxfy = cam_info["gt_fxfycxcy"][..., :2]
    return F.mse_loss(pred_fxfy / 224.0, gt_fxfy / 224.0)


# ---------------------------------------------------------------------------
# Perceptual loss module
# ---------------------------------------------------------------------------

class PerceptualLoss(nn.Module):
    """
    VGG19-based perceptual loss with MatConvNet weights.

    Modified from https://github.com/zhengqili/Crowdsampling-the-Plenoptic-Function
    and https://github.com/arthurhero/Long-LRM/blob/main/model/loss.py
    """

    def __init__(self, device="cpu"):
        """Build VGG19 feature extractor with MatConvNet weights on ``device``."""
        super().__init__()
        self.device = device
        self.vgg = self._build_vgg()
        self._load_weights()
        self._setup_feature_blocks()

    def _build_vgg(self):
        """Create VGG19 with average pooling instead of max pooling."""
        model = vgg19()
        for i, layer in enumerate(model.features):
            if isinstance(layer, nn.MaxPool2d):
                model.features[i] = nn.AvgPool2d(kernel_size=2, stride=2)
        return model.to(self.device).eval()

    def _load_weights(self):
        """Download and load pre-trained MatConvNet VGG19 weights."""
        weight_file = Path("pretrained_weights/imagenet-vgg-verydeep-19.mat")
        weight_file.parent.mkdir(exist_ok=True, parents=True)

        if dist.is_initialized():
            if torch.distributed.get_rank() == 0:
                if not weight_file.exists():
                    os.system(f"wget https://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat -O {weight_file}")
            torch.distributed.barrier()
        else:
            if not weight_file.exists():
                os.system(f"wget https://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat -O {weight_file}")

        vgg_data = scipy.io.loadmat(weight_file)
        vgg_layers = vgg_data["layers"][0]

        layer_indices = [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]
        filter_sizes = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]

        with torch.no_grad():
            for i, layer_idx in enumerate(layer_indices):
                weights = torch.from_numpy(vgg_layers[layer_idx][0][0][2][0][0]).permute(3, 2, 0, 1)
                self.vgg.features[layer_idx].weight = nn.Parameter(weights, requires_grad=False)

                biases = torch.from_numpy(vgg_layers[layer_idx][0][0][2][0][1]).view(filter_sizes[i])
                self.vgg.features[layer_idx].bias = nn.Parameter(biases, requires_grad=False)

    def _setup_feature_blocks(self):
        """Slice VGG into sequential blocks at fixed depth boundaries and freeze all params."""
        output_indices = [0, 4, 9, 14, 23, 32]
        self.blocks = nn.ModuleList()

        for i in range(len(output_indices) - 1):
            block = nn.Sequential(*list(self.vgg.features[output_indices[i]:output_indices[i + 1]]))
            self.blocks.append(block.to(self.device).eval())

        for param in self.vgg.parameters():
            param.requires_grad = False

    def _extract_features(self, x: torch.Tensor):
        """Pass images through each VGG block and collect intermediate feature maps."""
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features

    def _preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """Scale [0, 1] images to VGG's expected mean-subtracted pixel range."""
        mean = torch.tensor([123.6800, 116.7790, 103.9390]).reshape(1, 3, 1, 1).to(images.device)
        return images * 255.0 - mean

    @staticmethod
    def _compute_error(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        """Return the mean absolute difference between two feature tensors."""
        return torch.mean(torch.abs(real - fake))

    def forward(self, pred_img: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
        """Compute multi-scale perceptual loss between prediction and target."""
        target_img_p = self._preprocess_images(target_img)
        pred_img_p = self._preprocess_images(pred_img)

        target_features = self._extract_features(target_img_p)
        pred_features = self._extract_features(pred_img_p)

        e0 = self._compute_error(target_img_p, pred_img_p)
        e1 = self._compute_error(target_features[0], pred_features[0]) / 2.6
        e2 = self._compute_error(target_features[1], pred_features[1]) / 4.8
        e3 = self._compute_error(target_features[2], pred_features[2]) / 3.7
        e4 = self._compute_error(target_features[3], pred_features[3]) / 5.6
        e5 = self._compute_error(target_features[4], pred_features[4]) * 10 / 1.5

        return (e0 + e1 + e2 + e3 + e4 + e5) / 255.0


# ---------------------------------------------------------------------------
# Loss computer
# ---------------------------------------------------------------------------

class LossComputer(nn.Module):
    """Aggregate all training losses: photometric (L2 + LPIPS + perceptual), pose, and intrinsics."""

    def __init__(self, config):
        """Instantiate active loss modules based on their configured weights.

        Args:
            config: OmegaConf config with a ``training`` section containing
                loss weight fields (``lpips_loss_weight``, etc.).
        """
        super().__init__()
        self.config = config

        if self.config.training.lpips_loss_weight > 0.0:
            # Avoid multiple GPUs downloading the same LPIPS model concurrently.
            if torch.distributed.get_rank() == 0:
                self.lpips_loss_module = self._init_frozen_module(lpips.LPIPS(net="vgg"))
            torch.distributed.barrier()
            if torch.distributed.get_rank() != 0:
                self.lpips_loss_module = self._init_frozen_module(lpips.LPIPS(net="vgg"))

        if self.config.training.perceptual_loss_weight > 0.0:
            self.perceptual_loss_module = self._init_frozen_module(PerceptualLoss())

    def _init_frozen_module(self, module: nn.Module) -> nn.Module:
        """Set a module to eval mode and freeze all its parameters."""
        module.eval()
        for param in module.parameters():
            param.requires_grad = False
        return module

    def forward(
        self,
        rendering: torch.Tensor,   # (B, V, 3, H, W), range [0, 1]
        target: torch.Tensor,      # (B, V, 3, H, W), range [0, 1]
        cam_info: dict = None,
    ) -> edict:
        """Compute the weighted sum of all active losses.

        Args:
            rendering: Predicted images in [0, 1], shape (B, V, 3, H, W).
            target: Ground-truth images in [0, 1], shape (B, V, 3, H, W).
                May have 4 channels (RGBA); the alpha channel is discarded.
            cam_info: Dict with predicted and GT poses/intrinsics; required when
                any camera loss weight is non-zero.

        Returns:
            An edict containing ``loss`` (total weighted loss) plus individual
            loss terms and normalised variants.
        """
        b, v, _, h, w = rendering.size()
        rendering = rendering.reshape(b * v, -1, h, w)
        target = target.reshape(b * v, -1, h, w)

        if target.size(1) == 4:
            target, _ = target.split([3, 1], dim=1)

        l2_loss = torch.tensor(1e-8).to(rendering.device)
        if self.config.training.l2_loss_weight > 0.0:
            l2_loss = F.mse_loss(rendering, target)

        psnr = -10.0 * torch.log10(l2_loss)

        lpips_loss = torch.tensor(0.0).to(l2_loss.device)
        if self.config.training.lpips_loss_weight > 0.0:
            # LPIPS expects inputs in [-1, 1].
            lpips_loss = self.lpips_loss_module(
                rendering * 2.0 - 1.0, target * 2.0 - 1.0
            ).mean()

        perceptual_loss = torch.tensor(0.0).to(l2_loss.device)
        if self.config.training.perceptual_loss_weight > 0.0:
            perceptual_loss = self.perceptual_loss_module(rendering, target)

        pose_loss_dict = dict(
            trans_loss=torch.tensor(0.0).to(l2_loss.device),
            rot_loss=torch.tensor(0.0).to(l2_loss.device),
        )
        if self.config.training.trans_loss_weight > 0.0 or self.config.training.rot_loss_weight > 0.0:
            pose_loss_dict = compute_relative_camera_loss(cam_info)

        intrinsics_loss = torch.tensor(0.0).to(l2_loss.device)
        if self.config.training.intrinsics_loss_weight > 0.0:
            intrinsics_loss = compute_intrinsics_loss(cam_info)

        loss = (
            self.config.training.l2_loss_weight * l2_loss
            + self.config.training.lpips_loss_weight * lpips_loss
            + self.config.training.perceptual_loss_weight * perceptual_loss
            + self.config.training.trans_loss_weight * pose_loss_dict["trans_loss"]
            + self.config.training.rot_loss_weight * pose_loss_dict["rot_loss"]
            + self.config.training.intrinsics_loss_weight * intrinsics_loss
        )

        return edict(
            loss=loss,
            l2_loss=l2_loss,
            psnr=psnr,
            lpips_loss=lpips_loss,
            perceptual_loss=perceptual_loss,
            trans_loss=pose_loss_dict["trans_loss"],
            rot_loss=pose_loss_dict["rot_loss"],
            intrinsics_loss=intrinsics_loss,
            norm_perceptual_loss=perceptual_loss / l2_loss,
            norm_lpips_loss=lpips_loss / l2_loss,
        )
