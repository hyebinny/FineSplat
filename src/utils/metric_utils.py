import os
import json
import warnings
import imageio
import torch
import torch.nn.functional as F
import torchvision
import functools
import numpy as np
import PIL.Image as Image
from rich import print
from torch import Tensor
from einops import reduce, rearrange
from skimage.metrics import structural_similarity
from easydict import EasyDict as edict


# Suppress warnings for LPIPS loss loading
warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated since 0.13")
warnings.filterwarnings("ignore", category=UserWarning, message="Arguments other than a weight enum.*")


@torch.no_grad()
def compute_psnr(
    ground_truth: Tensor,
    predicted: Tensor,
) -> Tensor:
    """Compute Peak Signal-to-Noise Ratio between ground truth and predicted images.

    Args:
        ground_truth (Tensor): Images with shape [batch, channel, height, width],
            values in [0, 1].
        predicted (Tensor): Images with shape [batch, channel, height, width],
            values in [0, 1].

    Returns:
        Tensor: PSNR values for each image in the batch.
    """
    ground_truth = torch.clamp(ground_truth, 0, 1)
    predicted = torch.clamp(predicted, 0, 1)
    mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
    return -10 * torch.log10(mse)


@functools.lru_cache(maxsize=None)
def get_lpips_model(net_type: str = "vgg", device: str = "cuda"):
    """Load and cache an LPIPS model on the specified device.

    Args:
        net_type (str): Backbone network type passed to LPIPS (e.g. 'vgg', 'alex').
        device (str): Device string to move the model to.

    Returns:
        LPIPS: The loaded perceptual loss model.
    """
    from lpips import LPIPS
    return LPIPS(net=net_type).to(device)


@torch.no_grad()
def compute_lpips(
    ground_truth: Tensor,
    predicted: Tensor,
    normalize: bool = True,
) -> Tensor:
    """Compute Learned Perceptual Image Patch Similarity between images.

    Args:
        ground_truth (Tensor): Images with shape [batch, channel, height, width].
        predicted (Tensor): Images with shape [batch, channel, height, width].
            Values should be in [0, 1] when normalize=True, or [-1, 1] otherwise.
        normalize (bool): If True, rescale inputs from [0, 1] to [-1, 1] internally.

    Returns:
        Tensor: LPIPS values for each image in the batch (lower is better).
    """
    _lpips_fn = get_lpips_model(device=predicted.device)
    batch_size = 10  # Process in batches to save memory
    values = [
        _lpips_fn(
            ground_truth[i : i + batch_size],
            predicted[i : i + batch_size],
            normalize=normalize,
        )
        for i in range(0, ground_truth.shape[0], batch_size)
    ]
    return torch.cat(values, dim=0).squeeze()


@torch.no_grad()
def compute_ssim(
    ground_truth: Tensor,
    predicted: Tensor,
) -> Tensor:
    """Compute Structural Similarity Index between images.

    Args:
        ground_truth (Tensor): Images with shape [batch, channel, height, width],
            values in [0, 1].
        predicted (Tensor): Images with shape [batch, channel, height, width],
            values in [0, 1].

    Returns:
        Tensor: SSIM values for each image in the batch (higher is better).
    """
    ssim_values = []

    for gt, pred in zip(ground_truth, predicted):
        gt_np = gt.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()

        ssim = structural_similarity(
            gt_np,
            pred_np,
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
        ssim_values.append(ssim)

    return torch.tensor(ssim_values, dtype=predicted.dtype, device=predicted.device)



# Metric-time resizing. 2Xplat's evaluation protocol names a resolution and a filter for
# the metric stage, separately from the resolution the images were loaded at; a number is
# only comparable against another taken under the same pair. Ported from 2Xplat so the two
# repositories report the same figure for the same checkpoint.
DOWNSAMPLE_MODES = {
    "bilinear_aa": dict(mode="bilinear", align_corners=False, antialias=True),
    "bilinear": dict(mode="bilinear", align_corners=False),
    "bicubic_aa": dict(mode="bicubic", align_corners=False, antialias=True),
    "bicubic": dict(mode="bicubic", align_corners=False),
    "area": dict(mode="area"),
}


def _resize_lanczos(images: Tensor, resolution: int) -> Tensor:
    """Resize images with PIL Lanczos, matching the data loader's uint8 convention.

    Images are quantized to uint8 before resampling, exactly as _resize_to_cover in
    dataset_utils does, so the metric resize and the loaded ground truth agree.
    """
    images_u8 = (images.clamp(0, 1) * 255).round().to(torch.uint8)
    images_u8 = rearrange(images_u8, "v c h w -> v h w c").cpu().numpy()
    resized = np.stack([
        np.array(Image.fromarray(image).resize((resolution, resolution), Image.LANCZOS))
        for image in images_u8
    ])
    resized = torch.from_numpy(resized).to(images.device)
    return rearrange(resized, "v h w c -> v c h w").float() / 255.0


def _resize(images: Tensor, resolution: int, downsample: str) -> Tensor:
    """Resize images to a square resolution using the named interpolation method."""
    if downsample == "lanczos":
        return _resize_lanczos(images, resolution)
    if downsample not in DOWNSAMPLE_MODES:
        raise ValueError(
            f"Unknown metric_downsample '{downsample}'; "
            f"expected 'lanczos' or one of {sorted(DOWNSAMPLE_MODES)}"
        )
    resized = F.interpolate(images, size=(resolution, resolution), **DOWNSAMPLE_MODES[downsample])
    return resized.clamp(0, 1)


def resize_for_metrics(images: Tensor, resolution: int | None,
                       downsample: str = "bilinear_aa") -> Tensor:
    """Resize [views, 3, H, W] for the metric stage, or return it untouched.

    No-op when ``resolution`` is None or the images are already square at it, which is how
    2Xplat's own configs leave it for same-resolution evaluation.
    """
    if resolution is None or images.shape[-2:] == (resolution, resolution):
        return images
    return _resize(images.to(torch.float32), resolution, downsample)


@torch.no_grad()
def export_results(
    result: edict,
    out_dir: str,
    compute_metrics: bool = False,
    visualize: bool = False,
    metric_resolution: int | None = None,
    metric_downsample: str = "bilinear_aa",
    uid: int = 0
) -> None:
    """Save results including images and optional metrics and videos.

    Args:
        result (edict): EasyDict containing input, target, and rendered images,
            and optionally video frames.
        out_dir (str): Directory to save the evaluation results.
        compute_metrics (bool): Whether to compute and save metrics.
        visualize (bool): Whether to save target and rendered images as PNGs.
        uid (int): Integer prefix used to name the per-sample output directory.
    """
    os.makedirs(out_dir, exist_ok=True)

    target_data = result.target
    rendered_image = result.render
    input_data = result.input
    b, v, _, h, w = rendered_image.size()

    for batch_idx in range(input_data["image"].size(0)):
        scene_name = input_data["scene_name"][0]
        sample_dir = os.path.join(out_dir, f"{uid:06d}_{scene_name}")
        os.makedirs(sample_dir, exist_ok=True)

        target_indices = target_data["index"][batch_idx, :].cpu().numpy().squeeze(-1).astype(int)
        input_indices = input_data["index"][batch_idx, :].cpu().numpy().squeeze(-1).astype(int)
        target_indices_path = os.path.join(sample_dir, "target_indices.txt")
        input_indices_path = os.path.join(sample_dir, "input_indices.txt")
        np.savetxt(target_indices_path, target_indices, fmt="%d")
        np.savetxt(input_indices_path, input_indices, fmt="%d")

        if visualize:
            os.makedirs(os.path.join(sample_dir, "target"), exist_ok=True)
            os.makedirs(os.path.join(sample_dir, "rendering"), exist_ok=True)
            for i in range(v):
                target_path = os.path.join(sample_dir, "target", f"{i}.png")
                rendering_path = os.path.join(sample_dir, "rendering", f"{i}.png")
                torchvision.utils.save_image(
                    target_data["image"][batch_idx, i], target_path
                )
                torchvision.utils.save_image(
                    rendered_image[batch_idx, i], rendering_path
                )

        if compute_metrics:
            _save_metrics(
                target_data["image"][batch_idx],
                rendered_image[batch_idx],
                target_indices,
                sample_dir,
                scene_name,
                metric_resolution,
                metric_downsample,
            )


def _save_metrics(target: Tensor, prediction: Tensor, view_indices: np.ndarray, out_dir: str,
                  scene_name: str, metric_resolution: int | None = None,
                  metric_downsample: str = "bilinear_aa") -> None:
    """Compute PSNR, LPIPS, and SSIM for a single sample and write them to metrics.json.

    Args:
        target (Tensor): Ground-truth images with shape [views, channels, height, width].
        prediction (Tensor): Rendered images with shape [views, channels, height, width].
        view_indices (np.ndarray): Integer frame indices corresponding to each view.
        out_dir (str): Directory where metrics.json will be written.
        scene_name (str): Scene identifier stored in the summary.
        metric_resolution (int): If set, resize both images to this square resolution
            before computing metrics.
        metric_downsample (str): Resize method, 'lanczos' or a key of DOWNSAMPLE_MODES.
    """
    target = resize_for_metrics(target.to(torch.float32), metric_resolution, metric_downsample)
    prediction = resize_for_metrics(prediction.to(torch.float32), metric_resolution, metric_downsample)

    # Deliberately unclamped. Rendered colours do leave [0, 1], and compute_psnr clamps
    # internally while SSIM and LPIPS do not, so the three are not scored on quite the same
    # image - worth about 0.00002 SSIM and 0.0001 LPIPS. 2Xplat's own _save_metrics
    # (2Xplat's utils/metric_utils.py) does the same, and the baseline rows have to stay
    # comparable with the numbers that repo reports.
    psnr_values = compute_psnr(target, prediction)
    lpips_values = compute_lpips(target, prediction)
    ssim_values = compute_ssim(target, prediction)

    metrics = {
        "summary": {
            "scene_name": scene_name,
            "psnr": float(psnr_values.mean()),
            "lpips": float(lpips_values.mean()),
            "ssim": float(ssim_values.mean())
        },
        "per_view": []
    }

    for i, view_idx in enumerate(view_indices):
        metrics["per_view"].append({
            "view": int(view_idx), "psnr": float(psnr_values[i]), "lpips": float(lpips_values[i]), "ssim": float(ssim_values[i])
        })

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def create_video_from_frames(frames: np.ndarray, output_video_file: str, framerate: int = 30) -> None:
    """Create a video file from a sequence of image frames.

    Args:
        frames (np.ndarray): Array of image frames with shape (N, H, W, C).
            Values may be float in [0, 1] or uint8 in [0, 255].
        output_video_file (str): Path to save the output video file.
        framerate (int): Frames per second for the video. Default is 30.
    """
    frames = np.asarray(frames)

    # Normalize frames if values are in [0,1] range
    if frames.max() <= 1:
        frames = (frames * 255).astype(np.uint8)

    imageio.mimsave(output_video_file, frames, fps=framerate, quality=8)


def _save_video(frames: Tensor, out_dir: str) -> None:
    """Save rendered frames as a video file to out_dir/rendered_video.mp4.

    Args:
        frames (Tensor): Rendered frames with shape [v, c, h, w].
        out_dir (str): Directory where the video file will be written.
    """
    frames = np.ascontiguousarray(np.array(frames.to(torch.float32)))
    frames = rearrange(frames, "v c h w -> v h w c")
    create_video_from_frames(
        frames,
        f"{out_dir}/rendered_video.mp4",
        framerate=30
    )


def summarize_evaluation(evaluation_folder: str) -> None:
    """Aggregate per-scene metrics.json files into a summary CSV and average_metrics.txt.

    Args:
        evaluation_folder (str): Path to the directory containing per-scene subfolders,
            each of which should contain a metrics.json file.
    """
    subfolders = sorted(
        [
            os.path.join(evaluation_folder, dirname)
            for dirname in os.listdir(evaluation_folder)
            if os.path.isdir(os.path.join(evaluation_folder, dirname))
        ],
        key=lambda x: int(os.path.basename(x)) if os.path.basename(x).isdigit() else os.path.basename(x)
    )

    metrics = {}
    valid_subfolders = []

    for subfolder in subfolders:
        json_path = os.path.join(subfolder, "metrics.json")
        if not os.path.exists(json_path):
            print(f"!!! Metrics file not found in {subfolder}, skipping...")
            continue

        valid_subfolders.append(subfolder)

        with open(json_path, "r") as f:
            try:
                data = json.load(f)
                for metric_name, metric_value in data["summary"].items():
                    if metric_name == "scene_name":
                        continue
                    metrics.setdefault(metric_name, []).append(metric_value)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error reading metrics from {json_path}: {e}")

    if not valid_subfolders:
        print(f"No valid metrics files found in {evaluation_folder}")
        return

    csv_file = os.path.join(evaluation_folder, "summary.csv")
    with open(csv_file, "w") as f:
        header = ["Index"] + list(metrics.keys())
        f.write(",".join(header) + "\n")

        for i, subfolder in enumerate(valid_subfolders):
            basename = os.path.basename(subfolder)
            values = [str(metric_values[i]) for metric_values in metrics.values()]
            f.write(f"{basename},{','.join(values)}\n")

        f.write("\n")

        averages = [str(sum(values) / len(values)) for values in metrics.values()]
        f.write(f"average,{','.join(averages)}\n")

    print(f"Summary written to {csv_file}")
    print(f"Average: {','.join(averages)}")

    with open(os.path.join(evaluation_folder, "average_metrics.txt"), "w") as f:
        f.write(f"Average: {','.join(averages)}\n")
