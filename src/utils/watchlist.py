"""Fixed watchlist scenes evaluated on a schedule during refiner training.

Ten held-out test scenes and ten training scenes, evaluated at 6, 12 and 24 context
views. The scenes and their context/target indices never change, so the W&B curves
compare like with like across steps and across runs. Training also draws a 32-view
bucket, but that condition is not logged: the reported benchmark stops at 24.

Test scenes use the benchmark's own context/target indices, so their numbers sit on the
same protocol as the reported 140-scene results. Train scenes are absent from that index,
so indices are generated here with the same shape (N context / 8 target over the same
frame window the benchmark uses for that view count) and then frozen to a json.

Metrics are keyed ``watchlist/{split}_10s_{views}v_avg_{metric}``.
"""

import json
import os

import numpy as np
import torch
import torchvision

from src.datasets.dataset import InferenceDataset
from src.utils.metric_utils import compute_lpips, compute_psnr, compute_ssim

NUM_SCENES = 10
NUM_TGT = 8
# Context-view counts to log, with the frame window the benchmark json uses for each.
# 32 views are trained on but not logged - the evaluation protocol stops at 24.
VIEW_SPANS = {6: 50, 12: 100, 24: 150}


def _pick_indices(num_frames: int, num_ctx: int, span: int) -> dict:
    """Deterministic num_ctx context / NUM_TGT target split over the scene's first span frames."""
    end = min(span, num_frames - 1)
    ctx = sorted(set(np.linspace(0, end, num_ctx).astype(int).tolist()))
    pool = [i for i in range(end + 1) if i not in ctx]
    if len(pool) < NUM_TGT or len(ctx) < num_ctx:
        raise ValueError(f"scene too short for a {num_ctx}-view watchlist: {num_frames} frames")
    tgt = [pool[i] for i in np.linspace(0, len(pool) - 1, NUM_TGT).astype(int)]
    return {"context": ctx, "target": sorted(set(tgt))}


def _write_train_index(scene_paths: list[str], out_path: str, num_ctx: int, span: int) -> str:
    """Freeze context/target indices for the train watchlist scenes into a json."""
    index = {}
    for path in scene_paths:
        with open(path) as f:
            data = json.load(f)
        # RE10K directories are '<split>_<id>' while its eval datasets look indices up by
        # the bare id; DL3DV hashes carry no underscore, so the split is a no-op there.
        key = data["scene_name"].split("_", 1)[-1]
        index[key] = _pick_indices(len(data["frames"]), num_ctx, span)
    with open(out_path, "w") as f:
        json.dump(index, f, indent=1)
    return out_path


def _subset_list(src_list: str, dst_list: str, n: int) -> list[str]:
    """Write the first n scene paths of src_list to dst_list; return them."""
    with open(src_list) as f:
        paths = [line.strip() for line in f if line.strip()][:n]
    with open(dst_list, "w") as f:
        f.write("\n".join(paths) + "\n")
    return paths


def build_watchlists(config, work_dir: str) -> dict:
    """Return {(split, views): InferenceDataset} for every watchlist condition.

    ``config.watchlist.test_json_paths`` maps a context-view count - as a string, since
    the config is wrapped in an edict, which rejects integer keys - to the benchmark json
    that fixes the test scenes' indices at that count.
    """
    from copy import deepcopy
    import importlib

    os.makedirs(work_dir, exist_ok=True)
    json_paths = config.watchlist.test_json_paths
    # watchlist.dataset_name picks the eval dataset class (RE10K stores scenes in a
    # different per-scene format); watchlist.view_spans overrides the DL3DV ladder for
    # benchmarks that define fewer conditions.
    ds_name = config.watchlist.get("dataset_name", "src.datasets.dataset.InferenceDataset")
    mod, cls = ds_name.rsplit(".", 1)
    ds_cls = importlib.import_module(mod).__dict__[cls]
    spans = {int(k): v for k, v in
             config.watchlist.get("view_spans", {str(k): v for k, v in VIEW_SPANS.items()}).items()}
    datasets = {}

    test_list = os.path.join(work_dir, "watchlist_test.txt")
    _subset_list(config.watchlist.test_data_path, test_list, NUM_SCENES)
    train_list = os.path.join(work_dir, "watchlist_train.txt")
    train_paths = _subset_list(config.data.data_path, train_list, NUM_SCENES)

    for views, span in spans.items():
        base = deepcopy(config)
        base.data.input_frame_select_type = "json_context"
        base.data.target_frame_select_type = "json_target"
        base.data.num_input_frames = views
        base.data.num_target_frames = NUM_TGT

        test_cfg = deepcopy(base)
        test_cfg.data.data_path = test_list
        test_cfg.data.json_path = json_paths[str(views)]
        datasets[("test", views)] = ds_cls(test_cfg)

        train_cfg = deepcopy(base)
        train_cfg.data.data_path = train_list
        train_cfg.data.json_path = _write_train_index(
            train_paths, os.path.join(work_dir, f"watchlist_train_index_{views}v.json"), views, span)
        datasets[("train", views)] = ds_cls(train_cfg)

    return datasets


@torch.no_grad()
def evaluate(core, datasets: dict, device, amp, out_dir: str, step: int, save_renders: bool) -> dict:
    """Evaluate every watchlist condition and return flat metrics keyed for W&B.

    Averages run over all target views of all scenes, not over per-scene means, so a scene
    contributes in proportion to how many views it has.
    """
    core.eval()
    metrics = {}
    for (split, views), dataset in datasets.items():
        psnr, ssim, lpips = [], [], []
        for idx in range(len(dataset)):
            sample = dataset[idx]
            batch = {k: (v[None].to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in sample.items()}
            input_data = {k: v[:, :views] if isinstance(v, torch.Tensor) else v
                          for k, v in batch.items()}
            target_data = {k: v[:, views:] if isinstance(v, torch.Tensor) else None
                           for k, v in batch.items()}
            with amp:
                out = core(input_data, target_data)
            pred = out["render"][0].float().clamp(0, 1)
            gt = target_data["image"][0].float()
            psnr.append(compute_psnr(gt, pred).reshape(-1))
            ssim.append(compute_ssim(gt, pred).reshape(-1))
            lpips.append(compute_lpips(gt, pred).reshape(-1))

            if save_renders:
                scene_dir = os.path.join(out_dir, "watchlist", f"{split}_{views}v",
                                         f"step_{step:06d}", f"{idx:02d}_{sample['scene_name'][:16]}")
                os.makedirs(scene_dir, exist_ok=True)
                for i in range(pred.shape[0]):
                    torchvision.utils.save_image(pred[i], os.path.join(scene_dir, f"{i}.png"))

        tag = f"watchlist/{split}_10s_{views}v_avg"
        metrics[f"{tag}_psnr"] = torch.cat(psnr).mean().item()
        metrics[f"{tag}_ssim"] = torch.cat(ssim).mean().item()
        metrics[f"{tag}_lpips"] = torch.cat(lpips).mean().item()
    core.train()
    return metrics
