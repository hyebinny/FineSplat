"""Where experiment output goes.

    <experiments>/eval/     scores and renders, laid out by protocol
    <experiments>/train/    checkpoints and watchlists, one directory per run
    <experiments>/wandb/    W&B local files

Under ``eval`` every result is filed by its settings:

    <H>res/<resize_mode>/<dataset>/<N>v/<camera_mode>/<run>[/<ckpt_tag>]

from ``data.resize_h``, ``data.resize_mode``, ``inference.dataset`` (or the data list
name), ``data.num_input_frames``, ``inference.camera_mode`` (``_epa`` appended when
``inference.pose_optimization`` is set), ``inference.run`` and ``inference.ckpt_tag``.
``resize_mode`` sits high because the resize filter changes PSNR by more than Stage 2 does.
"""

import re
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO_ROOT.parent / "analysis"
EVAL_ROOT = EXPERIMENTS / "eval"
TRAIN_ROOT = EXPERIMENTS / "train"

def load_eval_config(argv=None):
    """``<config> [out_dir] [key=value ...]`` -> (config, out_dir or None); dotlist overrides are merged."""
    args = list(sys.argv[1:] if argv is None else argv)
    config = OmegaConf.load(args[0])
    out_dir = [a for a in args[1:] if "=" not in a]
    assert len(out_dir) <= 1, f"one output directory at most: {out_dir}"
    overrides = [a for a in args[1:] if "=" in a]
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
    return config, (Path(out_dir[0]) if out_dir else None)


DATASETS = ("dl3dv", "re10k", "scannetpp", "mipnerf360")
CAMERA_MODES = ("gt_pose_gt_intr", "pred_pose_gt_intr", "pred_pose_pred_intr")


def dataset_of(config) -> str:
    """The benchmark a config evaluates on, from an explicit key or the data list name."""
    name = config.get("inference", {}).get("dataset")
    if name:
        return name
    path = str(config.data.get("data_path", "") or config.data.get("json_path", ""))
    for d in DATASETS:
        if d in Path(path).name.lower() or d in path.lower():
            return d
    raise ValueError(f"cannot tell which dataset {path!r} belongs to; "
                     f"set inference.dataset to one of {DATASETS}")


def ckpt_tag(config) -> str | None:
    """A level under the run for a specific step checkpoint; None for ``*_last.pt`` / ``*_best.pt``."""
    tag = config.get("inference", {}).get("ckpt_tag")
    if tag:
        return tag
    path = config.get("inference", {}).get("checkpoint", {}).get("ckpt_path")
    if not path:
        return None
    m = re.fullmatch(r"(?:finesplat|refiner)_step_0*(\d+)", Path(path).stem)
    return f"step{m.group(1)}" if m else None


def eval_out_dir(config, run: str, root: Path | str = EVAL_ROOT) -> Path:
    """The directory this evaluation's results belong in."""
    inf = config.get("inference", {})
    mode = inf.get("camera_mode", "gt_pose_gt_intr")
    if mode not in CAMERA_MODES:
        raise ValueError(f"unknown camera_mode {mode!r}, expected one of {CAMERA_MODES}")
    if inf.get("pose_optimization", False):
        mode += "_epa"
    out = (Path(root) / f"{config.data.resize_h}res"
           / config.data.get("resize_mode", "lanczos") / dataset_of(config)
           / f"{config.data.num_input_frames}v" / mode / run)
    tag = ckpt_tag(config)
    return out / tag if tag else out


def train_out_dir(config, root: Path | str = TRAIN_ROOT) -> Path:
    """``train.out_dir`` if set, else ``<experiments>/train/<train.run>``."""
    train = config.get("train", {})
    if train.get("out_dir"):
        return Path(train["out_dir"])
    run = train.get("run")
    if not run:
        raise ValueError("set either train.out_dir or train.run")
    return Path(root) / run


def resolve_out_dir(config) -> Path:
    """``inference.out_dir`` if set, else ``eval_out_dir`` under ``inference.run``."""
    inf = config.get("inference", {})
    if inf.get("out_dir"):
        return Path(inf["out_dir"])
    run = inf.get("run")
    if not run:
        raise ValueError("set either inference.out_dir or inference.run")
    return eval_out_dir(config, run)


def wandb_dir(root: Path | str = EXPERIMENTS) -> str:
    """wandb appends its own ``wandb/`` to whatever it is given."""
    Path(root).mkdir(parents=True, exist_ok=True)
    return str(root)


if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "data": {"data_path": "src/data/dl3dv_test.txt", "resize_h": 224, "num_input_frames": 6},
        "inference": {"camera_mode": "pred_pose_gt_intr", "pose_optimization": True},
    })
    got = eval_out_dir(cfg, "gtPose_patch8_Eself_Eagg", root="/tmp/x")
    assert str(got) == ("/tmp/x/224res/lanczos/dl3dv/6v/pred_pose_gt_intr_epa/"
                        "gtPose_patch8_Eself_Eagg"), got
    cfg.inference.pose_optimization = False
    cfg.data.resize_h = 252
    assert str(eval_out_dir(cfg, "r", root="/tmp/x")) == \
        "/tmp/x/252res/lanczos/dl3dv/6v/pred_pose_gt_intr/r"
    cfg.data.resize_mode = "bilinear_antialias"
    assert str(eval_out_dir(cfg, "r", root="/tmp/x")) == \
        "/tmp/x/252res/bilinear_antialias/dl3dv/6v/pred_pose_gt_intr/r"
    cfg.inference.checkpoint = {"ckpt_path": "/a/b/finesplat_step_002000.pt"}
    assert eval_out_dir(cfg, "r", root="/tmp/x").parts[-2:] == ("r", "step2000")
    cfg.inference.checkpoint = {"ckpt_path": "/a/b/finesplat_last.pt"}
    assert eval_out_dir(cfg, "r", root="/tmp/x").name == "r"
    print("ok", got)
