"""Where experiment output goes.

Three kinds of output, three roots, so none of them buries the others:

    <experiments>/eval/     scores and renders, laid out by protocol
    <experiments>/train/    checkpoints and watchlists, one directory per run
    <experiments>/wandb/    whatever W&B writes locally

Under ``eval`` one rule, applied everywhere, so a result can be found from its settings
alone:

    <H>res/<resize_mode>/<dataset>/<N>v/<camera_mode>/<run>

``resize_mode`` sits high because it is not a property of the model being scored. DL3DV_480
is 480x270, so every evaluation resizes, and the filter moves 6-view PSNR from 26.09 to
27.02 - more than Stage 2 itself. Numbers from two filters are not comparable, so they
never share a parent.

``camera_mode`` gains an ``_epa`` suffix when the poses are realigned against the
predicted Gaussians before scoring, since that changes the number without changing any
config field the path already carries. ``run`` names the checkpoint or baseline being
scored - the fields above pin down the protocol but not the model, and a directory
holding two different models' scores under one name is the state this layout replaces.
"""

import re
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
# Everything produced from here on. The earlier tree is frozen beside it as 3Xplat_exp;
# nothing writes there any more, so a result's root says which era it belongs to.
EXPERIMENTS = REPO_ROOT.parent / "FineSplat_exp_final"
EVAL_ROOT = EXPERIMENTS / "eval"
TRAIN_ROOT = EXPERIMENTS / "train"

def load_eval_config(argv=None):
    """``<config> [out_dir] [key=value ...]`` -> (config, out_dir or None).

    Dotlist overrides (``inference.scene_range=[0,70]``) are merged into the config; the
    one argument without ``=`` is an explicit output directory.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    config = OmegaConf.load(args[0])
    out_dir = [a for a in args[1:] if "=" not in a]
    assert len(out_dir) <= 1, f"one output directory at most: {out_dir}"
    overrides = [a for a in args[1:] if "=" in a]
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
    return config, (Path(out_dir[0]) if out_dir else None)


DATASETS = ("dl3dv", "re10k", "scannetpp")
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
    """A level under the run for a specific checkpoint, or None for the run's latest.

    Scoring several checkpoints of one training run is a question about that run, so they
    belong under it rather than beside it as ``<run>_step2000``. ``finesplat_last.pt`` and
    ``finesplat_best.pt`` get no level - they are what "the run" means with no qualifier.
    """
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
    """Where a training run keeps its checkpoints: ``<experiments>/train/<run>``.

    ``train.out_dir`` still wins if set, for runs that belong somewhere else entirely.
    """
    train = config.get("train", {})
    if train.get("out_dir"):
        return Path(train["out_dir"])
    run = train.get("run")
    if not run:
        raise ValueError("set either train.out_dir or train.run")
    return Path(root) / run


def resolve_out_dir(config) -> Path:
    """Where this eval writes: ``inference.out_dir`` if given, else the rule above.

    Configs that name a directory outright keep it - one-off renders and the visualize/
    scripts have no business in the results tree. Everything else supplies
    ``inference.run`` and is filed automatically.
    """
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
