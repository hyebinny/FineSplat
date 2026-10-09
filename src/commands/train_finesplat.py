"""Train FineSplat: Stage 2 on a frozen Stage 1, or the whole model end-to-end.

Default (``stage2.e2e: false``): Stage 1 (2Xplat) and the ResNet-18 error backbone stay
frozen; only ErrorMapEncoder's projections, ErrorLifter and the Stage-2 network are
optimized, against a photometric loss on the held-out target views. The Stage-2 head is
zero-initialised, so step 0 reproduces Stage 1 exactly - the first logged loss is the
baseline. With ``stage2.e2e: true`` Stage 1 is unfrozen and trained jointly.

    torchrun --nproc_per_node 4 -m src.commands.train_finesplat --config src/configs/finesplat/train/stage2_224.yaml
    torchrun --nproc_per_node 8 -m src.commands.train_finesplat --config src/configs/finesplat/train/e2e_224.yaml
"""

import os
import time

import torch
import yaml
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from src.commands.inference import AMP_DTYPE, load_model
from src.utils.checkpoints import checkpoint_state
from src.commands.train import build_input_target_dicts, set_epoch
from src.datasets import get_train_data_loader
from src.model.loss import PerceptualLoss
from src.model.stage2 import FineSplat, default_conf
from src.utils import watchlist
from src.utils.eval_paths import train_out_dir, wandb_dir
from src.utils.init_utils import init_config
from src.utils.metric_utils import compute_lpips, compute_psnr, compute_ssim, get_lpips_model
from copy import deepcopy

from src.utils.training_utils import create_optimizer, update_ema
from src.utils.watchlist import build_watchlists


def _deep_merge(base: dict, over: dict) -> dict:
    """Recursive dict merge. init_config() hands back an edict, which OmegaConf.merge rejects."""
    out = {k: (_deep_merge(v, {}) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in (over or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def build_finesplat(base, config):
    """Instantiate the model named by ``stage2.arch``, merged onto its own defaults.

    "ptv3" is FineSplat's Stage 2 in src.model.stage2; "resplat" is the recurrent
    kNN-attention port in src.model.resplat_refiner (not shipped). They take different
    config trees, so each merges against its own module-level defaults.
    """
    raw = config.get("stage2", {})
    # init_config() hands back edicts, but callers that pass OmegaConf straight through would
    # otherwise reach _deep_merge as DictConfig, which fails isinstance(dict) and silently
    # replaces a whole defaults subtree instead of merging into it.
    if not isinstance(raw, dict):
        raw = OmegaConf.to_container(raw, resolve=True)
    conf = dict(raw)
    arch = conf.pop("arch", "ptv3")
    if arch == "ptv3":
        merged = _deep_merge(default_conf, conf)
        return FineSplat(base, merged), merged
    if arch == "resplat":
        from src.model.resplat_refiner import ReSplatRefiner, default_conf as resplat_conf

        merged = _deep_merge(resplat_conf, conf)
        return ReSplatRefiner(base, merged), merged
    raise ValueError(f"unknown stage2.arch: {arch!r} (expected 'ptv3' or 'resplat')")


def init_wandb(config, resume_id: str | None = None):
    """Start a W&B run from the repo's api_keys.yaml. Returns the wandb module, or None.

    init_wandb_and_backup() is not reused: it also mirrors the source tree and dumps the
    config, and demands a handful of `training.*` keys this run has no other use for.
    """
    conf = config.train.get("wandb", None)
    if not conf or not conf.get("project", None):
        return None
    import wandb

    key_path = conf.get("api_key_path", "src/configs/api_keys.yaml")
    if os.path.exists(key_path):
        key = yaml.safe_load(open(key_path)).get("wandb")
        if key:
            os.environ["WANDB_API_KEY"] = key
    wandb.init(
        dir=wandb_dir(conf["dir"]) if conf.get("dir") else wandb_dir(),
        id=resume_id,
        resume="allow" if resume_id else None,
        project=conf["project"],
        entity=conf.get("entity", None),
        name=conf.get("name", None),
        group=conf.get("group", None),
        config=dict(config),
    )
    return wandb


_PANEL_PAD = 8      # gap between tiles
_PANEL_LABEL = 16   # label strip above each tile


def _panel_font():
    """DejaVu at a readable size; PIL's bitmap default is the fallback."""
    from PIL import ImageFont

    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except OSError:
        return ImageFont.load_default()


def log_images(wandb, out, target_data, core, step: int, max_views: int | None = None) -> None:
    """Log GT | Stage 1 | Stage 2 for the current batch's first scene, each tile
    labelled with its (psnr, ssim, lpips) against the GT view above it.

    Defaults to every target view, i.e. exactly the views the loss is computed on.
    """
    from PIL import Image, ImageDraw

    base_rgb = core.render_baseline_targets(out["gaussians_base"], out["cameras"])
    v = target_data["image"].shape[1] if max_views is None else min(max_views, target_data["image"].shape[1])
    gt = target_data["image"][0, :v].float().clamp(0, 1)
    base = base_rgb[0, :v].float().clamp(0, 1)
    ref = out["render"][0, :v].detach().float().clamp(0, 1)

    scores = {}
    for name, pred in (("stage1", base), ("stage2", ref)):
        psnr = compute_psnr(gt, pred).reshape(-1).tolist()
        ssim = compute_ssim(gt, pred).reshape(-1).tolist()
        lpips = compute_lpips(gt, pred).reshape(-1).tolist()
        scores[name] = list(zip(psnr, ssim, lpips))

    h, w = gt.shape[-2:]
    cell_h = _PANEL_LABEL + h
    canvas = Image.new(
        "RGB",
        (3 * w + 4 * _PANEL_PAD, v * cell_h + (v + 1) * _PANEL_PAD),
        (255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)
    font = _panel_font()

    def to_pil(t):
        return Image.fromarray((t.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))

    for row in range(v):
        y = _PANEL_PAD + row * (cell_h + _PANEL_PAD)
        for col, (name, img) in enumerate((("GT", gt), ("stage1", base), ("stage2", ref))):
            x = _PANEL_PAD + col * (w + _PANEL_PAD)
            label = name if name == "GT" else (
                f"{name} ({scores[name][row][0]:5.2f}, "
                f"{scores[name][row][1]:.3f}, {scores[name][row][2]:.3f})"
            )
            draw.text((x, y + 2), label, fill=(0, 0, 0), font=font)
            canvas.paste(to_pil(img[row]), (x, y + _PANEL_LABEL))

    wandb.log({"targets/gt|stage1|stage2": wandb.Image(canvas, caption="(psnr, ssim, lpips) vs GT")},
              step=step)


def stage2_state_dict(core) -> dict:
    """Trainable weights only. The frozen Stage 1 lives under `model.` and would otherwise
    turn every checkpoint into a 5.7 GB copy of a checkpoint we already have on disk."""
    return {k: v for k, v in core.state_dict().items() if not k.startswith("model.")}


def setup_distributed() -> tuple[int, int, int]:
    """Return (rank, world_size, local_rank); initialises the process group if launched by torchrun."""
    if "RANK" not in os.environ:
        return 0, 1, 0
    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank


def lr_at(step: int, base_lr: float, warmup: float, total: int,
          schedule: str = "constant", decay_ratio: float = 0.2) -> float:
    """Linear warm-up, then either cosine decay to zero (2Xplat's schedule) or a flat rate.

    ``warmup`` is a step count, or a fraction of ``total`` when below 1 - ReSplat expresses
    the same thing as ``warm_up_ratio: 0.01`` on its OneCycleLR. 2Xplat itself warms up over
    0 steps, but it is fine-tuning a trained model at lr 2e-5, while Stage 2 starts
    from a zero-initialised head at a much larger lr and takes a visible hit on its first
    update without one.

    The cosine branch is ``get_cosine_schedule_with_warmup`` written out, which is what
    2Xplat's own training uses (``scheduler_type`` defaults to cosine in
    src/commands/train.py).

    ``constant`` exists for runs whose step budget is not yet fixed: cosine reaches zero at
    ``total``, so such a run cannot be extended, only restarted.
    """
    warmup = warmup * total if 0 < warmup < 1 else warmup
    if warmup > 0 and step < warmup:
        # min() guards the case where the ratio lands below one step: dividing by a
        # fractional warm-up would otherwise return a multiple of the peak lr.
        return base_lr * min(1.0, (step + 1) / warmup)
    if schedule == "constant":
        return base_lr
    if schedule == "wsd":
        # Warmup-stable-decay: hold the peak, then anneal over the last decay_ratio of the
        # budget. Cosine reaches zero exactly at `total`, so a cosine run can only be
        # restarted, never extended; a wsd run can be carried further by moving `total`,
        # because everything before the decay window is flat.
        decay_steps = max(1.0, decay_ratio * total)
        decay_start = total - decay_steps
        if step < decay_start:
            return base_lr
        progress = (step - decay_start) / decay_steps
    else:
        progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.141592653589793)).item())


def main() -> None:
    config = init_config()
    rank, world, local_rank = setup_distributed()
    device = torch.device("cuda", local_rank)
    is_main = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(config.train.get("seed", 1) + rank)

    base = load_model(config, device)
    base.eval()
    model, stage2_conf = build_finesplat(base, config)
    model = model.to(device)
    if world > 1:
        # e2e unfreezes the whole base; DA3 keeps a few heads that never see the loss, so
        # DDP has to tolerate unused parameters there. Frozen runs keep the fast path.
        model = DistributedDataParallel(
            model, device_ids=[local_rank],
            find_unused_parameters=bool(config.train.get("ddp_find_unused", model.e2e)),
        )
    core = model.module if world > 1 else model

    # e2e needs real train mode: the DA3 grad-checkpoint gate and the single-joint-pass
    # branch both read module.training. Frozen runs keep everything in eval as before
    # (Stage 2's own modules have no dropout/BN, so train mode never mattered there).
    if core.e2e:
        core.train()

    # EMA of the full model (e2e default on): 2Xplat's own training keeps one and ships the
    # EMA weights, and without it the raw unposed e2e run degrades on held-out scenes while
    # its train loss keeps falling. Rank 0 only - DDP keeps the replicas identical, so one
    # copy suffices; ~5.6 GB extra on GPU 0.
    use_ema = bool(config.train.get("ema", core.e2e))
    ema = None
    if use_ema and is_main:
        ema = deepcopy(core)
        ema.requires_grad_(False)
        ema.eval()
    ema_decay = float(config.train.get("ema_decay", 0.999))

    params = [p for p in model.parameters() if p.requires_grad]
    # Resolved before W&B starts so a resumed training continues the same run.
    out_dir = str(train_out_dir(config))
    os.makedirs(out_dir, exist_ok=True)
    resume = config.train.get("resume", False)
    resume_path = None
    if resume:
        resume_path = os.path.join(out_dir, "finesplat_last.pt") if resume is True else str(resume)
        if resume is True and not os.path.exists(resume_path):
            resume_path = os.path.join(out_dir, "refiner_last.pt")  # runs written before the rename
        if not os.path.exists(resume_path):
            resume_path = None
    prev_ckpt = (torch.load(resume_path, map_location="cpu", weights_only=False)
                 if resume_path else None)

    wandb = init_wandb(config, (prev_ckpt or {}).get("wandb_id")) if is_main else None
    wandb_id = wandb.run.id if wandb is not None else None
    if is_main:
        print(f"trainable params: {sum(p.numel() for p in params) / 1e6:.2f}M", flush=True)
    # 2Xplat's own optimizer (src/utils/training_utils.py): AdamW with weight decay applied
    # only to multi-dimensional parameters, so biases and LayerNorm weights are left alone.
    opt, _, _ = create_optimizer(
        core, config.train.get("weight_decay", 0.05), config.train.lr,
        (config.train.get("beta1", 0.9), config.train.get("beta2", 0.95)),
    )
    # train.stage2_lr: a separate peak lr for everything outside the base (core.model),
    # e.g. a from-scratch Stage 2 trained jointly with a fine-tuned Stage 1. Every group
    # follows the same schedule, scaled to its own peak_lr.
    stage2_lr = config.train.get("stage2_lr", None)
    if stage2_lr is not None:
        ref_ids = {id(p) for n, p in core.named_parameters() if not n.startswith("model.")}
        groups = []
        for g in opt.param_groups:
            for is_ref in (False, True):
                ps = [p for p in g["params"] if (id(p) in ref_ids) == is_ref]
                if ps:
                    groups.append({"params": ps, "weight_decay": g["weight_decay"],
                                   "peak_lr": stage2_lr if is_ref else config.train.lr})
        opt = torch.optim.AdamW(groups, lr=config.train.lr, betas=opt.defaults["betas"])
        if is_main:
            n_ref = sum(p.numel() for p in params if id(p) in ref_ids)
            print(f"stage2_lr {stage2_lr} on {n_ref / 1e6:.2f}M params, base lr {config.train.lr} "
                  f"on the rest", flush=True)

    # The eval dataset reads fixed context/target indices from the benchmark json, which
    # only covers the 140 test scenes. Training uses the repo's own loader, which samples
    # views itself (input uniform / target random) and needs no json.
    loader = get_train_data_loader(config, num_workers=config.train.get("num_workers", 4))

    # 2Xplat's own recipe (src/model/loss.py, configs/train_stage*.yaml): 1.0 * MSE +
    # 0.2 * Chen-Koltun VGG19 perceptual, LPIPS off. Stage 2 follows it so that the
    # residual is trained against the same objective the frozen base was. lpips_weight > 0
    # adds LPIPS(VGG) on top; photometric: "l1" restores ZipSplat's original term.
    photometric = config.train.get("photometric", "mse")
    assert photometric in ("mse", "l1"), photometric
    perceptual_weight = config.train.get("perceptual_weight", 0.2)
    lpips_weight = config.train.get("lpips_weight", 0.0)
    # Which views the loss reads, train.loss_views: {target: N, context: M}. target says
    # how many of the batch's target views enter each step (absent = all of them); context
    # says how many context views are sampled in alongside (0 = none, the default). The
    # Stage 2 reads its context rendering error but is scored on held-out views, so nothing
    # in the target-only objective says to close that error - test-time optimisation closes
    # it and gains +1.76 PSNR held-out against Stage 2's +0.75, which says the signal
    # transfers. Both sets are pooled into one photometric + perceptual term, so their
    # relative weight is exactly their view-count ratio.
    # context also takes a list aligned with training.num_views - [3, 3, 3, 1] against
    # buckets [6, 12, 24, 32] - since the affordable quota shrinks as views grow; it is
    # handed to the model as {view_count: quota}.
    loss_views = config.train.get("loss_views", None) or {}
    loss_target_views = loss_views.get("target", None)
    ctx_views = loss_views.get("context", 0) or 0
    if not isinstance(ctx_views, int):
        ctx_views = {int(n): int(c)
                     for n, c in zip(config.training.num_views, ctx_views)}
    core.context_loss_views = ctx_views
    # posed:unposed input mix. With this probability a step selects DA3's cameras
    # (pred_pose_pred_intr) instead of the config's camera_mode - DA3 runs either way, so
    # the flip only changes which cameras _select_cameras returns. Each rank draws its own
    # coin, so a step's effective ratio varies but its expectation is exact. 0 keeps the
    # run fully posed.
    unposed_ratio = float(config.train.get("unposed_ratio", 0.0))
    posed_camera_mode = core.model.geometry_expert.camera_mode
    perceptual_fn = None
    if perceptual_weight > 0:
        perceptual_fn = PerceptualLoss(device=str(device)).to(device).eval()
        perceptual_fn.requires_grad_(False)
    lpips_fn = get_lpips_model(device=str(device)) if lpips_weight > 0 else None
    amp = torch.autocast(device_type="cuda", dtype=AMP_DTYPE[config.train.get("amp_dtype", "bf16")])
    num_steps = config.train.num_steps
    warmup = config.train.get("warmup_steps", 200)
    lr_schedule = config.train.get("lr_schedule", "constant")
    # null/0 disables clipping. clip_grad_norm_ still reports the norm at max_norm=inf,
    # because the scale factor is clamped to 1, so train/grad_norm stays comparable.
    clip = config.train.get("grad_clip", 0.5)
    clip = float("inf") if not clip else clip
    eval_every = config.train.get("ckpt_every", 2500)
    watchlists = build_watchlists(config, os.path.join(out_dir, "watchlist")) if is_main else None
    best_psnr = -float("inf")

    epoch_now, batch_in_epoch = [0], [0]

    def run_watchlist(step: int) -> dict:
        """Evaluate the watchlist, log it, and refresh finesplat_last/finesplat_best."""
        nonlocal best_psnr
        # The pose mix flips the base's camera mode per training step; the watchlist always
        # measures under the config's own mode.
        core.model.geometry_expert.camera_mode = posed_camera_mode
        wl_model = ema if ema is not None else core
        wl_model.model.geometry_expert.camera_mode = posed_camera_mode
        wl = watchlist.evaluate(wl_model, watchlists, device, amp, out_dir, step,
                                save_renders=config.train.get("save_watchlist_renders", True))
        print("  watchlist " + "  ".join(f"{k.split('/')[-1]} {v:.4f}" for k, v in wl.items()),
              flush=True)
        if wandb is not None:
            wandb.log(wl, step=step)
        # Everything needed to continue: the moments matter as much as the weights, and
        # the epoch plus the batch index inside it put the sampler back where it was.
        # e2e checkpoints must carry the (no longer frozen) Stage 1 as well, so they store
        # the full state dict; frozen runs keep the small Stage-2-only dict.
        state = core.state_dict() if core.e2e else stage2_state_dict(core)
        ckpt = {"step": step, "epoch": epoch_now[0], "batch_in_epoch": batch_in_epoch[0],
                "state": state, "optimizer": opt.state_dict(),
                "best_psnr": best_psnr, "wandb_id": wandb_id,
                "watchlist": wl, "stage2_conf": stage2_conf}
        if ema is not None:
            ckpt["ema"] = ema.state_dict()
        torch.save(ckpt, os.path.join(out_dir, "finesplat_last.pt"))
        # finesplat_last.pt is overwritten every time, so also keep one file per evaluation:
        # a run can then be read back at any checkpointed step, not only at its end. These
        # are 170 MB each, so a 10k run at ckpt_every 1000 leaves 1.7 GB behind - prune by
        # hand once the run has been looked at.
        if config.train.get("keep_step_checkpoints", True):
            torch.save(ckpt, os.path.join(out_dir, f"finesplat_step_{step:06d}.pt"))
        # Selected on the train split so no test scene ever steers which weights are kept.
        # Averaged over the logged view counts rather than read off 6v alone.
        train_psnr = sum(v for k, v in wl.items()
                         if k.startswith("watchlist/train") and k.endswith("_psnr"))
        if train_psnr > best_psnr:
            best_psnr = train_psnr
            torch.save(ckpt, os.path.join(out_dir, "finesplat_best.pt"))
            print(f"  new best (watchlist train psnr sum {best_psnr:.4f})", flush=True)
        return wl

    # train.init_stage2_ckpt: warm-start the Stage-2 weights from a finished run (weights
    # only - step, optimizer moments and schedule start fresh). A resume checkpoint in
    # out_dir takes precedence, so an interrupted warm-started run still picks itself up.
    init_ckpt = config.train.get("init_stage2_ckpt", None)
    if init_ckpt and prev_ckpt is None:
        w = torch.load(str(init_ckpt), map_location="cpu", weights_only=False)
        missing = core.load_state_dict(checkpoint_state(w), strict=False).missing_keys
        assert not [k for k in missing if not k.startswith("model.")], missing
        if is_main:
            print(f"initialised Stage-2 weights from {init_ckpt} (step {w.get('step')})", flush=True)

    # resume: true picks up out_dir/finesplat_last.pt; a path resumes from that file.
    start_step, start_epoch, skip_batches = 0, 0, 0
    if prev_ckpt is not None:
        missing = core.load_state_dict(checkpoint_state(prev_ckpt), strict=False).missing_keys
        assert not [k for k in missing if not k.startswith("model.")], missing
        if "optimizer" in prev_ckpt:
            opt.load_state_dict(prev_ckpt["optimizer"])
        if ema is not None and "ema" in prev_ckpt:
            ema.load_state_dict(checkpoint_state(prev_ckpt, use_ema=True))
        start_step = prev_ckpt["step"]
        start_epoch = prev_ckpt.get("epoch", 0)
        skip_batches = prev_ckpt.get("batch_in_epoch", 0)
        best_psnr = prev_ckpt.get("best_psnr", best_psnr)
        if is_main:
            note = "" if "optimizer" in prev_ckpt else "  (no optimizer state - moments restart)"
            print(f"resumed from {resume_path}: step {start_step}, epoch {start_epoch}, "
                  f"batch {skip_batches}{note}", flush=True)
    elif resume and is_main:
        print("resume requested but no checkpoint found; starting from scratch", flush=True)

    step, t0 = start_step, time.time()
    for epoch in range(start_epoch, 1_000_000):
        if step >= num_steps:
            break
        set_epoch(loader, epoch)
        epoch_now[0] = epoch
        for batch_idx, batch in enumerate(loader):
            if step >= num_steps:
                break
            # On the first resumed epoch, replay the sampler up to where it stopped. Only
            # the loader runs; nothing reaches the GPU.
            if epoch == start_epoch and batch_idx < skip_batches:
                continue
            batch_in_epoch[0] = batch_idx
            # Watchlist evaluation and checkpointing run before this step's update, so the
            # numbers at step N reflect exactly N updates. Step 0 is therefore the frozen
            # 2Xplat reference point that every later value is read against - which only
            # holds if nothing has been optimized yet, hence the placement.
            if is_main and step % eval_every == 0:
                run_watchlist(step)

            input_data, target_data = build_input_target_dicts(batch, device)

            for group in opt.param_groups:
                group["lr"] = lr_at(step, group.get("peak_lr", config.train.lr), warmup, num_steps, lr_schedule,
                                    config.train.get("lr_decay_ratio", 0.2))

            unposed = unposed_ratio > 0 and torch.rand(()).item() < unposed_ratio
            core.model.geometry_expert.camera_mode = (
                "pred_pose_pred_intr" if unposed else posed_camera_mode)
            with amp:
                out = model(input_data, target_data)
            gt = target_data["image"].float()
            flat_g = gt.flatten(0, 1)
            # Every Stage-2 step is supervised, not just the last: an auxiliary term per
            # step is what makes each one improve the state it receives. Averaged so the
            # loss scale does not depend on num_steps.
            renders = out["renders"]
            ctx_renders = out.get("context_renders") or []
            # The pool of views the loss reads: a sample of the targets (or all of them)
            # plus this step's sampled context views. The context renders come out of the
            # model, one per step, because only there is the render differentiable.
            t_idx = None
            if loss_target_views and loss_target_views < gt.shape[1]:
                t_idx = torch.randperm(gt.shape[1], device=gt.device)[:loss_target_views]
            gt_pool = gt if t_idx is None else gt[:, t_idx]
            if ctx_renders:
                ctx_gt = input_data["image"][:, out["context_indices"]].float()
                gt_pool = torch.cat([gt_pool, ctx_gt], dim=1)
            flat_gp = gt_pool.flatten(0, 1)
            per_pass, photo_sum, perc_sum, lp_sum = [], None, None, None
            for p, r in enumerate(renders):
                rf = r.float().clamp(0, 1)
                rl = rf if t_idx is None else rf[:, t_idx]
                if ctx_renders:
                    rl = torch.cat([rl, ctx_renders[p].float().clamp(0, 1)], dim=1)
                flat_rl = rl.flatten(0, 1)
                pi = F.mse_loss(rl, gt_pool) if photometric == "mse" else F.l1_loss(rl, gt_pool)
                pci = (perceptual_fn(flat_rl, flat_gp) if perceptual_fn is not None
                       else rl.new_zeros(()))
                lpi = (lpips_fn(flat_gp, flat_rl, normalize=True).mean()
                       if lpips_fn is not None else rl.new_zeros(()))
                per_pass.append((pi, pci, lpi, rf))
                photo_sum = pi if photo_sum is None else photo_sum + pi
                perc_sum = pci if perc_sum is None else perc_sum + pci
                lp_sum = lpi if lp_sum is None else lp_sum + lpi
            n = len(renders)
            # e2e: the unfrozen Stage 1's own prediction (step 0) carries the same objective,
            # so Stage 1 keeps being supervised directly and not only through the Stage-2
            # steps. Targets only - there is no per-step context render for step 0.
            r0 = out.get("render0")
            if r0 is not None:
                r0l = r0.float().clamp(0, 1)
                r0l = r0l if t_idx is None else r0l[:, t_idx]
                gt0 = gt if t_idx is None else gt[:, t_idx]
                pi0 = F.mse_loss(r0l, gt0) if photometric == "mse" else F.l1_loss(r0l, gt0)
                pc0 = (perceptual_fn(r0l.flatten(0, 1), gt0.flatten(0, 1))
                       if perceptual_fn is not None else r0l.new_zeros(()))
                lp0 = (lpips_fn(gt0.flatten(0, 1), r0l.flatten(0, 1), normalize=True).mean()
                       if lpips_fn is not None else r0l.new_zeros(()))
                photo_sum, perc_sum, lp_sum = photo_sum + pi0, perc_sum + pc0, lp_sum + lp0
                n += 1
            photo, perc, lp = photo_sum / n, perc_sum / n, lp_sum / n
            loss = photo + perceptual_weight * perc + lpips_weight * lp
            render = renders[-1].float().clamp(0, 1)
            flat_r = render.flatten(0, 1)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, clip)
            opt.step()
            if ema is not None:
                update_ema(ema, core, decay=ema_decay)

            if is_main and step % config.train.get("log_every", 50) == 0:
                with torch.no_grad():
                    psnr = compute_psnr(flat_g, flat_r).mean().item()
                sec_per_step = (time.time() - t0) / (step + 1)
                pass_psnr = ""
                if len(per_pass) > 1:
                    with torch.no_grad():
                        vals = [compute_psnr(flat_g, rf.flatten(0, 1)).mean().item()
                                for *_, rf in per_pass]
                    pass_psnr = "  pass " + "/".join(f"{v:.2f}" for v in vals)
                print(f"step {step:>6}/{num_steps}  loss {loss.detach().item():.4f}  "
                      f"{photometric} {photo.detach().item():.4f}  perc {float(perc.detach()):.4f}  "
                      f"psnr {psnr:.3f}{pass_psnr}  "
                      f"gnorm {float(grad_norm):.3f}  lr {opt.param_groups[0]['lr']:.2e}  "
                      f"{sec_per_step:.2f}s/step", flush=True)
                if wandb is not None:
                    # train/* are averaged over the Stage-2 steps; pass/* breaks them out
                    # so it is visible whether each step actually improves on the one before.
                    log = {
                        "train/loss": loss.detach().item(),
                        f"train/{photometric}": photo.detach().item(),
                        "train/perceptual": float(perc.detach()),
                        "train/lpips": float(lp.detach()),
                        "train/psnr": psnr, "train/grad_norm": float(grad_norm),
                        "train/lr": opt.param_groups[0]["lr"],
                        "train/error_views": out["error_views"].mean().item(),
                        "train/unposed": float(unposed),
                        "train/sec_per_step": sec_per_step,
                    }
                    if len(per_pass) > 1:
                        with torch.no_grad():
                            for i, (pi, pci, lpi, rf) in enumerate(per_pass, start=1):
                                log[f"pass/{i}_{photometric}"] = pi.detach().item()
                                log[f"pass/{i}_perceptual"] = float(pci.detach())
                                log[f"pass/{i}_psnr"] = compute_psnr(
                                    flat_g, rf.flatten(0, 1)).mean().item()
                    wandb.log(log, step=step)
            if wandb is not None and is_main and step % config.train.get("image_every", 500) == 0:
                log_images(wandb, out, target_data, core, step,
                           max_views=config.train.get("image_views", None))

            step += 1

    if is_main:
        # The in-loop schedule fires on step % eval_every, and the loop exits at
        # num_steps - 1, so the final weights would otherwise never be measured.
        run_watchlist(step)
        print("done", flush=True)
        if wandb is not None:
            wandb.finish()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
