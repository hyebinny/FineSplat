# FineSplat-224 configs

Everything the paper's FineSplat-224 numbers come from. Paths are relative to the repo root;
checkpoints go under `pretrained_weights/` and outputs next to the repo in `../FineSplat_exp_final/`
(see `src/utils/eval_paths.py`).

## Weights

```
pretrained_weights/
  finesplat_224res.pt                                FineSplat-224 (Stage 1 + Stage 2, end-to-end; the only file evaluation needs)
  da3_giant.safetensors                              DA3-Giant                  } Stage-1 experts, needed for
  mvp_patchwise_offset_dl3dv_lr_novdo.pt             MVP appearance expert      } training only; from the DA3
  2Xplat/2xplat_patchwise_offset_dl3dv_lr_novdo.pt   Stage 1 (2Xplat)           } and 2Xplat releases
```

## Data

`data.root` / `data.split` / `data.scene_list`: scene `i` of the list lives at
`<root>/<split>/<id>/opencv_cameras.json` (DL3DV). `data.json_path` fixes the context / target
frames of every benchmark scene (`src/data/<dataset>/eval_ctx<N>_tgt<M>.json`). RE10K and
ScanNet++ keep their own loaders (`inference.dataset_name`) and point `data.data_path` at a
local copy.

## Training

```
# Stage 2 on the frozen Stage 1 (GT cameras), 20k iterations on 4 GPUs
torchrun --nproc_per_node 4 -m src.commands.train_finesplat --config src/configs/finesplat/train/stage2_224.yaml

# end-to-end fine-tuning, unposed, 10k iterations on 8 GPUs (initialised from the run above)
torchrun --nproc_per_node 8 -m src.commands.train_finesplat --config src/configs/finesplat/train/e2e_224.yaml
```

## Evaluation

One config per dataset / view count / camera setting. `posed` = GT poses and intrinsics,
`predgt` = predicted poses with GT intrinsics, `unposed` = both predicted. Pose-free settings
run EPA (200 Adam steps on the target poses) before scoring.

```
# step 0 (Stage 1) and step 4 (FineSplat), all Gaussians
python -m src.commands.eval_passes src/configs/finesplat/eval/dl3dv_6v_posed.yaml

# step 4 pruned to ZipSplat's budget (10,368 Gaussians per view) by contribution score
python -m src.commands.eval_prune  src/configs/finesplat/eval/dl3dv_6v_posed.yaml
```

Results land in `../FineSplat_exp_final/eval/224res/lanczos/<dataset>/<N>v/<camera_mode>/finesplat_224/{step0,step4,step4_prune}/`,
one `metrics.json` per scene plus `summary.csv`. `inference.scene_range=[lo,hi]` on the command
line shards a run across GPUs; `inference.vis_save` / `inference.ply_save` write renders / Gaussians.
