# Configs

Everything the paper's FineSplat-224 numbers come from, one directory per dataset:

```
src/configs/
  DL3DV/
    train/stage2.yaml          Stage 2 on the frozen Stage 1 (GT cameras), 20k iterations, 4 GPUs
    train/e2e.yaml             end-to-end fine-tuning, pose-free, 10k iterations, 8 GPUs
    eval/{6v,12v,24v}/posed.yaml     GT poses + intrinsics
    eval/{6v,12v,24v}/unposed.yaml   DA3 poses + intrinsics, EPA on the targets
  RE10K/eval/6v/{posed,unposed}.yaml
  ScanNetpp/eval/{32v,64v,128v}/unposed.yaml
```

Paths are relative to the repo root. Outputs go next to the repo (`../FineSplat_exp_final/`,
`../FineSplat_e2e_ckpt/`; see `src/utils/eval_paths.py`).

## Weights

```
pretrained_weights/
  finesplat_224res.pt                                FineSplat-224 (Stage 1 + Stage 2, end-to-end); the only file evaluation needs
  da3_giant.safetensors                              DA3-Giant                  } Stage-1 experts, training only;
  mvp_patchwise_offset_dl3dv_lr_novdo.pt             MVP appearance expert      } from the DA3 and 2Xplat
  2Xplat/2xplat_patchwise_offset_dl3dv_lr_novdo.pt   Stage 1 (2Xplat)           } releases
```

## Data

`data.root` / `data.split` / `data.scene_list`: scene `i` of the list lives at
`<root>/<split>/<id>/opencv_cameras.json` (DL3DV). `data.json_path` fixes the context / target
frames of every benchmark scene (`src/data/<dataset>/eval_ctx<N>_tgt<M>.json`). RE10K and
ScanNet++ keep their own loaders (`inference.dataset_name`) and point `data.data_path` at a
local copy.

## Training

```
torchrun --nproc_per_node 4 -m src.commands.train_finesplat --config src/configs/DL3DV/train/stage2.yaml
torchrun --nproc_per_node 8 -m src.commands.train_finesplat --config src/configs/DL3DV/train/e2e.yaml
```

`e2e.yaml` starts from the Stage-2 run above (`train.init_stage2_ckpt`).

## Evaluation

```
# step 0 (Stage 1) and step 4 (FineSplat), all Gaussians
python -m src.commands.eval_passes src/configs/DL3DV/eval/6v/posed.yaml

# step 4 pruned to ZipSplat's budget (10,368 Gaussians per view) by contribution score
python -m src.commands.eval_prune  src/configs/DL3DV/eval/6v/posed.yaml
```

Results land in `../FineSplat_exp_final/eval/224res/lanczos/<dataset>/<N>v/<camera_mode>/finesplat_224/{step0,step4,step4_prune}/`,
one `metrics.json` per scene plus `summary.csv`. `inference.scene_range=[lo,hi]` on the command
line shards a run across GPUs; `inference.vis_save` / `inference.ply_save` write renders / Gaussians.
