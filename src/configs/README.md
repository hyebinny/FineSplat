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
  finesplat_224res.pt      FineSplat-224 (Stage 1 + Stage 2, end-to-end); the only file evaluation needs
  train/stage1_224res.pt   Stage 1 (2Xplat, with its DA3 and MVP weights); starting point of train/stage2.yaml
  train/stage2_224res.pt   Stage 2 trained on the frozen Stage 1 (20k); starting point of train/e2e.yaml
```

## Data

`data.root` / `data.split` / `data.scene_list`: scene `i` of the list lives at
`<root>/<split>/<id>/opencv_cameras.json` (DL3DV). `data.json_path` fixes the context / target
frames of every benchmark scene (`src/data/<dataset>/eval_ctx<N>_tgt<M>.json`). RE10K and
ScanNet++ keep their own loaders (`inference.dataset_name`) and point `data.data_path` at a
local copy. All configs expect the data under `datasets/` in the repository root
(`datasets/dl3dv_480`, `datasets/scannetpp/test`; symlinks are fine, the directory is gitignored).

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
