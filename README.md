<div align="center">
<h1>FineSplat: Fine Patches, Local Groups, Detailed Splats</h1>

[Hyebin Kim](https://github.com/hyebinny)\*, [Hwasik Jeong](https://github.com/HwasikJeong)\*, [Joonhyuk Park](https://github.com/joonh-park), [Gyeongjin Kang](https://gynjn.github.io/info/), [Jungwoo Kim](https://github.com/jungcow), [Younghyun Kim](https://github.com/yhyun225), [Eunbyung Park](https://silverbottlep.github.io/)

Yonsei University

\*Equal contribution
</div>

<div align="center" style="margin-top: 2em; margin-bottom: 1.0em">
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/FineSplat-ArXiv-red" alt="arXiv"></a> <a href="https://hyebinny.github.io/FineSplat/"><img src="https://img.shields.io/badge/FineSplat-Project_Page-blue" alt="Project Page"></a> <a href="https://huggingface.co/hyebinny/FineSplat"><img src="https://img.shields.io/badge/FineSplat-Hugging_Face-yellow" alt="Hugging Face"></a>
</div>

<div align="center" style="margin-top: 5.0em; margin-bottom: 1.0em">
  <img src="./assets/teaser.png" alt="FineSplat" />
</div>

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/hyebinny/FineSplat.git
cd FineSplat

# 2. Create and activate conda environment
conda create -n finesplat python=3.12 -y
conda activate finesplat

# 3. Install dependencies (adjust PyTorch and CUDA version to match your system)
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt

# 4. gsplat: intrinsic gradient version
pip install git+https://github.com/QitaoZhao/gsplat.git --no-build-isolation
```

## Checkpoints

We release one checkpoint, FineSplat-224, which holds the whole model (Stage 1 and Stage 2,
fine-tuned end-to-end). Download it from [🤗 Hugging Face](https://huggingface.co/hyebinny/FineSplat)
and place it at:

```
path/to/FineSplat/pretrained_weights/finesplat_224res.pt
```

Nothing else is needed for evaluation: the DA3, MVP and 2Xplat weights are part of this file, so the
evaluation configs leave `model.da_model_weights_path`, `model.mvp_weights_path` and
`inference.ckpt_path` empty.

Training Stage 2 from scratch starts from the Stage-1 experts instead; see [Training](#training).

## Datasets

FineSplat uses the same data as [2Xplat](https://github.com/HwasikJeong/2Xplat). Download DL3DV from
the [official repository](https://huggingface.co/DL3DV); we use **DL3DV-ALL-480P** at 224x224. Each
scene is a folder `<split>/<scene id>/` holding the images and an `opencv_cameras.json`.

Point the configs at your copy with three keys under `data`:

```yaml
data:
  root: /path/to/dl3dv_480          # dataset root
  split: test                       # or train
  scene_list: src/data/DL3DV/test_scenes.txt   # scene ids, one per line
```

Scene `i` of the list is read from `<root>/<split>/<id>/opencv_cameras.json`. The benchmark's
context / target frames per scene are fixed by `data.json_path`
(`src/data/DL3DV/eval_ctx{6,12,24}_tgt8.json`, from YoNoSplat). `src/data/DL3DV/train_scenes.txt`
lists the 10,350 training scenes.

RE10K and ScanNet++ (cross-dataset evaluation) have their own loaders, selected by
`inference.dataset_name`, and take `data.data_path` (a text file of per-scene `scene_meta.json` paths
for RE10K; the directory of `.torch` chunks for ScanNet++). Their context / target indices are in
`src/data/RE10K/` and `src/data/ScanNetpp/`.

## Evaluation

`src/configs/<dataset>/eval/<N>v/` has one config per camera setting:
`DL3DV/eval/{6v,12v,24v}/{posed,unposed}.yaml`, `RE10K/eval/6v/{posed,unposed}.yaml`,
`ScanNetpp/eval/{32v,64v,128v}/unposed.yaml`.

| Setting | Context cameras | Target cameras |
| ------- | --------------- | -------------- |
| `posed` | GT poses and intrinsics | GT |
| `unposed` | DA3 poses and intrinsics | DA3, then EPA |

EPA (evaluation-time pose alignment) optimizes the target poses for 200 Adam steps against the
rendering before scoring, as in 2Xplat and YoNoSplat.

```bash
# step 0 (Stage 1) and step 4 (FineSplat), all Gaussians
CUDA_VISIBLE_DEVICES=0 python -m src.commands.eval_passes src/configs/DL3DV/eval/6v/posed.yaml

# step 4 pruned to ZipSplat's budget (10,368 Gaussians per view) by contribution score
CUDA_VISIBLE_DEVICES=0 python -m src.commands.eval_prune  src/configs/DL3DV/eval/6v/posed.yaml
```

Results are written next to the repository, under
`../FineSplat_exp_final/eval/224res/lanczos/<dataset>/<N>v/<camera mode>/finesplat_224/` as
`step0/`, `step4/` (eval_passes) and `step4_prune/` (eval_prune): one `metrics.json` per scene plus
`summary.csv` and `average_metrics.txt`. A second positional argument sets the output directory.

Useful `inference.*` options (in the yaml, or appended to the command as `inference.key=value`):

| Option | Description |
| ------ | ----------- |
| `checkpoint` | `{ckpt_path, use_ema, full_state}` of the FineSplat checkpoint. `ckpt_path: null` evaluates Stage 1 alone. |
| `eval_steps` | Which Stage-2 steps to score, e.g. `[0, 4]`. EPA runs once per scored step. |
| `epa_steps` | EPA length (200). |
| `pose_optimization` | `false` renders at DA3's target poses without EPA. |
| `scene_range` | `[lo, hi]` scores scenes `lo..hi-1` only, to split a run across GPUs. |
| `prune_keep_per_view` | Gaussians kept per context view by `eval_prune`. |
| `vis_save` / `ply_save` | Also write the target renders as PNGs / the Gaussians as `point_cloud.ply` + `cameras.json`. |

## Training

Stage 2 is first trained on a frozen Stage 1 with GT cameras, then the whole model is fine-tuned
end-to-end in the pose-free setting. Both read `data.root` / `data.scene_list` as above and log a
fixed watchlist of scenes to W&B (`train.wandb`, API key in `src/configs/api_keys.yaml`).

Training needs the Stage-1 experts, which we do not redistribute: the geometry expert
[DA3-Giant](https://huggingface.co/depth-anything/DA3-GIANT-1.1/resolve/main/model.safetensors)
and the patch-aligned appearance expert / 2Xplat checkpoint from
[2Xplat](https://github.com/HwasikJeong/2Xplat). The training configs expect them at
`pretrained_weights/da3_giant.safetensors`, `pretrained_weights/mvp_patchwise_offset_dl3dv_lr_novdo.pt`
and `pretrained_weights/2Xplat/2xplat_patchwise_offset_dl3dv_lr_novdo.pt`.

```bash
# Stage 2 on the frozen Stage 1: 20k iterations, 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 -m src.commands.train_finesplat \
  --config src/configs/DL3DV/train/stage2.yaml

# end-to-end fine-tuning: 10k iterations, 8 GPUs, initialised from the run above
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 -m src.commands.train_finesplat \
  --config src/configs/DL3DV/train/e2e.yaml
```

Checkpoints go to `../FineSplat_exp_final/train/<run>/` (Stage 2) and `../FineSplat_e2e_ckpt/<run>/`
(end-to-end) as `finesplat_last.pt`, `finesplat_best.pt` and `finesplat_step_<N>.pt`. Stage 1 itself is
trained with [2Xplat](https://github.com/HwasikJeong/2Xplat)'s own recipe.

## Related Projects

This project builds on many excellent research works. We thank the authors for sharing their code.

- [2Xplat](https://github.com/HwasikJeong/2Xplat)
- [MVP](https://github.com/Gynjn/MVP)
- [Point Transformer V3](https://github.com/Pointcept/PointTransformerV3)
- [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)

## Citation

If you find our work useful, please cite:

```bibtex
@article{kim2026finesplat,
          title={FineSplat: Fine Patches, Local Groups, Detailed Splats},
          author={Hyebin Kim and Hwasik Jeong and Joonhyuk Park and Gyeongjin Kang and Jungwoo Kim and Younghyun Kim and Eunbyung Park},
          journal={arXiv preprint arXiv:XXXX.XXXXX},
          year={2026}
}
```
