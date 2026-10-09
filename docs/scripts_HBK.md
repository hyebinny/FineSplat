# 실행 스크립트 정리

`src/utils/eval_paths.py`에서 eval 결과들이 저장될 경로를 정함 (repo 옆 `../FineSplat_exp_final/`)

## 용어

- **Stage 1**: 2Xplat (DA3 + MVP). 4x4 patch마다 token 하나, token마다 Gaussian 4개
- **Stage 2**: `src/model/stage2.py`. Stage 1 token을 `num_steps`(=4) step 동안 고치는 모듈. 결과 폴더의 `step0`이 Stage 1 출력, `step4`가 FineSplat 출력
- **FineSplat checkpoint**: `pretrained_weights/finesplat/e2e_224_step10k_ema.pt` (Stage 1 + Stage 2, end-to-end, EMA)

## 데이터

DL3DV는 `data.root` + `data.split` + `data.scene_list`로 지정함. scene id 목록(`src/data/DL3DV/{train,test}_scenes.txt`)의 각 id가
`<root>/<split>/<id>/opencv_cameras.json`에 대응됨. context/target frame은 `data.json_path`(`src/data/DL3DV/eval_ctx{6,12,24}_tgt8.json`)가 정함.
RE10K, ScanNet++는 loader가 따로 있고(`inference.dataset_name`) `data.data_path`로 로컬 사본을 가리킴 (RE10K 목록 파일은 git 추적 안 함).
video용 frame 목록은 `src/data/video/DL3DV/ctx{6,12,24}.json`.

## FineSplat eval

데이터셋 / view 수 / 카메라 설정마다 config가 있음 (예: `dl3dv_6v_posed`, `dl3dv_12v_predgt`, `re10k_6v_unposed`, `scannetpp_128v_unposed`)  
`posed`: GT pose + GT intrinsics  
`predgt`: DA3 pose + GT intrinsics  
`unposed`: DA3 pose + DA3 intrinsics  
`predgt`, `unposed`는 metric 계산 전 EPA(MSE loss, 200 step) 적용함

```bash
# step 0(Stage 1 Gaussian)과 step 4(FineSplat), pruning 없음
CUDA_VISIBLE_DEVICES=0 python -m src.commands.eval_passes src/configs/finesplat/eval/dl3dv_6v_posed.yaml

# step 4를 ZipSplat 개수(view당 10,368개)로 pruning (contribution score)
CUDA_VISIBLE_DEVICES=0 python -m src.commands.eval_prune  src/configs/finesplat/eval/dl3dv_6v_posed.yaml
```

결과 위치: `../FineSplat_exp_final/eval/224res/lanczos/<dataset>/<N>v/<camera_mode>/finesplat_224/` 아래 `step0/`, `step4/`(eval_passes), `step4_prune/`(eval_prune).

자주 쓰는 `inference.*` 설정 (yaml에 적거나, 명령 뒤에 `inference.key=value`로 덮어써요):

| key | 효과 |
|---|---|
| `scene_range: [lo, hi]` | `[lo, hi)` test scene을 구간별로 나눠서 돌림 (GPU 여러 장에 분산해서 eval). 나눠 돌린 뒤에는 summary를 한 번 더 만들어서 종합함 |
| `eval_steps: [0, 4]` | metric 계산할 Stage 2 step. EPA는 step마다 한 번씩 독립적으로 적용됨 |
| `epa_steps: 200` | EPA step 수. default: 200 |
| `pose_optimization: false` | EPA 없이 DA3가 예측한 target pose 그대로 렌더링 |
| `vis_save: {root, subdir, tags: {4: name}}` | target 렌더 PNG 저장 여부 |
| `ply_save: {root, tags: {4: name}}` | scene마다 Gaussian을 `point_cloud.ply` + `cameras.json`으로 저장 |
| `prune_keep_per_view: [10368]` | eval_prune에서 context view 한 장당 남길 개수(10368=ZipSplat 설정, `step4_prune/`에 저장; 다른 값은 `step4_contrib<k>pv/`) |
| `checkpoint: {ckpt_path, use_ema, full_state}` | 불러올 FineSplat checkpoint. end-to-end checkpoint는 `use_ema: true`, `full_state: true` |

예: GPU 2장에 나눠 eval (24 view)

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.commands.eval_passes src/configs/finesplat/eval/dl3dv_24v_unposed.yaml inference.scene_range=[0,70] &
CUDA_VISIBLE_DEVICES=1 python -m src.commands.eval_passes src/configs/finesplat/eval/dl3dv_24v_unposed.yaml inference.scene_range=[70,140] &
```

## Stage 1 eval

e2e 모델의 Stage 1 eval (FineSplat checkpoint의 step 0만 점수 냄; 결과는 `finesplat_224/step0/`):

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.commands.eval_passes src/configs/finesplat/eval/dl3dv_6v_posed.yaml \
  inference.eval_steps=[0]
```

e2e가 아닌 4x4->4GS으로 학습된 2Xplat에 대해 eval (checkpoint를 비우면 Stage 1은 `inference.ckpt_path`의
2Xplat 가중치 그대로이고, Stage 2 head는 zero-init이라 step 0이 곧 2Xplat 출력; 결과는 `2xplat_224/step0/`):

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.commands.eval_passes src/configs/finesplat/eval/dl3dv_6v_posed.yaml \
  inference.checkpoint.ckpt_path=null inference.eval_steps=[0] inference.run=2xplat_224
```

## 학습

```bash
# Stage 2: Stage 1 고정, GT 카메라, 20k iteration, 4 GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 -m src.commands.train_finesplat \
  --config src/configs/finesplat/train/stage2_224.yaml

# end-to-end fine-tuning: unposed, 10k iteration, 8 GPU. 위 run의 Stage 2에서 시작 (train.init_stage2_ckpt)
torchrun --nproc_per_node 8 -m src.commands.train_finesplat \
  --config src/configs/finesplat/train/e2e_224.yaml
```

checkpoint는 `finesplat_last.pt` / `finesplat_best.pt` / `finesplat_step_<N>.pt`로 저장됨 (`train.resume: true`면 `finesplat_last.pt`에서 이어서 학습).
end-to-end checkpoint에는 optimizer 상태까지 들어 있으니 공유할 때는 EMA만 뽑아서:

```python
c = torch.load("finesplat_last.pt", map_location="cpu", mmap=True, weights_only=False)
torch.save({"ema": c["ema"], "step": c["step"], "stage2_conf": c["stage2_conf"]}, "ema_only.pt")
```

## TTO (Stage 1 token의 test-time optimization)

checkpoint는 eval_passes와 같이 불러오지만 Stage 2 출력은 쓰지 않고, Stage 1의 fused token을 context view에 맞춰 최적화  
`inference.tto.steps`에서는 한 번의 실행에서 여러 step TTO의 eval을 할 수 있음

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.commands.eval_tto src/configs/finesplat/eval/dl3dv_6v_posed.yaml \
  inference.tto.lrs=[0.4] inference.tto.steps=[1,5,10,50]
# -> .../finesplat_224/step0_tto0.4_s<k>/
```
