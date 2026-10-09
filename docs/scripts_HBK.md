# 실행 스크립트 정리

`src/utils/eval_paths.py`에서 eval 결과들이 저장될 경로를 정함

## FineSplat eval

데이터셋 / view 수 / 카메라 설정마다 config가 있음 (예: `dl3dv_6v_posed`, `dl3dv_12v_predgt`, `re10k_6v_unposed`, `scannetpp_128v_unposed`)  
`posed`: GT pose + GT intrinsics  
`predgt`: DA3 pose + GT intrinsics  
`unposed`: DA3 pose + DA3 intrinsics  
`predgt`, `unposed`는 metric 계산 전 EPA(MSE loss, 200 step) 적용함

```bash
# iteration 0(Stage 1 Gaussian)과 iteration 4(FineSplat), pruning 없음
CUDA_VISIBLE_DEVICES=0 python -m src.commands.eval_passes src/configs/finesplat/eval/dl3dv_6v_posed.yaml

# iteration 4를 ZipSplat 개수(view당 10,368개)로 pruning (contribution score)
CUDA_VISIBLE_DEVICES=0 python -m src.commands.eval_prune  src/configs/finesplat/eval/dl3dv_6v_posed.yaml
```

결과 위치: `../FineSplat_exp_final/eval/224res/lanczos/<dataset>/<N>v/<camera_mode>/finesplat_224/` 아래 `iter0/`, `iter4/`(eval_passes), `iter4_prune/`(eval_prune).

자주 쓰는 `inference.*` 설정 (yaml에 적거나, 명령 뒤에 `inference.key=value`로 덮어써요):

| key | 효과 |
|---|---|
| `scene_range: [lo, hi]` | `[lo, hi)` test scene을 구간별로 나눠서 돌림 (GPU 여러 장에 분산해서 eval). 나눠 돌린 뒤에는 summary를 한 번 더 만들어서 종합함 |
| `eval_iters: [0, 4]` | metric 계산할 Stage 2 iteration. EPA는 매 iteration마다 한 번씩 독립적으로 적용됨 |
| `epa_steps: 200` | EPA step 수. default: 200 |
| `pose_optimization: false` | EPA 없이 DA3가 예측한 target pose 그대로 렌더링 |
| `vis_save: {root, subdir, tags: {4: name}}` | target 렌더 PNG 저장 여부 |
| `ply_save: {root, tags: {4: name}}` | scene마다 Gaussian을 `point_cloud.ply` + `cameras.json`으로 저장 |
| `prune_keep_per_view: [10368]` | eval_prune에서 context view 한 장당 남길 개수(10368=ZipSplat 설정) |

예: GPU 2장에 나눠 eval (24 view)

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.commands.eval_passes src/configs/finesplat/eval/dl3dv_24v_unposed.yaml inference.scene_range=[0,70] &
CUDA_VISIBLE_DEVICES=1 python -m src.commands.eval_passes src/configs/finesplat/eval/dl3dv_24v_unposed.yaml inference.scene_range=[70,140] &
```

## Stage 1 eval

e2e 모델의 Stage 1 eval (FineSplat checkpoint의 iteration 0만 점수 냄; 결과는 `finesplat_224/iter0/`):

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.commands.eval_passes src/configs/finesplat/eval/dl3dv_6v_posed.yaml \
  inference.eval_iters=[0]
```

e2e가 아닌 4x4->4GS으로 학습된 2Xplat에 대해 eval (refiner checkpoint를 비우면 base는 `inference.ckpt_path`의
2Xplat 가중치 그대로이고, refiner head는 zero-init이라 iteration 0이 곧 2Xplat 출력; 결과는 `2xplat_224/iter0/`):

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.commands.eval_passes src/configs/finesplat/eval/dl3dv_6v_posed.yaml \
  inference.refiner.ckpt_path=null inference.eval_iters=[0] inference.run=2xplat_224
```

## 학습

```bash
# Stage 2: Stage 1 고정, GT 카메라, 20k iteration, 4 GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 -m src.commands.train_refiner \
  --config src/configs/finesplat/train/stage2_224.yaml

# end-to-end fine-tuning: unposed, 10k iteration, 8 GPU. 위 run의 refiner에서 시작
torchrun --nproc_per_node 8 -m src.commands.train_refiner \
  --config src/configs/finesplat/train/e2e_224.yaml
```

## TTO (Stage 1 token의 test-time optimization)

checkpoint는 eval_passes와 같이 불러오지만 refiner 출력은 쓰지 않고, Stage 1의 fused token을 context view에 맞춰 최적화  
`inference.tto.steps`에서는 한 번의 실행에서 여러 step TTO의 eval을 할 수 있음

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.commands.eval_tto src/configs/finesplat/eval/dl3dv_6v_posed.yaml \
  inference.tto.lrs=[0.4] inference.tto.steps=[1,5,10,50]
# -> .../finesplat_224/iter0_tto0.4_s<k>/
```
