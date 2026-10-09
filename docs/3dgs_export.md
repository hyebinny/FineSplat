# 2Xplat 예측 3DGS 내보내기 & original 3DGS로 이어서 최적화하기

2Xplat이 feed-forward로 예측한 scene별 Gaussian을 표준 3DGS 형식(`point_cloud.ply`)으로
저장하고, 그 위에 reference 구현(graphdeco-inria/gaussian-splatting)의 per-scene 최적화를
이어서 돌리는 방법을 정리한 문서입니다.

전체 흐름:

```
2Xplat inference (save_gaussians: true)
  └─> <scene_dir>/point_cloud.ply + cameras.json        # 이 repo가 저장
        └─> prep.py                                      # COLMAP 씬 + init 체크포인트 생성
              └─> gaussian-splatting train.py            # 예측 GS에서 이어서 최적화
```

한 DL3DV test scene으로 end-to-end 검증된 상태입니다:
2Xplat init 24.70 PSNR(held-out 8뷰) → original 3DGS 1,000 iter 후 26.87 (+2.17),
densification 75,264 → 95,648 GS.

---

## 1. GS 저장하기

`src/configs/3Xplat_init_posed_only/` 아래 config들은 전부 `inference.save_gaussians: true`가
기본입니다. 그대로 돌리면 됩니다:

```bash
cd /scratch2/binny56/projects/ICLR2027/2Xplat_base
CUDA_VISIBLE_DEVICES=0 python -m src.commands.inference \
  --config src/configs/3Xplat_init_posed_only/dl3dv_224_6v.yaml
```

scene마다 결과 디렉터리에 두 파일이 생깁니다:

```
experiments/inference/<ckpt명>/gt_pose_gt_intr/i6t8/000001_<scene_hash>/
├── point_cloud.ply    # 표준 3DGS 레이아웃의 Gaussian들
└── cameras.json       # 이 Gaussian들이 사는 좌표계의 카메라 (아래 참고)
```

용량 감각: 224 해상도 6뷰 기준 scene당 75,264 GS ≈ 7.7 MB. 뷰 수에 비례해 늘어납니다
(24뷰 = 301,056 GS ≈ 31 MB). GS가 필요 없는 재측정이면 CLI에서
`inference.save_gaussians=false`로 끄면 됩니다.

관련 코드는 `src/utils/ply_io.py` 하나에 다 있습니다
(`gaussians_to_ply` / `gaussians_from_ply` / `save_scene_cameras`).

## 2. 반드시 알아야 하는 두 가지 규약

**(a) 좌표계 — cameras.json 없이는 PLY가 무용지물입니다.**
2Xplat의 geometry expert는 모든 pose를 scene scale로 정규화한 뒤 Gaussian을 디코드합니다.
즉 PLY 안의 Gaussian들은 데이터셋의 원본 카메라(`opencv_cameras.json`)가 아니라
**정규화된 카메라 좌표계**에 배치됩니다. `cameras.json`이 바로 그 좌표계의 context/target
카메라(c2w, fxfycxcy, 원본 frame index)를 담고 있고, 이 GS 위에 무언가를 최적화하려면
이미지를 반드시 이 카메라들과 짝지어야 합니다. 원본 pose를 쓰면 투영이 어긋납니다.

**(b) opacity — SH 계수가 아니라 logit으로 저장됩니다.**
2Xplat 내부에서는 opacity도 SH 함수라서 실효 logit이 `0.28209(C0) × coeff`입니다.
표준 3DGS는 저장값을 그대로 logit으로 읽으므로, export가 C0를 곱해서 쓰고 import가
도로 나눕니다. 이 변환이 없으면 INRIA 쪽에서 opacity가 logit 기준 3.55배 과장되어
렌더가 무너집니다(실측 13 dB). `ply_io.py`가 알아서 처리하므로 **PLY를 직접 파싱하지 말고
`gaussians_from_ply`를 쓰거나, 표준 3DGS 도구로만 읽으세요.**

그 외 레이아웃은 reference와 동일합니다: `scale`은 log 공간, `rot_0..3`은 (w,x,y,z),
f_rest는 channel-major, sh_degree 1 (f_rest 9개), normals는 0.

## 3. original 3DGS로 이어서 최적화하기

INRIA train.py는 (1) COLMAP 형식 씬 디렉터리와 (2) 시작 상태가 필요합니다. 둘 다
`/home/binny56/projects/gaussian-splatting/3Xplat_sample/prep.py`가 만들어줍니다:

```bash
cd /home/binny56/projects/gaussian-splatting/3Xplat_sample
CUDA_VISIBLE_DEVICES=0 python prep.py \
  <위 1번의 scene 디렉터리>  \
  /scratch/y5ryan/datasets/dl3dv_480/test/<scene_hash>/opencv_cameras.json
```

prep.py가 하는 일:
- `scene/images/` — 2Xplat 로더로 그 scene의 224×224 context crop을 다시 만들어 저장
  (cameras.json의 intrinsic과 정확히 일치하도록). 저장된 pose와 로더 pose가 일치하는지
  assert로 검증합니다.
- `scene/sparse/0/{cameras.txt, images.txt, points3D.txt}` — cameras.json을 COLMAP text
  형식으로 변환. points3D는 GS 중심+색 (stock 초기화 fallback용).
- `chkpnt_init.pth` — PLY의 **전체 상태**(위치·SH·scale·rotation·opacity)를 INRIA의
  `GaussianModel.load_ply`로 읽어 `--start_checkpoint` 형식으로 포장. INRIA 코드는 한 줄도
  수정하지 않습니다.

그다음 학습:

```bash
cd /home/binny56/projects/gaussian-splatting
CUDA_VISIBLE_DEVICES=0 python train.py \
  -s 3Xplat_sample/scene -m 3Xplat_sample/model \
  --sh_degree 1 --white_background \
  --start_checkpoint 3Xplat_sample/chkpnt_init.pth \
  --iterations 1000 --save_iterations 1000
```

플래그 셋 다 필수입니다:
- `--sh_degree 1` : 우리 PLY의 f_rest 개수(9)와 맞아야 `load_ply`가 통과합니다.
- `--white_background` : 2Xplat 렌더러가 흰 배경입니다. 검은 배경으로 돌리면 시작점부터
  4 dB가량 어긋납니다.
- `--start_checkpoint` : 이게 없으면 points3D에서 cold start (위치+색만, scale/rot/opacity
  재초기화)라 예측 GS를 잇는 게 아니게 됩니다.

정상 시작의 표식: `[ITER 1] Evaluating train: PSNR ~27` (2Xplat 수준에서 출발).
13 근처에서 시작하면 opacity 규약이나 배경색이 틀린 것입니다.

## 4. 환경

- 이 repo의 `2xplat` conda env를 그대로 씁니다. INRIA의 CUDA 확장 3종
  (`diff_gaussian_rasterization`, `simple_knn`, `fused_ssim`)이 이미 설치돼 있습니다.
  재설치가 필요하면:
  ```bash
  cd /home/binny56/projects/gaussian-splatting
  CUDA_HOME=/usr/local/cuda-13.0 pip install --no-deps --no-build-isolation \
    ./submodules/diff-gaussian-rasterization ./submodules/simple-knn ./submodules/fused-ssim
  ```
  `--no-deps`를 꼭 붙이세요. 이 env는 numpy 1.26.4에 고정돼 있고, 빼먹으면 pip이 numpy를
  2.x로 올려버립니다.

## 5. 검증 스크립트

- round-trip: 저장 → 재로드 → 재렌더가 원본과 일치하는지 (파라미터 max|Δ| ≈ 0, 렌더 동일).
  `src/utils/ply_io.py`의 함수들로 수행하며, 참고용 스크립트가
  `3Xplat_sample/check_render.py`에 있습니다 (INRIA rasterizer로 context 뷰 PSNR 출력 —
  26~28 나오면 정상).
- held-out 평가: target 뷰는 cameras.json의 `target` 블록으로 렌더해서 GT와 비교하면
  됩니다.
