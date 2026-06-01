# P2: FFHQ-1024 GAN — Claude Guidelines

p2 작업 시 root `CLAUDE.md`에 **추가로** 로드. 세부 사양은 `docs/superpowers/specs/2026-06-01-p2-ffhq-gan-design.md`, 구현 plan은 `docs/superpowers/plans/2026-06-01-p2-d256-stylegan2.md` 참조.

## 핵심 제약 (project02.pdf)

- 마감: **2026-06-09 23:59**, 이메일 제출 (khshim@skku.edu)
- Generator hard threshold **≤ 40M params** (현 D-256 모델: 30.04M)
- z: 512-dim, output 3×1024×1024 (bilinear wrapper로 256→1024)
- valid/test 학습 금지, 외부 데이터 금지, **pretrained 금지**
- 허용 lib: PyTorch, TorchVision, OpenCV/PIL/skimage/matplotlib, WandB, **pytorch-fid**
- 금지: HuggingFace, Lightning, Accelerate, Albumentations

## 디렉토리 위치

### 데이터 (모두 gitignore)
- 학습: `p2/data/train_50k_256.zip` — `train.py`가 읽는 경로 (config의 `train_zip`)
- FID real-stats: `p2/data/valid_10k_256.zip` (extract → `p2/data/valid_10k_256_dir/`)
- 데스크탑: 원본 zip이 `p2/` 루트에 있으면 `p2/data/`에 symlink로 연결
- Colab: Drive `MyDrive/p2-data/`에서 `p2/data/`로 cp (노트북 셀 5번이 자동)

### 학습 산출물 (모두 gitignore)
- ckpt: `p2/runs/d256_main/ckpt_XXXXXXXXX.pt` (200k images마다 비동기 저장)
- 최종: `p2/runs/d256_main/final.pt` (학습 종료 시)
- samples: `p2/runs/d256_main/samples/grid_*.png`
- WandB: `p2/wandb/` (local cache)

### 제출 산출물
- `p2/checkpoints/model.pth` — Generator state (EMA) — 학습 후 final.pt에서 추출
- `p2/checkpoints/model.onnx` — 리더보드 제출용 (export_onnx.py)
- `p2/checkpoints/fid_stats_256.npz` — FID real-side 통계 캐시 (valid에서 1회 빌드)

### 제출 zip 구조 (PDF spec)
```
2020314315_project02.zip
├── src/
├── checkpoints/model.pth
├── 2020314315_project02_report.pdf  (6 pages, 11pt)
├── pyproject.toml
└── README.md
```
package_submission.py가 자동 빌드 (plan task 18).

## 학습 환경

- **Colab Pro+ A100** (가용 시) — 컴퓨트 유닛 952 → ~79h
- bf16 mixed precision (StyleGAN2-ADA 관례)
- 14M images 목표, A100에서 ~22h
- ckpt every 200k images + Drive backup (resumable)

## ABORT GATE (학습 실패 판단)

- 1h: throughput < 100 img/s @ bs32 mixed → abort
- 6h: FID > 60 → abort
- 12h: FID > 30 → abort
- 언제든: NaN/Inf, loss 발산, grad explosion, OOM 반복

abort 시 옵션 C(baseline 확장) 별도 구체화 — 그 시점에 결정.

## 코드 컨벤션

- 교수님 제공 baseline (`student_baseline_v0520.zip` 내부)의 `dataset.py`, `augment.py`는 **그대로 사용**, 1바이트도 수정 금지 (CLAUDE.md root §4 적용)
- baseline 패키지는 D path에서 미사용이나 C fallback 대비 `p2/student_baseline_v0520.zip`에 보관
- 학습 hyperparameter는 StyleGAN2-ADA 공식 값 사용 (lr 0.002, β2 0.99, R1 γ 10, PL weight 2.0, lazy 16/8)
