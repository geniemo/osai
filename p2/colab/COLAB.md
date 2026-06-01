# P2 D-256 Colab Reproduction Guide

p1 v2.final 패턴 — **매 세션 GitHub fresh clone**, Drive는 데이터 zip과 ckpt 보관 용도. 교수님이 우리 코드를 그대로 재현할 수 있도록 환경을 격리합니다.

## Pre-flight (재현자 + 학습자 모두 1회)

### Drive 폴더 구조

```
/MyDrive/osai/p2/
├── train_50k_256.zip          ← 학습 데이터 (필수, 1.65GB)
└── valid_10k_256.zip          ← FID 측정용 (선택, 0.33GB)
```

학습 중/후 자동 생성:
```
/MyDrive/osai/p2/
├── runs/d256_main/
│   ├── ckpt_XXXXXXXXX.pt      ← 200k images마다 자동 저장 (Disconnect 보존)
│   ├── final.pt               ← 학습 종료 시
│   ├── samples/grid_*.png
│   └── train.log
├── fid_stats_256.npz          ← FID 측정 1회 캐시
└── checkpoints/model.onnx     ← ONNX export 결과
```

### Colab Secret

좌측 사이드바 🔑 (Secrets) → 이름 `WANDB_API_KEY` 등록 + Notebook access 토글 ON.

### Colab GPU

런타임 → 런타임 유형 변경 → **A100 GPU**. L4도 가능 (학습 시간 ~2배 늘어남).

## 노트북 흐름 (10 cells)

1. **Drive mount**
2. **설정 변수** (`REPO_URL`, `BRANCH`, `DRIVE`, `RUN_DIR`, `DATA_LOCAL`)
3. **저장소 fresh clone** — `cd /content && rm -rf osai && git clone --branch improve --depth 1`
4. **의존성 install** — pip install
5. **GPU 확인**
6. **WandB 로그인** — Colab Secret `WANDB_API_KEY`
7. **학습 데이터 Colab 로컬로 cp** — Drive → `/content/p2_data/train_50k_256.zip` (throughput 이유)
8. **학습 launch** — `train.py --train-zip /content/p2_data/... --run-dir /content/drive/MyDrive/osai/p2/runs/d256_main`
9. (Disconnect 후) **Resume** — Drive에서 최신 ckpt 자동 탐색 후 `--resume`
10. **FID 측정** + **ONNX export**

## 학습 launch 명령 핵심

```bash
python p2/train.py \
    --config p2/configs/d256.yaml \
    --train-zip /content/p2_data/train_50k_256.zip \
    --run-dir /content/drive/MyDrive/osai/p2/runs/d256_main
```

- `--train-zip`로 데이터 경로 override (Colab 로컬 SSD)
- `--run-dir`로 ckpt 저장 위치 override (Drive 직접 저장)
- yaml의 hyperparameter는 그대로 (재현성)

## Resume 명령

```bash
python p2/train.py \
    --config p2/configs/d256.yaml \
    --train-zip /content/p2_data/train_50k_256.zip \
    --run-dir /content/drive/MyDrive/osai/p2/runs/d256_main \
    --resume /content/drive/MyDrive/osai/p2/runs/d256_main/ckpt_XXXXXXXXX.pt
```

`--resume`은 G/D/G_ema/optG/optD/RNG/pl_mean/wandb_run_id 모두 복원 → bit-for-bit 재개.

## ABORT GATE 모니터링

WandB(`ffhqgen-skku-p2/d256-stylegan2-scratch`):

| 시점 | 확인 | 기준 |
|---|---|---|
| 1h | throughput | < 100 img/s @ bs32 mixed → abort |
| 6h (~3M imgs) | FID 자기측정 | > 60 → abort |
| 12h (~7M imgs) | FID | > 30 → abort |
| 언제든 | loss/grad | NaN/Inf, 발산, OOM 반복 |

abort 시 학습 중단 + 결과 보고 → 옵션 C(baseline 확장) 별도 구체화.

## 재현성 보장

이 노트북을 그대로 실행하면:
- GitHub `geniemo/osai` repo의 `improve` 브랜치 HEAD를 fetch
- 동일 hyperparameter (`p2/configs/d256.yaml`)
- 동일 random seed (`42`)
- 동일 모델 아키텍처 (StyleGAN2 skip-G, 30.04M params)

→ 충분히 학습 시 동일 FID 도달. 재현 시 학습 시간만 환경(A100/L4/T4)에 따라 다름.

데이터(`train_50k_256.zip`)는 외부 자원이라 PDF spec의 "training data 위치"에 따라 제공.
