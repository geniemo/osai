# P2 — FFHQ 1024×1024 GAN (StyleGAN2-style, 256 native + bilinear→1024)

학번 2020314315 박지원. 2026-1 오픈소스AI실습 Project 2.

## Architecture

- **Generator** 30.04M params: StyleGAN2 skip-G (Mapping 8L + Synthesis Const 4×4 → ResBlocks → 256×256)
  - z 512 → MappingNet (8 FC, lr_mul 0.01) → w 512
  - SynthesisNet: const(4×4) → ModulatedConv + AddNoise + BiasAct (×N) → ToRGB (skip 합산)
  - 출력 256×256 → `SubmissionWrapper`가 bilinear로 1024×1024 resize + clamp(-1, 1)
- **Discriminator** 28.86M params: residual blocks + MinibatchStd + EqualLR

## Training environment

- Colab Pro+ A100 (가용 시) — 952 compute units → ~79h
- bf16 mixed precision
- 14M images 목표 (실측 ~11.2M 진행)
- ckpt 매 200k images마다 Drive에 저장 → disconnect 자동 복원

## Install dependencies

```bash
pip install -e .
# or
pip install torch>=2.1 torchvision>=0.16 numpy>=1.24 pillow>=10.0 \
    pyyaml>=6.0 wandb>=0.16 onnx>=1.15 onnxruntime>=1.16 \
    pytorch-fid>=0.3 scipy>=1.11
```

Python ≥ 3.10. GPU + CUDA 권장 (A100/L4/T4 모두 동작).

## How to train

```bash
# 데이터: train_50k_256.zip 을 data/ 디렉토리에 두기 (또는 --train-zip 으로 경로 지정)
PYTHONPATH=. python train.py \
    --config configs/d256.yaml \
    --train-zip data/train_50k_256.zip \
    --run-dir runs/d256_main
```

### Resume after disconnect
```bash
PYTHONPATH=. python train.py \
    --config configs/d256.yaml \
    --train-zip data/train_50k_256.zip \
    --run-dir runs/d256_main \
    --resume runs/d256_main/ckpt_XXXXXXXXX.pt
```

`--resume`은 G/D/G_ema/optG/optD/RNG/pl_mean/wandb_run_id 모두 복원하여 bit-for-bit 이어 학습.

### Hyperparameters (configs/d256.yaml)
- Loss: NS-Logistic + R1 (lazy 16, γ=10) + Path Length (lazy 8, weight 2.0)
- Optimizer: Adam β=(0, 0.99), lr=2e-3 both G/D
- EMA half-life: 20k images
- Augment: DiffAug color + translation
- Style mixing prob: 0.9
- Precision: bf16 mixed
- Batch size: 32, total 14M images

## How to generate images from noise

```bash
# 64장 sample grid PNG 생성 (EMA G 사용, 기본)
PYTHONPATH=. python generate.py \
    --ckpt checkpoints/model.pth \
    --out samples.png \
    --n 64 \
    --seed 12345
```

`checkpoints/model.pth`는 EMA Generator state + meta(generator_config)만 포함된 slim 파일.

### ONNX (leaderboard 제출과 동일 형식)
```bash
# (B, 512) z → (B, 3, 1024, 1024) image, dynamic batch
PYTHONPATH=. python export_onnx.py \
    --ckpt checkpoints/model.pth \
    --out model.onnx \
    --dynamic-batch
```

ONNX 추론:
```python
import onnxruntime as ort, numpy as np
sess = ort.InferenceSession("model.onnx")
z = np.random.randn(8, 512).astype(np.float32)
img = sess.run(None, {"z": z})[0]  # (8, 3, 1024, 1024), range [-1, 1]
```

## How to evaluate FID (self-measurement)

```bash
# 1회: real-side stats 캐시
python -m pytorch_fid path/to/valid_dir checkpoints/fid_stats_256.npz --save-stats

# ckpt FID
PYTHONPATH=. python eval_fid.py \
    --ckpt checkpoints/model.pth \
    --stats checkpoints/fid_stats_256.npz \
    --n 8000 --batch 32
```

## Submission contents

| 파일 | 용도 |
|---|---|
| `src/` | 모듈 (networks/, losses, augment, dataset, fid, utils) |
| `configs/d256.yaml` | 학습 hyperparameters |
| `train.py` | 학습 진입점 |
| `generate.py` | sample grid 생성 |
| `export_onnx.py` | ONNX export (리더보드용) |
| `eval_fid.py` | 자체 FID 측정 |
| `checkpoints/model.pth` | slim G_ema state + meta (제출용, ~120 MB) |
| `2020314315_project02_report.pdf` | 6 pages report |
| `pyproject.toml` | 의존성 명세 |
| `README.md` | (이 파일) |

리더보드에는 `model.onnx`를 별도 제출 (위 ONNX export 명령으로 생성).

## AI tools used

Claude Code (Anthropic Opus 4.x) — 설계, 코드 작성, ONNX export 디버깅, 보고서 초안 검토. 리포트 §AI Usage 참조.
