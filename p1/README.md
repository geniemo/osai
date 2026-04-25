# Project 1 — Semantic Segmentation (학번: 2020314315)

Pascal-VOC 21-class semantic segmentation. ResNet-50 + DeepLabV3+ (OS=16) 디폴트, MobileNetV3-Large + LR-ASPP 백업. 자세한 설계는 `../docs/superpowers/specs/2026-04-24-p1-segmentation-design.md` 참조.

## Setup

```bash
cd p1
uv sync                                        # PyTorch + CUDA 12.8 binary
uv run python -m src.data.download             # VOC + COCO 다운 (최초 1회, ~30-60분)
```

## Training

```bash
# Stage 1 → Stage 2 자동 (resume 자동)
uv run python -m src.train --config src/config/default.yaml

# 또는 stage 명시
uv run python -m src.train --config src/config/default.yaml --stage 2

# 라이트 백업 (A)
uv run python -m src.train --config src/config/light.yaml
```

중간 ckpt: `checkpoints/training_state_stage{1,2}.pth` (resume용), `best.pth` (val mIoU 갱신 시), `model.pth` (학습 종료, EMA only).

## Evaluation (val mIoU)

```bash
uv run python -m src.eval --config src/config/default.yaml --ckpt checkpoints/model.pth
```

## Inference (TTA)

```bash
# 추론 (PDF 컨벤션: submit/img → submit/pred)
uv run python -m src.infer --config src/config/default.yaml \
    --ckpt checkpoints/model.pth \
    --input submit/img --output submit/pred
```

TTA: multi-scale [0.5, 0.75, 1.0, 1.25, 1.5] × hflip = 10× forward. `--no-tta`로 비활성화 가능.

## FLOPs Measurement (채점 기준)

```bash
# 1) ONNX export (가중치 제거, 10MB 이하)
uv run python -m src.export_onnx --config src/config/default.yaml \
    --ckpt checkpoints/model.pth --out model_structure.onnx

# 2) FLOPs 측정 (입력 [1, 3, 480, 640], ONNX 그래프 기준)
uv run python -m src.measure_flops --onnx model_structure.onnx
# → "[ONNX] model_structure.onnx: ~80 GFLOPs"

# 3) (optional) PyTorch sanity check
uv run python -m src.measure_flops --config src/config/default.yaml --ckpt checkpoints/model.pth
```

## Submission (3 채널)

### 1. 학교 사이트 (코드베이스 zip)

```bash
cd ..
zip -r p1/2020314315_project01.zip \
    p1/src p1/checkpoints/model.pth p1/submit \
    p1/2020314315_project01_report.pdf p1/pyproject.toml p1/README.md \
    -x '**/__pycache__/*'
```

### 2. 채점 사이트 — PNG zip

```bash
uv run python -m src.package_submission \
    --pred output/pred_FINAL \
    --out submission_pred.zip
# 검증: 1000 PNG, 000-999.png, 픽셀 [0,20], <500MB
```

### 3. 채점 사이트 — ONNX

`model_structure.onnx` 그대로 업로드 (≤10MB, 입력 [1,3,480,640]).

## Reproducibility

- 데스크탑(5070 Ti)에서 탐색 → Colab T4/L4에서 처음부터 전체 파이프라인 재실행이 제출물
- WandB Overview에 T4/L4 GPU 증거 자동 기록 (`run.summary["gpu_name"]`)
- `checkpoints/training_state_stage{1,2}.pth`로 세션 끊겨도 resume

## Library Notes

- **torch>=2.7 + cu128 binary** (5070 Ti Blackwell sm_120 호환)
- **HuggingFace, Albumentations 미사용** (PDF 정책 준수)
- **`onnx`**: ONNX 제출 형식 처리 (modeling/training 라이브러리 아님, 교수님이 가중치 제거 코드 직접 제공)
- **AI 도구**: Claude Code (설계, 코드 작성, 리뷰) — report에 사용 내역 기재

## Tests

```bash
uv run pytest tests/ -v
```

Expected: 34 passed, 1 skipped (VOC 데이터 없을 때)
