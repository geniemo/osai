# Project 1 — Pascal-VOC Semantic Segmentation (학번: 2020314315)

ResNet-50 + DeepLabV3+ (Output Stride 16). COCO+VOC 2-stage 학습 + Copy-Paste augmentation + Boundary-weighted CE loss.

## 환경 설정

```bash
cd p1
uv sync                                    # PyTorch 2.11 + CUDA 12.8 binary
uv run python -m src.data.download \
    --voc-root data/voc --coco-root data/coco   # VOC + COCO 다운로드 (~30-60분, 1회)
uv run python -m src.build_coco_masks \
    --coco-root data/coco --num-workers 4       # COCO mask cache 사전 생성 (~30-60분, 1회)
```

## 학습 (2-stage)

```bash
# Stage 1: COCO+VOC mixed, 160K iter, vanilla
uv run python -m src.train --config src/config/colab_v2_final_s1.yaml --stage 1

# Stage 2: VOC only, 8K iter, Copy-Paste + Boundary loss (Stage 1 best ckpt에서 bootstrap)
uv run python -m src.train --config src/config/colab_v2_final_s2.yaml --stage 2
```

중간 ckpt: `training_state_stage{1,2}.pth` (resume용), `best.pth` (val mIoU 갱신 시), `model.pth` (학습 종료, EMA only).

세션 끊겨도 동일 명령 재실행 시 ckpt에서 자동 resume + WandB run continue.

## 평가

```bash
# Raw single-scale mIoU
uv run python -m src.eval --config src/config/colab_v2_final_s2.yaml --ckpt checkpoints/best.pth

# TTA mIoU (multi-scale [0.5, 0.75, 1.0, 1.25, 1.5] + hflip)
uv run python -m src.eval_tta --config src/config/colab_v2_final_s2.yaml --ckpt checkpoints/best.pth
```

## 추론 (재현용)

```bash
uv run python -m src.infer \
    --config src/config/colab_v2_final_s2.yaml \
    --ckpt checkpoints/model.pth \
    --input submit/img --output submit/pred
```

`submit/img/0001.jpg` → `submit/pred/0001.png` (동일 파일명, PNG mode L, pixel ∈ [0, 20]).

TTA 기본 활성화. `--no-tta` 플래그로 비활성 가능.

## FLOPs 측정 (채점 기준)

```bash
# 1) ONNX export (가중치 제거, 채점용 graph)
uv run python -m src.export_onnx \
    --config src/config/colab_v2_final_s2.yaml \
    --ckpt checkpoints/model.pth \
    --out model_structure.onnx

# 2) FLOPs 측정 (입력 [1, 3, 480, 640], MAC × 2 = FLOP convention)
uv run python -m src.measure_flops --onnx model_structure.onnx
# → "[ONNX] model_structure.onnx: 162.24 GFLOPs"

# 3) (선택) PyTorch hook 기반 sanity check
uv run python -m src.measure_flops \
    --config src/config/colab_v2_final_s2.yaml \
    --ckpt checkpoints/model.pth
```

## 제출물 패키징

### 학교 사이트 (코드베이스 zip)

```bash
cd ..
zip -r 2020314315_project01.zip \
    p1/src p1/checkpoints/model.pth p1/submit \
    p1/2020314315_project01_report.pdf p1/pyproject.toml p1/README.md \
    -x '**/__pycache__/*'
```

### 채점 사이트 (PNG zip)

```bash
uv run python -m src.package_submission \
    --pred submit/pred \
    --out submission_pred.zip
```

검증: 1000 PNG (000.png~999.png), 픽셀 ∈ [0, 20], <500MB.

### 채점 사이트 (ONNX)

`model_structure.onnx` 그대로 업로드 (≤10MB, 입력 [1, 3, 480, 640]).

## Colab 재현 (전체 파이프라인)

`colab/colab_v2_final.ipynb` 노트북 한 번 실행으로 데이터 다운로드 → COCO mask cache → Stage 1 → Stage 2 → 평가 → ONNX → submission zip 모두 자동 생성.

T4/L4 GPU 권장. ckpt가 Drive에 저장되어 세션 끊겨도 resume.

## 라이브러리

- **torch 2.11 + cu128**, **torchvision** (CNN backbone, augmentation)
- **opencv-python**, **Pillow**, **numpy** (이미지 처리)
- **pycocotools** (COCO 라벨 매핑)
- **onnx** (ONNX 제출용)
- **wandb** (학습 로깅)

HuggingFace 라이브러리, Albumentations, PyTorch Lightning, Accelerate 등 modeling/training 보조 3rd party는 사용하지 않음.

## AI 도구

본 과제 진행 시 **Claude Code (Anthropic)** 와 **ChatGPT (OpenAI)** 를 코드 구현·디버깅·문서 작성 보조에 사용. 자세한 내역은 리포트의 "AI 도구 사용 내역" 절 참조.

## Tests

```bash
uv run pytest tests/ -q
```

Expected: 61 passed.
