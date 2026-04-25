# Project 1 — Semantic Segmentation Design Spec

> 학번 2020314315, 마감 2026-05-05 23:59. 본 문서는 brainstorm 결과를 정리한 설계 명세이며, 후속 implementation plan의 근거 문서가 됩니다.

---

## 1. Goal & Scoring

### 1.1 Task

Pascal-VOC 21-class semantic segmentation. 모든 픽셀을 21개 클래스(background + 20)로 분류. mIoU(background 포함)이 기본 metric. ignore label(255)은 loss와 mIoU 양쪽에서 제외.

### 1.2 Scoring formula (외부)

```
S = S_mIoU × S_FLOPs + S_Code + S_Report
```

- `S_mIoU`: 비공개 GT 기준 mIoU (21 클래스), 0-3-4-5 absolute scale (threshold는 4/29 공지)
- `S_FLOPs`: ONNX 그래프에서 입력 [1, 3, 480, 640]로 측정, 0-3-4-5 absolute scale
- `S_Code`: 5점 — 모듈 구조, 재현성, ckpt 처리, 가독성, 수업 내용 활용
- `S_Report`: 5점 — architecture, training recipe, validation, ablation, failure analysis, WandB evidence

`S_mIoU × S_FLOPs`는 **곱**이라 둘 중 하나가 0이면 성능 점수 전체 0.

### 1.3 Submission channels (총 3건)

| 채널 | 무엇 | 형식 | 제한 | 점수 |
|---|---|---|---|---|
| A. 학교 사이트 | 코드베이스 zip | `2020314315_project01.zip` (PDF 구조) | — | S_Code, S_Report |
| B-1. 채점 사이트 | PNG 추론 결과 | flat zip, `000.png`~`999.png` | 압축해제 ≤500MB | S_mIoU |
| B-2. 채점 사이트 | 모델 구조 ONNX | `.onnx` (가중치 제거) | ≤10MB | S_FLOPs |

---

## 2. Constraints

### 2.1 Library

- **허용**: PyTorch, TorchVision, OpenCV, Pillow, Scikit-Image, Matplotlib, WandB, pycocotools, onnx
- **금지**: HuggingFace 전체, Albumentations, Lightning, Accelerate, 그 외 modeling/data/aug/training/eval 지원하는 3rd party
- **`onnx` 정당화**: 모델 포맷/serialization 라이브러리이지 modeling/training/eval 지원 도구가 아님. 교수님이 ONNX 제출을 명시적으로 요구 + 가중치 제거 코드 직접 제공 → 사실상 명시적 허용
- **AI 도구**: Claude Code 사용 — **리포트에 사용 내역(어떤 부분에 어떻게 도움 받았는지) 기록 필수** (PDF 슬라이드 8 명시)

### 2.2 Data

- **허용**: ImageNet-1K, MS-COCO, Pascal-VOC
- **금지**: VOC val/test annotation을 학습에 사용
- **외부 평가셋**: `p1/submit/img/`에 1000장 (000.jpg~999.jpg) 공개. 비공개 GT로 채점 (PDF 14p 컨벤션)

### 2.3 Modeling

- CNN만 (no RNN/Transformer)
- TorchVision **classification** pretrained만 사용 가능 (segmentation pretrained 금지)
- Quantized 금지

### 2.4 Reproducibility

- 데스크탑(RTX 5070 Ti)에서 탐색 → Colab T4/L4에서 처음부터 전체 파이프라인 재실행이 제출물
- 재현은 "같은 설계를 재실행"이지 "같은 mIoU 숫자"는 아님 (±1-2% 자연)
- WandB Overview에 T4/L4 GPU 증거 필수
- Resumable training 필수 (model/opt/sched/scaler/epoch/RNG state + WandB run resume)

---

## 3. Locked Design Decisions

| 항목 | 결정 | 근거 |
|---|---|---|
| Output classes | VOC 21 (background + 20) | test set이 VOC 도메인 (샘플 확인) |
| Approach | (가) **B 디폴트 + A 백업, 모듈러 swap** | mIoU 5점 노리고 4/29 threshold 따라 fallback |
| Backbone B | ResNet-50 (TorchVision IMAGENET1K_V2), OS=16 | mIoU 상한 + ONNX 호환 |
| Neck/Head B | ASPP (rates [6,12,18]) + DeepLabV3+ decoder w/ low-level skip | DLv3+ 표준 |
| Backbone A | MobileNetV3-Large (TorchVision IMAGENET1K_V2), dilated | FLOPs 5점 보장 |
| Neck/Head A | LR-ASPP | 경량 |
| Output stride | 16 | mIoU/FLOPs 균형 |
| 학습 입력 | random scale [0.5, 2.0] + 480×480 crop | DeepLab 표준 |
| 추론 입력 | 원본 aspect 보존 + multi-scale TTA + hflip | mIoU 추가 (FLOPs 점수 무영향) |
| FLOPs 측정 입력 | (1, 3, 480, 640) ONNX 단발 forward | 채점 사양 |
| 데이터 전략 | (C) **VOC + COCO 2-stage** (Stage1=mixed, Stage2=VOC train only — SBD 미사용) | PDF 명시 안 된 SBD 보수적 제외 |
| Augmentation | 기본 5개 + GaussianBlur + RandomErasing + RandomGrayscale | torchvision.v2 표준 + cheap wins |
| COCO 재현 | Colab에서도 처음부터 다운로드 + mask 생성 | 재현성 100% |
| Loss | CE(ignore=255) + 0.5 Dice + 0.4 Aux CE on layer3 | DLv3 표준 + class imbalance |
| Optimizer | SGD + momentum 0.9 + wd 5e-4, backbone 0.1× LR | DeepLab 표준 |
| LR scheduler | Polynomial decay (power=0.9) + linear warmup | DLv3 표준 |
| Mixed precision | fp16 (T4 호환) + GradScaler | T4/L4/5070Ti 공통 |
| EMA | decay 0.9999, validation/inference/export에 사용 | +0.5~1 mIoU |
| ONNX export | EMA model, opset 17, 가중치 제거, 고정 shape | 10MB 한도 |
| ONNX FLOPs counter | (b) `onnx` 라이브러리 + 직접 graph 순회 (custom) | 라이브러리 정책 안전 + 표준 공식 |
| WandB project | `osai-p1-local` (entity: `g1nie-sungkyunkwan-university`) | 단일 project, tags로 desktop/colab 구분 |
| Random seed | 42, cudnn benchmark=True (속도 우선) | 결정성 절대값 아닌 mIoU 점수 |

---

## 4. Project Structure

`p1/` 자체가 학교 사이트 zip 루트.

```
p1/
├── pyproject.toml                       # uv project (HF 의존성 없음)
├── uv.lock
├── README.md                            # 학습/설치/추론/FLOPs/재현/제출
├── 2020314315_project01_report.pdf      # 최종 리포트 (마감 직전)
├── checkpoints/
│   └── model.pth                        # EMA model_state만 (제출용, 가벼움)
├── submit/
│   ├── img/                             # 빈 폴더 (placeholder, PDF 요구)
│   └── pred/                            # 빈 폴더 (reproduce 결과 위치)
├── input/
│   └── (legacy, 미사용)                  # test image는 submit/img/로 이동
├── output/                              # 추론 결과 (gitignore)
│   └── pred_<TAG>/                      # 실험별 prediction
└── src/
    ├── __init__.py
    ├── train.py                         # 학습 진입점 (resume, AMP, EMA, WandB)
    ├── eval.py                          # val mIoU 측정 진입점
    ├── infer.py                         # input → output (TTA)
    ├── export_onnx.py                   # model → ONNX (구조만)
    ├── measure_flops.py                 # PyTorch + ONNX 양쪽 측정
    ├── package_submission.py            # output/ → 채점용 .zip 검증/패키징
    ├── config/
    │   ├── default.yaml                 # B (ResNet50 + DLv3+)
    │   ├── light.yaml                   # A (MobileNetV3 + LR-ASPP)
    │   └── colab.yaml                   # Colab override
    ├── data/
    │   ├── __init__.py
    │   ├── builder.py                   # build_dataloaders(cfg)
    │   ├── voc.py                       # Pascal-VOC train + val (SBD 미사용)
    │   ├── coco.py                      # COCO + VOC class mapping + mask cache
    │   ├── transforms.py                # torchvision.v2 joint image-mask
    │   └── download.py                  # 자동 다운로드 (재현성)
    ├── models/
    │   ├── __init__.py
    │   ├── builder.py                   # build_model(cfg) — backbone/neck/head swap
    │   ├── seg_model.py                 # SegmentationModel + export_mode()
    │   ├── backbones/
    │   │   ├── resnet.py                # TorchVision pretrained, OS=16 (dilated)
    │   │   └── mobilenet.py             # TorchVision MobileNetV3-Large
    │   ├── necks/
    │   │   ├── aspp.py                  # ASPP (DeepLabV3+ style)
    │   │   └── lr_aspp.py               # LR-ASPP (light)
    │   ├── heads/
    │   │   └── deeplabv3plus.py         # decoder + classifier
    │   └── aux/
    │       └── fcn_head.py              # 보조 CE head (학습 전용)
    ├── losses/
    │   └── seg_loss.py                  # CE + Dice + Aux 결합, ignore=255
    └── utils/
        ├── metrics.py                   # ConfusionMatrix mIoU
        ├── flops.py                     # PyTorch counter (w4) + ONNX counter (custom)
        ├── checkpoint.py                # save/load: model/ema/opt/sched/scaler/iter/RNG
        ├── seed.py                      # set_seed, RNG state get/set
        └── viz.py                       # mask 시각화 (failure case 분석)
```

### 4.1 진입점 (README에 들어갈 명령)

```bash
# 학습 (Stage 1 → Stage 2 자동 전이, resume 자동)
python -m src.train --config src/config/default.yaml

# val mIoU
python -m src.eval --config src/config/default.yaml --ckpt checkpoints/model.pth

# 추론 (TTA, PDF 컨벤션: submit/img → submit/pred)
python -m src.infer --config src/config/default.yaml \
    --ckpt checkpoints/model.pth \
    --input submit/img --output submit/pred

# 채점용 PNG zip
python -m src.package_submission --pred submit/pred --out submission_pred.zip

# ONNX export (가중치 제거)
python -m src.export_onnx --config src/config/default.yaml \
    --ckpt checkpoints/model.pth --out model_structure.onnx

# FLOPs 측정 (PyTorch + ONNX 양쪽)
python -m src.measure_flops --config src/config/default.yaml --ckpt checkpoints/model.pth
python -m src.measure_flops --onnx model_structure.onnx
```

---

## 5. Data Pipeline

### 5.1 데이터셋 구성

| Stage | 데이터셋 | 이미지 수 | 출처 |
|---|---|---|---|
| Stage 1 (사전학습) | COCO 2017 train (VOC subset) | ~95K (필터 후) | cocodataset.org (~25GB) |
| Stage 1 (사전학습) | VOC 2012 train | 1,464 | VOC 공식 |
| Stage 2 (파인튜닝) | VOC 2012 train | 1,464 | 위와 동일 |
| Validation (양 stage) | VOC 2012 val | 1,449 | VOC 공식 |

- **SBD (Semantic Boundary Dataset) 사용 안 함** — PDF 정책에 ImageNet/COCO/VOC만 명시되어 있고 SBD는 별도 데이터셋. 보수적 해석으로 제외. 교수님이 4/29 QA에서 SBD 허용 명시 시 `voc.py`에 `split="trainaug"` 분기 + `download.py`에 `download_sbd_and_merge` 다시 추가하면 즉시 활성화 가능
- VOC 2012 val은 학습 절대 금지 (코드 경로 분리)
- **`p1/submit/img/` 격리 가드 (개발 안전망)**: train/val DataLoader가 어떤 코드 경로로도 접근 금지. `data/builder.py`에서 `voc_root` / `coco_root` 경로 normalize 후 `submit/` 또는 `input/`이 path component로 포함되면 assertion 실패시켜 학습 중단. test image는 `infer.py`/`eval.py`에서만 입력으로 받음

### 5.2 COCO → VOC 클래스 매핑

20-class hard-coded mapping (`src/data/coco.py`):

```python
COCO_TO_VOC = {
    1:15, 2:2, 3:7, 4:14, 5:1, 6:6, 7:19, 9:4, 16:3, 17:8,
    18:12, 19:13, 20:17, 21:10, 44:5, 62:9, 63:18, 64:16, 67:11, 72:20,
}
# VOC 0 = background, 255 = ignore
```

### 5.3 COCO 마스크 생성

- pycocotools로 polygon/RLE → binary mask 변환
- VOC class object → 해당 VOC class id로 paint
- **Non-VOC class object → 255 (ignore)** — false negative 방지
- 사전 캐싱: `data/coco/coco_voc_masks/{image_id}.png`로 저장 (1회만 ~30-60분)

### 5.4 Augmentation (joint image-mask)

torchvision.transforms.v2 사용 (`tv_tensors.Mask`로 NEAREST + ignore=255 fill 자동).

| 단계 | 종류 | 적용 |
|---|---|---|
| 1 | RandomScale [0.5, 2.0] | image + mask |
| 2 | RandomCrop 480×480 (pad image=0, mask=255) | image + mask |
| 3 | RandomHorizontalFlip p=0.5 | image + mask |
| 4 | ColorJitter (b/c/s=0.4, h=0.1) | image only |
| 5 | GaussianBlur (p=0.3, σ ∈ [0.1, 2.0]) | image only |
| 6 | RandomGrayscale (p=0.1) | image only |
| 7 | RandomErasing (p=0.25) | image only |
| 8 | ToDtype(float32, scale=True) + Normalize (ImageNet) | image only |

Validation transform: ToDtype + Normalize만 (no aug, 원본 크기 유지).

### 5.5 2-stage 데이터 결합

- **Stage 1**: `ConcatDataset(coco_voc(~95K), voc_train(1.4K))` 단순 random shuffle (COCO 강하게 dominant, 광범위 학습 목적)
- **Stage 2**: VOC train(1,464장) only — SBD 미사용으로 데이터 적음, **stage2_iters를 30K → 8K로 축소** (88 epoch 정도, overfitting 방지)

### 5.6 DataLoader

```yaml
data:
  voc_root: ./data/voc
  coco_root: ./data/coco
  batch_size: 16        # 5070 Ti 16GB / T4 16GB / L4 24GB
  num_workers: 8
  pin_memory: true
  persistent_workers: true
  crop_size: 480
  scale_range: [0.5, 2.0]
```

---

## 6. Model Architecture

### 6.1 Backbone B: ResNet-50, OS=16

`resnet50(weights=ResNet50_Weights.IMAGENET1K_V2, replace_stride_with_dilation=[False, False, True])`.

Forward (입력 480×640 기준):

```
x: (B, 3, 480, 640)
  ↓ stem
x: (B, 64, 120, 160)        H/4
  ↓ layer1
c2: (B, 256, 120, 160)      H/4   ← low-level skip
  ↓ layer2 (stride 2)
c3: (B, 512, 60, 80)        H/8
  ↓ layer3 (stride 2)
c4: (B, 1024, 30, 40)       H/16
  ↓ layer4 (dilated, no stride)
c5: (B, 2048, 30, 40)       H/16  ← high-level → ASPP
```

`forward()` 반환: `(c2, c5)` 두 텐서.

### 6.2 Neck B: ASPP

```
input (B, 2048, 30, 40)
  ├─ 1×1 conv → 256                      
  ├─ 3×3 conv, dilation=6  → 256          
  ├─ 3×3 conv, dilation=12 → 256
  ├─ 3×3 conv, dilation=18 → 256
  └─ AdaptiveAvgPool2d(1) → 1×1 conv → 256 → bilinear upsample
concat (1280) → 1×1 conv → 256 → Dropout(0.5)
```

### 6.3 Head B: DeepLabV3+ Decoder

```
ASPP out (B, 256, 30, 40) → bilinear upsample ×4 → (B, 256, 120, 160)
c2 (B, 256, 120, 160) → 1×1 conv → 48 → (B, 48, 120, 160)
concat → (B, 304, 120, 160)
3×3 conv → 256 → 3×3 conv → 256 → 1×1 conv → 21
bilinear upsample ×4 → (B, 21, 480, 640)
```

### 6.4 Backbone A + Neck A: MobileNetV3-Large + LR-ASPP

`mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)` + 마지막 블록 dilation으로 OS=16. low-level (40ch, H/8) + high-level (960ch, H/16).

LR-ASPP head: high-level branch (1×1 conv 128 + sigmoid attention from GAP) × low-level branch (1×1 conv 21) → 8× upsample.

### 6.5 Aux Head (학습 전용)

layer3(c4, 1024ch) → 1×1 conv → 256 → BN → ReLU → 1×1 conv → 21 → upsample to input size.

**ONNX export 시 `model.export_mode()` 호출하면 aux_head는 forward에서 제거** → FLOPs 점수에 무영향.

### 6.6 FLOPs 추정 (구현 후 ONNX counter로 확정)

| 컴포넌트 | B | A |
|---|---|---|
| Backbone | ~32 GFLOPs | ~1.8 GFLOPs |
| Neck | ~8 | ~0.3 |
| Head | ~5 | ~0.2 |
| **Total** | **~45** | **~2.3** |

### 6.7 Pretrained 정책

- IMAGENET1K_V2 사용 (V1 대비 +0.5 mIoU)
- Backbone 전체 trainable (frozen 아님)
- BN momentum 기본 0.1 유지
- 입력 정규화: ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

---

## 7. Loss + Training Strategy

### 7.1 Loss

```
L = CE(main, ignore=255) + 0.5 × Dice(main, ignore=255) + 0.4 × CE(aux, ignore=255)
```

Dice 구현: per-class (background 포함) + ignore mask 적용 + epsilon 1e-6.

### 7.2 Optimizer

```python
SGD([
    {"params": backbone.parameters(), "lr": base_lr * 0.1},
    {"params": head.parameters(),     "lr": base_lr},
    {"params": aux_head.parameters(), "lr": base_lr},
], momentum=0.9, weight_decay=5e-4)
```

### 7.3 LR Schedule

Polynomial decay: `lr(t) = base_lr * (1 - t/T)^0.9`. Linear warmup 첫 1000 iters (Stage 1), 500 iters (Stage 2).

### 7.4 2-stage cycle

| 항목 | Stage 1 (COCO+VOC pretrain) | Stage 2 (VOC finetune) |
|---|---|---|
| 데이터 | COCO_VOC(~95K) + VOC_train(1.4K) concat | VOC_train(1,464) only |
| iters | 80,000 | **8,000** (SBD 미사용 → 데이터 적어 축소) |
| batch_size | 16 | 16 |
| base_lr | 0.01 | 0.001 |
| warmup | 1000 iters | 500 iters |
| schedule | poly 0.9 | poly 0.9 |
| validation | 매 5,000 iters | 매 1,000 iters |

Stage 1 종료 → best ckpt 자동 로드 → Stage 2 시작 (같은 train.py). Stage 2 8K iter ≈ 88 epoch (1.4K samples / batch 16 = 91 iter/epoch).

### 7.5 Mixed precision (fp16)

`torch.amp.autocast('cuda', dtype=torch.float16)` + `GradScaler`. unscale 후 grad clip max_norm=1.0. Scaler state도 ckpt에 포함.

### 7.6 EMA

- decay 0.9999
- 매 step (warmup 후) EMA 업데이트
- BN running stats도 EMA buffer로 동기화
- Validation, final inference, ONNX export 모두 **EMA model 사용**

### 7.7 Resumable training

Checkpoint 구조:

```python
{
    "iter": int,
    "stage": int,                       # 1 or 2
    "model_state": ...,
    "ema_state": ...,
    "optimizer_state": ...,
    "scheduler_state": ...,
    "scaler_state": ...,
    "best_miou": float,
    "rng_state": {python, numpy, torch, cuda},
    "wandb_run_id": str,
    "config": dict,                     # 검증용
}
```

저장 정책:
- `checkpoints/training_state.pth` — 매 5000 iters 덮어쓰기 (resume용)
- `checkpoints/best.pth` — val mIoU 갱신 시 (EMA + raw)
- `checkpoints/model.pth` — 학습 종료 시 EMA만 (제출용)

자동 resume: `train.py` 시작 시 `training_state.pth` 존재하면 자동 로드 + WandB run id로 resume + config diff 비교.

### 7.8 Reproducibility

```python
seed = 42
random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True   # 속도 우선
# DataLoader worker_init_fn으로 worker별 seed 고정
```

### 7.9 WandB

```python
wandb.init(
    entity="g1nie-sungkyunkwan-university",
    project="osai-p1-local",
    name=f"{stage}_{date}_{backbone}_{exp_id}",
    config=cfg,
    id=ckpt.get("wandb_run_id"),
    resume="allow",
    tags=[env, data_combo, stage_tag],
)
```

로깅:
- per-step (50 iters): loss components, lr (backbone/head), iter_sec
- per-validation: mIoU, mIoU_ema, per-class IoU × 21, best_mIoU
- summary: params total/trainable, GFLOPs (PyTorch + ONNX), GPU name, data stage

GPU 증거: `wandb.run` 시스템 metrics에 자동 기록 + `summary["gpu_name"]` 명시.

**Run 가시성**: WandB run은 **public 또는 entity team 공유 권한 유지** (PDF "마감 후 수업 시간에 WandB 페이지 직접 확인" 요구). private/삭제 금지. team `g1nie-sungkyunkwan-university` project 안에 두면 교수님 접근 가능 (필요 시 share link 명시).

### 7.10 학습 시간 추정

| 환경 | Stage 1 (80K) | Stage 2 (8K) | 합계 |
|---|---|---|---|
| 5070 Ti | ~2.2h | ~0.2h | ~2.4h |
| L4 | ~3.2h | ~0.3h | ~3.5h |
| T4 | ~4.4h | ~0.5h | ~4.9h |

---

## 8. Evaluation / Inference / Submission

### 8.1 mIoU (학습 중 validation)

ConfusionMatrix 누적, ignore=255 제외, 클래스 c가 GT/Pred 모두 안 나타나면 nanmean으로 평균에서 제외.

```python
class SegMetric:
    def update(self, pred, target):
        valid = target != 255
        idx = target[valid] * 21 + pred[valid]
        self.cm += torch.bincount(idx, minlength=441).view(21, 21)
    def compute(self):
        tp = self.cm.diag().float()
        denom = self.cm.sum(0) + self.cm.sum(1) - tp
        iou = torch.where(denom > 0, tp / denom, torch.nan)
        return torch.nanmean(iou).item(), iou.tolist()
```

### 8.2 추론 (test → PNG)

- 원본 이미지 크기 유지
- TTA: scales [0.5, 0.75, 1.0, 1.25, 1.5] × hflip [F, T] = 10번 forward
- 각 scale에서 logits → original H×W로 upsample → softmax → hflip 보정 → 누적
- argmax → uint8 [0..20] → PNG mode='L' 저장
- 입력 dim은 32 배수로 round (OS=16 정렬)

### 8.3 PNG 출력 형식

- 이름: `000.png` ~ `999.png` (3-digit, 입력 jpg와 매칭)
- mode='L', 단일 채널 uint8, 픽셀 [0, 20]
- 크기 = 원본 jpg와 동일
- 1000장 합계 ~10-30MB (500MB 한도 충분)

### 8.4 ONNX Export

```python
model.eval()
model.export_mode()                  # aux_head 제거
dummy = torch.zeros(1, 3, 480, 640)
torch.onnx.export(
    model, dummy, "model.onnx",
    input_names=["input"], output_names=["logits"],
    dynamic_axes=None,               # 고정 shape
    opset_version=17,
    do_constant_folding=True,
)
# 가중치 제거 (교수님 제공 코드)
m = onnx.load("model.onnx")
for init in m.graph.initializer:
    init.ClearField("raw_data")
    init.ClearField("float_data")
    init.ClearField("int32_data")
    init.ClearField("int64_data")
onnx.save(m, "model_structure.onnx")
# 검증: ≤10MB
```

EMA 가중치 사용. 예상 크기 ~1-2MB.

### 8.5 ONNX FLOPs Counter (custom)

- `onnx.shape_inference.infer_shapes`로 모든 중간 shape 도출
- node iterate, op_type별 FLOPs:
  - **Conv**: `N × C_out × H_out × W_out × (C_in/groups) × K_h × K_w` (MAC, ×2 안 함)
  - **Gemm/MatMul**: `M × K × N`
  - **GlobalAveragePool, Resize**: 옵션 (작아서 무시 가능)
  - **BN, ReLU, Add, Mul, Concat**: 0 (관행)
- PyTorch counter (w4 `compute_flops`)와 cross-check, ±5% 안에 들어와야 정상

### 8.6 Submission packaging

#### Channel A (학교 사이트)
```bash
zip -r 2020314315_project01.zip \
    src/ checkpoints/model.pth submit/ \
    2020314315_project01_report.pdf pyproject.toml README.md \
    -x "**/__pycache__/*"
```

#### Channel B-1 (PNG zip)
```bash
python -m src.package_submission --pred submit/pred --out submission_pred.zip
```

`package_submission.py` 검증:
- 정확히 1000 파일
- 모든 이름 `^\d{3}\.png$`
- flat root
- 픽셀 [0, 20] 검증
- 압축해제 ≤500MB

#### Channel B-2 (ONNX)
```bash
python -m src.export_onnx --ckpt checkpoints/model.pth --out model_structure.onnx
```

자동 검증: 입력 [1,3,480,640], ≤10MB, 가중치 없음.

---

## 9. Workflow & Phases

날짜는 명시하지 않음 — 가능한 빨리 진행. 외부 이벤트(4/29 threshold, 5/5 마감)만 고정.

| Phase | 작업 | 산출물 |
|---|---|---|
| 0. Design + Plan | brainstorm → spec → implementation plan | spec.md (이 문서), plan.md |
| 1. Foundation | p1/pyproject, data pipeline (VOC+COCO+aug+mapping), utils (metrics, flops, ckpt, seed) | dataloader 동작, 단위 검증 |
| 2. Model | ResNet50/MobV3 backbones, ASPP/LR-ASPP necks, DLv3+/LR-ASPP heads, builder swap, aux head | forward shape + ONNX export 검증 |
| 3. Training loop | train.py iter-based, 2-stage dispatch, AMP, EMA, ckpt full-state, WandB, resume | smoke test (1K iter) + resume 검증 |
| 4. End-to-end smoke | 짧은 학습 (10K + 5K iter) on 5070 Ti | 첫 mIoU, ONNX <10MB 확인 |
| ★ 4/29 Threshold | (외부) | B/A/C 결정 (model-architect+efficiency+loss 팀) |
| 5. Full ablation | 데스크탑 full scale (80K+30K), 3-4개 ablation | best config 선정 |
| 6. Colab 공식 run | Drive에 COCO 업로드 → Colab T4/L4에서 처음부터 학습 | T4/L4 GPU 증거 있는 WandB run |
| 7. Submission + Report | TTA 추론 → PNG zip + ONNX export, report PDF (6p, 11pt) **with AI usage section**, 코드베이스 zip | 3 제출물 완성 |
| 8. Buffer | 막판 수정 | 제출 (5/5 23:59 전) |

### 9.1 Agent Team 매핑

| Phase | Agent / Team | 메커니즘 |
|---|---|---|
| 1 | data-augmentation-engineer + miou-specialist | parallel 단발 |
| 2 | model-architect | 단발 |
| 3 | training-strategist | 단발 |
| 4 | debugger-competitor × 3-4 | 팀 (필요 시) |
| 4/29 결정 | model-architect + efficiency-optimizer + loss-designer | 팀 |
| 5 | training-strategist + efficiency-optimizer | 팀 (mid-experiment) |
| 6 | colab-reproducer | 단발 |
| 7 | submission-checker + wandb-inspector + colab-reproducer | 팀 |

토큰 비용 관리: 팀 호출 시 `"Use Sonnet for each teammate"` 명시 가능 (`agent-team-design.md` §2.3).

---

## 10. Risk Management

| 위험 | 확률 | 영향 | 완화 |
|---|---|---|---|
| 5070 Ti CUDA/PyTorch 호환성 | 중 | 데스크탑 학습 불가 | **Phase 1 첫 task로 PyTorch import + cuda available + sm_120 호환 검증** (안 되면 Colab Plan B 즉시 전환) |
| COCO 다운로드 느림 (~25GB) | 중 | Phase 5 지연 | Phase 1 시작 동시 background로 |
| Colab 세션 끊김 | 높음 | 재개 필요 | resume first-class (Phase 3 핵심) |
| 4/29 threshold harsh (예: 5점=≤5GFLOPs) | 낮음 | B 폐기 → A 전환 | 모듈러로 config 1줄 변경 |
| ONNX export op 미지원 | 낮음 | FLOPs 채점 0점 | Phase 2에서 export 검증 |
| ONNX FLOPs counter 교수님 도구와 불일치 | 중 | FLOPs 점수 추정 오차 | 직접 구현 + PyTorch counter cross-check + report 명시 |
| 큰 이미지 OOM | 낮음 | 일부 추론 실패 | 사전 크기 검사, 큰 건 batch 1 또는 sliding |
| 리포트 6p 초과 | 낮음 | 채점 페널티 | Phase 7 시작 시 outline 먼저 |

---

## 11. Open Questions / Tradeoffs (추적용)

- **TTA scale 개수**: 5-scale 풀로 진행. 시간 부담 시 [0.75, 1.0, 1.25] 3-scale로 축소 가능 (-0.3 mIoU)
- **Validation TTA**: 미적용 (시간 ×10). final inference에만
- **Class weighting**: CE에 class weight 안 줌 (Dice가 imbalance 흡수)
- **PNG palette mode**: mode='L' 기본 (단순). palette는 시각화 시 별도
- **Stage 1 epoch 양**: 80K iter (~12 epoch). 더 늘리면 diminishing returns
- **EMA decay**: 0.9999 (effective window ~10K iter). 0.999면 reactive
- **Aux loss 가중치**: 0.4 (DLv3 표준). 학습 안정 도움
- **데이터 sampling 가중치**: Stage 1에서 random shuffle (oversampling 안 함). VOC oversample은 Stage 2가 담당
- **SBD 사용 여부 (PENDING — 교수님 확인 대기)**: 현재 SBD 미사용 (PDF 명시 X로 보수적 제외). 4/29 QA에서 허용 확인 시 즉시 활성화. SBD 활성화 시 VOC 학습 데이터 1,464 → 10,582로 7배 증가, mIoU +5~10% 기대

---

## 12. Locked deliverables

브레인스토밍 결과로 다음이 잠금:
- 6개 설계 섹션 (project structure / data / model / loss+training / eval+infer+submit / workflow)
- 모듈러 구조 (B 디폴트 + A 백업, config 1줄로 swap)
- 3 채널 제출 흐름 (codebase zip + PNG zip + ONNX)
- ONNX 기반 FLOPs 측정 (custom counter)
- 2-stage 학습 (COCO+VOC pretrain → VOC finetune)
- EMA + AMP + resume + WandB + agent team Phase 매핑

### 12.1 수업 내용 활용 (S_Code 5점 항목)

PDF "Try to use many things learned in class" 충족을 위해 다음 자산 명시적 재사용/적용:

- **w4 reference 재사용**: `w4/utils/compute_utils.py` (PyTorch FLOPs counter) → `src/utils/flops.py`로 이식 (import 경로 + ONNX counter 추가)
- **w4 architectural reference**: `w4/models/deeplab_v3.py` (ASPP/Backbone/DLv3 구조 educational 주석 풍부) → `src/models/necks/aspp.py` 작성 시 reference로 사용 (단, TorchVision pretrained backbone으로 변경 필요)
- **w5 training pattern reference**: `w5/train.py` (iter-based loop + WandB integration + AverageMeter + best ckpt 패턴 + YAML config) → `src/train.py` 작성 시 동일 골격 적용 (단, segmentation-specific 확장 필요)
- **수업 lecture 적용**: w4 lecture/ex0X 시리즈 (BatchNorm, GroupNorm, conv 1x1, batch norm 등) 개념을 model code 주석에서 명시적으로 reference (예: ASPP의 1×1 conv가 왜 채널 mixing인지, BN momentum 선택 근거)

**리포트 §Code Quality에 위 재사용 매핑을 표로 명시.**

### 12.2 리포트 필수 섹션 (S_Report 5점 항목)

리포트 (PDF, 6p 이하, 11pt) 구성:

1. **Model Architecture** — backbone/neck/head 구조 + 수업 내용 매핑
2. **Training Recipe** — 2-stage, optimizer, scheduler, AMP, EMA, loss 결합
3. **Validation Results** — 최종 mIoU + per-class IoU + WandB curve 캡처
4. **Ablation / Trial History** — 데스크탑 탐색 중 시도한 변형들 (LR, aug, dice 가중치 등)
5. **Failure Case Analysis** — 약한 클래스 / 잘못된 sample 시각화 (`utils/viz.py`)
6. **WandB Evidence** — Overview page 캡처 (T4/L4 GPU 증거 포함), monitoring 캡처
7. **AI Tools Usage** — Claude Code 사용 내역 (어떤 phase에서 어떤 도움 — 설계, 코드 리뷰, 디버깅 등)

이 spec을 근거로 다음 단계인 implementation plan을 작성한다.
