# OSAI Project 1 — Pascal-VOC Semantic Segmentation Report

**학번**: 2020314315  **이름**: 박지원  **마감**: 2026-05-05

---

## 1. 과제 개요

Pascal-VOC 21-class (background + 20 object) semantic segmentation을 수행하는 CNN 기반 모델을 설계·학습·평가했다. 평가 지표는 mIoU (per-class IoU 평균, ignore=255 제외) 이며, 채점은 `S = S_mIoU × S_FLOPs + S_Code + S_Report` 공식으로 mIoU와 FLOPs(@1×3×480×640)의 trade-off를 함께 본다. 본 솔루션은 **DeepLabV3+ with ResNet-50 backbone (Output Stride 16)** 을 채택하고, COCO+VOC 2-stage 학습 + Copy-Paste augmentation + Boundary-weighted CE loss를 적용했다. 전체 파이프라인은 Colab L4 GPU에서 처음부터 학습·평가했다.

---

## 2. 모델 아키텍처

### 2.1 구조

| 부분 | 모듈 | 비고 |
|---|---|---|
| Backbone | ResNet-50 (ImageNet-1K pretrained) | TorchVision `IMAGENET1K_V2` weight |
| Neck | ASPP (rates = [6, 12, 18], 256ch) | DeepLabV3 multi-scale context |
| Head | DeepLabV3+ decoder (256ch) | low-level skip from layer1 (OS=4) |
| Auxiliary | FCN aux head (layer3 입력) | 학습 시에만 사용, 추론 시 제거 |

- **Output Stride 16**: ResNet layer4의 첫 stride를 1로 변경하고 dilation을 적용해 최종 feature를 OS=32 → OS=16으로 유지. ASPP의 multi-scale receptive field와 효율의 절충점.
- **DeepLabV3+ Decoder**: ASPP 출력 (OS=16) 을 4× upsample → layer1 low-level feature (OS=4, 256ch로 1×1 project 후) 와 concat → 3×3 conv 두 번 → 4× upsample로 원본 해상도 복원. 단순 bilinear upsample 대비 thin object boundary 정밀도 ↑.
- **Aux head**: layer3 feature에 추가 supervision을 주어 backbone gradient flow 안정 → 수렴 가속. 학습 후 `export_mode()` 호출로 제거되어 ONNX/inference 측정 FLOPs에 영향 없음.

### 2.2 모델 파라미터 및 FLOPs

| 항목 | 값 |
|---|---|
| 총 parameters | **45.08M** |
| **GFLOPs @ (1, 3, 480, 640)** | **162.24** |

ONNX 그래프 traversal 기반 자체 counter로 측정 (Conv/Gemm/MatMul 카운트, MAC × 2 = FLOP convention). 채점 사이트 측정값 (162.46 GFLOPs) 과 일치. PyTorch hook 기반 sanity check도 동일 결과.

### 2.3 설계 의도

- **CNN-only 제약 + 안정성**: 검증된 표준 segmentation 구조 (DeepLabV3+) 채택. ResNet-50은 ImageNet pretrained가 풍부하고 segmentation에서 수렴 안정성 입증된 backbone.
- **OS=16 선택**: OS=8 (4× 더 무거움, ~600 GFLOPs) 대비 약 ¼ 비용으로 비슷한 mIoU 가능. FLOPs 채점에 유리.
- **DeepLabV3+ "+" decoder**: thin object (chair leg, bicycle frame 등) 의 spatial detail 보존이 mIoU에 결정적이라 판단 → ASPP만 쓰는 DeepLabV3가 아닌 decoder 포함 DeepLabV3+ 사용.

---

## 3. 학습 레시피

### 3.1 2-stage 학습 전략

DeepLab 계열 표준 레시피를 따라 두 단계로 분리.

| Stage | 데이터셋 | Iter | Base LR | 목적 |
|---|---|---|---|---|
| **Stage 1** | COCO 2017 (VOC subset) + VOC train | 160,000 | 0.01 | 다양한 데이터로 generalization 학습 |
| **Stage 2** | VOC train만 | 8,000 | 0.001 | VOC 분포에 정밀 fine-tune |

- **COCO 활용**: VOC 20개 class에 매핑되는 COCO category만 사용 (~80K images), 나머지는 ignore_index=255 처리. VOC 1.4K + COCO mapped → 데이터량 ~50배 확보.
- **2-stage 분리 이유**: COCO mask는 instance segmentation 기반이라 노이즈가 있어, 마지막에 깨끗한 VOC로 정밀 적응. Stage 2 LR을 1/10로 낮춰 catastrophic forgetting 방지.

### 3.2 Optimizer & Scheduler

```yaml
optimizer: SGD (momentum=0.9, weight_decay=5e-4)
backbone LR mult: 0.1   # backbone은 base_lr × 0.1, head는 base_lr
scheduler: poly (power=0.9), warmup 1000 iter (Stage 1) / 500 iter (Stage 2)
batch_size: 16
amp_dtype: fp16          # T4 호환, 메모리 절감
ema_decay: 0.9999        # 모델 가중치 EMA, 평가/추론에 사용
grad_clip: 1.0           # AMP 안정성
```

- **Backbone LR mult**: pretrained backbone은 작은 LR로 미세 조정, head는 큰 LR로 처음부터 학습. DeepLab 논문 공통 관행.
- **Poly schedule**: cosine 대비 더 aggressive한 LR decay → 후반 안정 수렴.
- **EMA**: 학습 중 weight noise 줄여 일반화 ↑. Best mIoU는 EMA 모델 기준.
- **fp16 AMP**: T4 (Turing) 가 bf16 미지원 → fp16으로 통일. GradScaler로 underflow 처리.

### 3.3 Augmentation Pipeline (양 stage 공통)

torchvision.transforms.v2 기반 image-mask joint transform:

```
RandomResize (480 × [0.5, 2.0])
→ RandomCrop 480×480 (pad ignore=255)
→ RandomHorizontalFlip 50%
→ ColorJitter (brightness/contrast/saturation 0.4, hue 0.1)
→ RandomGrayscale 10%
→ GaussianBlur (kernel 5, σ ∈ [0.1, 2.0])
→ ToDtype(float32) + ImageNet Normalize
→ RandomErasing 25%
```

- Multi-scale + RandomCrop으로 다양한 객체 크기 대응.
- ColorJitter/Grayscale/Blur로 photometric robustness 확보.
- Mask는 NEAREST 보간 + ignore=255 fill로 라벨 보존.

### 3.4 Stage 2 한정 추가 기법

#### (1) Copy-Paste Augmentation (Ghiasi et al. 2021)

VOC train의 SegmentationObject로부터 instance pool (~1700개) 사전 추출 후, 학습 시 50% 확률로 1~3개를 다른 학습 이미지에 paste한다.

- **도입 의도**: VOC에서 chair, bottle, pottedplant 등 **얇거나 작은 객체**는 등장 빈도가 낮아 학습 부족. Copy-Paste로 강제로 다양한 context에 노출시켜 보완.
- **Stage 2 한정 적용**: Stage 1 (COCO+VOC scratch) 에서는 데이터가 이미 풍부하고 모델이 fundamental feature 학습 중이라 추가 다양성이 오히려 수렴을 방해할 수 있음. Stage 2 fine-tune 단계에서 이미 학습된 모델이 다양한 instance 배치를 학습하기에 적합.

#### (2) Boundary-weighted CE Loss

Class boundary 픽셀(이웃에 다른 class 존재)에 α=5.0 가중치를 부여한 CE loss.

```
boundary_mask: max-pool(target) ≠ -max-pool(-target)  (3×3 kernel)
weight = α=5.0 (boundary) or 1 (interior)
loss = Σ (CE × weight) / Σ weight
```

- **도입 의도**: segmentation에서 가장 어려운 부분이 객체 경계. 일반 CE는 픽셀 균등 처리라 boundary가 minority. 가중치를 두면 가늘거나 복잡한 윤곽 (bicycle wheel, chair leg) 정확도 ↑.
- **Stage 2 적용**: 이미 학습된 모델의 경계 부분만 정밀 보강하는 fine-tune 효과.

### 3.5 Total Loss

```
Stage 1: L = CE(main) + 0.5 · Dice(main) + 0.4 · CE(aux)
Stage 2: L = CE(main) + 0.5 · Dice(main) + 0.5 · BoundaryCE(main) + 0.4 · CE(aux)
```

- **Dice loss**: per-pixel CE의 class imbalance 완화 (region-level supervision).
- **Aux loss**: 중간 layer3 feature에 CE supervision → backbone gradient 안정.
- 모든 loss에서 ignore_index=255 픽셀은 제외.

### 3.6 Test-Time Augmentation (TTA)

추론 시 5-scale [0.5, 0.75, 1.0, 1.25, 1.5] × hflip on/off = 10 forward pass의 softmax 확률 평균 후 argmax. Multi-scale로 작은 객체 (low scale 우세)와 큰 객체 (high scale 우세) 모두 커버.

---

## 4. 검증 결과 (Colab L4)

### 4.1 Validation mIoU (VOC 2012 val, 1449 images)

| 평가 방식 | mIoU |
|---|---|
| Raw (single scale, no flip) | **0.7776** |
| **TTA** (5-scale + hflip) | **0.8103** |

TTA로 +3.27% 향상.

### 4.2 Per-class IoU (TTA)

| Class | IoU | Class | IoU | Class | IoU |
|---|---|---|---|---|---|
| background | 0.951 | bus | 0.938 | dog | 0.904 |
| aeroplane | 0.936 | car | 0.904 | horse | 0.923 |
| bicycle | 0.498 | cat | 0.934 | motorbike | 0.905 |
| bird | 0.927 | chair | 0.418 | person | 0.890 |
| boat | 0.803 | cow | 0.922 | pottedplant | 0.682 |
| bottle | 0.761 | diningtable | 0.656 | sheep | 0.872 |
|  |  |  |  | sofa | 0.531 |
|  |  |  |  | train | 0.859 |
|  |  |  |  | tvmonitor | 0.802 |

- **고성능 class** (IoU > 0.90): background, large/distinct object (aero, bird, cat, dog, horse, cow 등)
- **저성능 class** (IoU < 0.55): chair, bicycle, sofa — thin/occlusion-prone object (§6 분석)

### 4.3 채점 사이트 결과 (special test set)

| 평가 | mIoU |
|---|---|
| TTA on test set (special images, 1000장) | **0.785** |

Val 0.8103 → test 0.785의 -2.5% 갭은 special test set의 분포 차이 (난이도 ↑, occlusion 비율 ↑) 로 추정.

### 4.4 FLOPs

- ONNX 측정 (1, 3, 480, 640): **162.24 GFLOPs** (= 81.12 GMACs × 2)
- 채점 사이트 측정값과 일치 (162.46 GFLOPs)
- 측정 명령: `uv run python -m src.measure_flops --onnx model_structure.onnx`

---

## 5. Failure Case 분석

### 5.1 약한 class 5개

| Class | IoU | 원인 추정 |
|---|---|---|
| chair | 0.418 | 얇은 다리 + 다양한 형태 (의자/소파/스툴), 사람과의 occlusion 빈번 |
| bicycle | 0.498 | 얇은 프레임/바퀴 (sparse mask), 사람과 자주 겹침 |
| sofa | 0.531 | chair와 시각적 유사, 위에 쿠션/사람 occlusion |
| diningtable | 0.656 | 위에 놓인 음식/그릇 occlusion으로 표면 mask 부정확 |
| pottedplant | 0.682 | 잎 + 화분 두 부분, boundary 복잡, instance 작음 |

이 5개 class만 평균 0.10 끌어올려도 mIoU +2.4% 가능. 약한 class 개선이 mIoU의 dominant factor.

### 5.2 공통 어려움

- **Thin/sparse boundary**: bike frame, chair leg 등 가는 구조물은 Copy-Paste + Boundary loss로 일정 보완했으나 본질적 한계 존재.
- **Occlusion**: 사람이 의자에 앉음, 식탁 위 음식 등 → mask가 작고 분리되어 학습 어려움.
- **Visual ambiguity**: chair-sofa 둘 다 좌석류로 시각적 유사성 → 모델이 자주 혼동.

### 5.3 향후 개선 가능성

- 약한 class 표적 데이터 추가 학습
- chair/sofa 같이 시각 유사 class에 contrastive supervision
- 작은 객체에 더 큰 weight 또는 class-balanced sampling

---

## 6. WandB Evidence

### 6.1 Run 정보

- **Project**: `osai-p1-colab`
- **GPU**: NVIDIA L4 — Overview의 `gpu_name` 필드에 기록
- **Total params**: 45,076,170 (45.08M) — Overview의 `params/total`
- **Tags**: colab, voc+coco, final, copy-paste, boundary

### 6.2 Run summary

| 항목 | Stage 1 | Stage 2 |
|---|---|---|
| Best step | 155,000 | 8,000 |
| val/best_mIoU | 0.7549 | 0.7776 |
| GPU | NVIDIA L4 | NVIDIA L4 |

### 6.3 링크

- WandB project: <https://wandb.ai/g1nie-sungkyunkwan-university/osai-p1-colab>
- Run name 패턴: `colab-v2.final-s1_*_resnet50_stage1`, `colab-v2.final-s2_*_resnet50_stage2`

### 6.4 캡처 (PDF에 삽입 권장)

- WandB Overview page (GPU name = L4, params/total = 45M, run config)
- Train loss curve (Stage 1 + Stage 2)
- Val mIoU_ema curve

---

## 7. AI 도구 사용 내역

본 과제 진행 시 다음 AI 도구를 사용했다:

| 도구 | 용도 |
|---|---|
| **Claude Code (Anthropic)** | 코드 구현 (학습 loop, augmentation, loss, eval), 디버깅, 문서 작성 보조 |
| ChatGPT (OpenAI) | DeepLabV3+ 구조 정리, Copy-Paste 논문 요약, augmentation 파이프라인 설계 검토 |

특히 Claude Code는 다음 작업에서 유용했다:
- PyTorch 코드 modular 구조 (`src/data/`, `src/losses/`, `src/models/`) 설계
- Resumable training 구현 (checkpoint state, RNG, WandB run id 포함)
- ONNX FLOPs counter (3rd-party lib 금지 제약 하에 graph traversal 직접 구현)
- Boundary-weighted CE loss 구현 (max-pool 기반 boundary detection)

모든 코드와 설계는 직접 검토하고 수정했으며, AI는 "구현 보조" 역할에 한정.

---

## 8. 재현 방법

### 8.1 코드 구조

```
src/
├── data/        # VOC, COCO dataset + Copy-Paste augmentation + transforms
├── models/      # ResNet-50 backbone, ASPP neck, DeepLabV3+ head, FCN aux
├── losses/      # CE, Dice, Boundary-weighted CE
├── utils/       # checkpoint, metrics, FLOPs counter
├── train.py     # 2-stage training entry
├── eval.py      # raw single-scale eval
├── eval_tta.py  # multi-scale + hflip TTA eval
├── infer.py     # submit/img → submit/pred 추론 (TTA)
├── export_onnx.py
├── measure_flops.py
└── package_submission.py
```

### 8.2 학습 (Colab L4)

```bash
!uv run python -m src.train --config src/config/colab_v2_final_s1.yaml --stage 1
!uv run python -m src.train --config src/config/colab_v2_final_s2.yaml --stage 2
```

Stage 1 ~10h, Stage 2 ~30분. ckpt가 Drive에 저장되어 세션 끊겨도 자동 resume.

### 8.3 평가

```bash
# Raw mIoU
!uv run python -m src.eval --config src/config/colab_v2_final_s2.yaml --ckpt checkpoints/best.pth
# TTA mIoU
!uv run python -m src.eval_tta --config src/config/colab_v2_final_s2.yaml --ckpt checkpoints/best.pth
```

### 8.4 추론 (submit/img → submit/pred)

```bash
!uv run python -m src.infer \
    --config src/config/colab_v2_final_s2.yaml \
    --ckpt checkpoints/best.pth \
    --input submit/img --output submit/pred
```

### 8.5 FLOPs 측정

```bash
!uv run python -m src.export_onnx \
    --config src/config/colab_v2_final_s2.yaml \
    --ckpt checkpoints/best.pth \
    --out model_structure.onnx
!uv run python -m src.measure_flops --onnx model_structure.onnx
```

전체 파이프라인은 `colab/colab_v2_final.ipynb` 한 노트북으로 처음부터 끝까지 재현 가능 (Drive 마운트 → 데이터 다운로드 → COCO mask cache → Stage 1 → Stage 2 → 평가 → ONNX → zip).

---

## 9. 결론

본 과제에서는 ResNet-50 + DeepLabV3+ (OS=16) 표준 segmentation 아키텍처를 채택하고, 2-stage 학습 전략 (COCO+VOC 160K iter pretraining + VOC fine-tune 8K iter) 에 다음 두 가지 핵심 기법을 적용했다:

1. **Stage 2에 Copy-Paste augmentation** → thin/small object 학습 보강
2. **Stage 2에 Boundary-weighted CE loss** → 객체 경계 정확도 ↑

설계 핵심은 **"augmentation/loss를 무차별 추가하지 않고, 약점 (thin object, boundary) 분석 → 가설 → 표적 적용"** 순서였다. 특히 Copy-Paste와 Boundary loss를 Stage 1 (scratch 학습) 이 아닌 Stage 2 (fine-tune) 에 한정한 것은 학습 안정성 확보를 위한 의도적 결정이다.

최종 결과: VOC val TTA mIoU **0.8103**, special test set mIoU **0.785**, FLOPs **162.24 G**.
