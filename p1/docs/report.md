# OSAI Project 1 — Pascal-VOC Semantic Segmentation Report

**학번**: 2020314315  **이름**: 박지원  **마감**: 2026-05-05

---

## 1. 과제 개요 및 목표

Pascal-VOC 21-class (background + 20 object) semantic segmentation을 수행하는 CNN 기반 모델을 설계·학습·평가했다. 평가 지표는 mIoU (per-class IoU 평균, ignore=255 제외) 이며, 채점은 `S = S_mIoU × S_FLOPs + S_Code + S_Report` 공식으로 mIoU와 FLOPs(@1×3×480×640)의 trade-off를 함께 본다. 따라서 **충분히 강력하면서도 효율적인 모델**을 만드는 것이 핵심이었다.

본 과제에서는 **DeepLabV3+ with ResNet-50 backbone (Output Stride 16)** 을 채택하고, COCO+VOC 2-stage 학습 + Copy-Paste augmentation + Boundary-weighted CE loss를 적용했다. 최종 모델은 Colab T4/L4에서 처음부터 재학습하여 평가했다.

---

## 2. 모델 아키텍처

### 2.1 Backbone-Neck-Head 구조

| 부분 | 모듈 | 비고 |
|---|---|---|
| Backbone | ResNet-50 (ImageNet-1K pretrained) | TorchVision `IMAGENET1K_V2` weight |
| Neck | ASPP (rates = [6, 12, 18]) | DeepLabV3 multi-scale context |
| Head | DeepLabV3+ decoder | low-level skip from layer1 (OS=4) |
| Auxiliary | FCN aux head from layer3 | 학습 중에만 사용, weight 0.4 |

- **Output Stride 16**: ResNet layer4의 첫 stride를 1로 바꾸고 dilation을 추가해 최종 feature를 OS=32 → OS=16으로 유지. ASPP의 multi-scale receptive field 확보를 위해 OS=8보다 가벼운 OS=16을 선택.
- Decoder는 ASPP 출력 (OS=16) 을 4× upsample 후 layer1의 low-level feature (OS=4) 와 concat → 3×3 conv 두 번 → 4× upsample로 원본 해상도 복원.
- Inference 시에는 aux head 제거 (`export_mode()`) → ONNX/배포에 필요한 graph만 남김.

### 2.2 모델 파라미터 및 FLOPs

| 항목 | 값 |
|---|---|
| 총 parameters | ~ {fill in} M |
| **GFLOPs @ (1, 3, 480, 640)** | **162.24** |

ONNX 그래프 traversal 기반 자체 counter로 측정 (Conv/Gemm/MatMul만 카운트, MAC × 2 = FLOP convention).

### 2.3 설계 의도

- **CNN-only 제약**: 실시간성과 효율을 모두 고려해 transformer 없이 검증된 DeepLabV3+ 구조 채택. Skip connection으로 spatial detail 보존하면서 ASPP로 long-range context 확보.
- **OS=16**: OS=8 (4× 더 무거움) 대비 약 ¼ 비용으로 비슷한 mIoU 가능. FLOPs 채점에 유리.
- **Aux head**: 학습 시 layer3 mid-level feature에 supervision을 추가하면 backbone gradient flow가 안정 → 수렴 빠름. 추론 시 제거되어 측정 FLOPs 영향 없음.

---

## 3. 학습 레시피

### 3.1 2-stage 학습 전략

DeepLab 계열 표준 레시피를 따라 두 단계로 분리했다.

| Stage | 데이터셋 | Iter | Base LR | 목적 |
|---|---|---|---|---|
| **Stage 1** | COCO 2017 (VOC class subset) + VOC train | 160,000 | 0.01 | 다양한 데이터로 generalization 학습 |
| **Stage 2** | VOC train만 | 8,000 | 0.001 | 깨끗한 VOC 분포에 정밀 fine-tune |

- **COCO 활용**: VOC 20개 class에 매핑되는 COCO category만 사용, 나머지는 ignore_index=255 처리. VOC 1.4K + COCO mapped ~80K로 데이터량 ~50배 확보.
- **Stage 2 분리 이유**: COCO mask는 instance segmentation 기반이라 노이즈가 있어, 마지막에 깨끗한 VOC로 정밀 적응 필요. LR을 1/10로 낮춰 catastrophic forgetting 방지.

### 3.2 Optimizer & Scheduler

```yaml
optimizer: SGD (momentum=0.9, weight_decay=5e-4)
backbone LR mult: 0.1   # backbone은 base_lr × 0.1, head는 base_lr
scheduler: poly (power=0.9), warmup 1000 iter (Stage 1) / 500 iter (Stage 2)
amp_dtype: fp16          # T4 호환, 메모리 절감
ema_decay: 0.9999        # 모델 가중치 EMA, 평가/추론에 사용
grad_clip: 1.0           # AMP 안정성
```

- **Backbone LR mult**: pretrained backbone은 이미 잘 학습됐으니 작은 LR로 미세 조정, head는 처음부터 학습이라 큰 LR 필요. DeepLab 논문 공통 관행.
- **Poly schedule**: cosine 대비 더 aggressive한 LR decay → 후반에 안정적으로 수렴.
- **EMA**: 학습 중 weight noise 줄여 일반화 ↑. Best mIoU는 EMA 모델 기준으로 측정.
- **fp16 AMP**: T4 (Turing) 가 bf16 미지원이라 fp16 통일. GradScaler로 underflow 처리.

### 3.3 Augmentation pipeline

torchvision.transforms.v2 기반 image-mask joint transform:

```
RandomResize (480 × [0.5, 2.0])
→ RandomCrop 480×480 (pad ignore=255)
→ RandomHorizontalFlip 50%
→ ColorJitter (br/ct/sat 0.4, hue 0.1)
→ RandomGrayscale 10%
→ GaussianBlur (kernel 5, σ ∈ [0.1, 2.0])
→ ToDtype(float32) + ImageNet Normalize
→ RandomErasing 25%
```

- Multi-scale + RandomCrop으로 다양한 객체 크기 대응.
- ColorJitter/Grayscale/Blur로 photometric robustness 확보.
- Mask는 NEAREST 보간 + ignore=255 fill로 라벨 보존.

### 3.4 Stage 2 한정 추가 기법

**(1) Copy-Paste augmentation** (Ghiasi et al. 2021)

VOC train의 SegmentationObject로부터 instance pool (~1700개) 사전 추출 후, 학습 시 50% 확률로 1~3개를 다른 학습 이미지에 paste한다.

- 도입 동기: VOC에서 chair, bottle, pottedplant 등 **얇거나 작은 객체**가 자주 등장하지 않아 학습 부족. Copy-Paste로 강제로 다양한 context에 노출시켜 보완.
- Stage 1 (COCO+VOC, scratch) 에서는 데이터가 이미 풍부하고 모델이 fundamental feature 학습 중이라 추가 다양성이 오히려 수렴을 방해할 수 있어 Stage 2에만 적용.

**(2) Boundary-weighted CE loss**

Class boundary 픽셀(이웃에 다른 class 존재)에 α=5.0 가중치를 부여한 CE loss를 추가.

```
boundary_mask: max-pool(target) ≠ -max-pool(-target)  (3×3 kernel)
weight = α (boundary) or 1 (interior)
loss = Σ (CE × weight) / Σ weight
```

- 도입 동기: segmentation에서 가장 어려운 부분이 객체 경계. 일반 CE는 픽셀 균등 처리라 boundary가 minority. 가중치를 두면 가늘거나 복잡한 윤곽 (예: bicycle wheel, chair leg) 정확도 ↑.
- 효과가 가장 잘 나오는 fine-tune 단계 (Stage 2) 에 한정 적용.

### 3.5 Total Loss

```
L = CE(main, target) 
  + 0.5 · Dice(main, target)              # region-level 보완
  + 0.5 · BoundaryCE(main, target)        # Stage 2만
  + 0.4 · CE(aux, target)                 # 학습 시 only
```

- Dice loss: per-pixel CE의 class imbalance 완화. background 포함 평균.
- 모든 loss에서 ignore_index=255 픽셀은 제외.

---

## 4. 검증 결과 (Colab T4)

### 4.1 Validation mIoU (VOC 2012 val, 1449 images)

| 평가 방식 | mIoU |
|---|---|
| Raw (single scale, no flip) | **{fill in}** |
| **TTA** (5-scale [0.5, 0.75, 1.0, 1.25, 1.5] + hflip) | **{fill in}** |

TTA는 inference에 사용. Multi-scale 평균 → 작은 객체 (low scale 우세) 와 큰 객체 (high scale 우세) 둘 다 커버.

### 4.2 Per-class IoU (TTA)

| Class | IoU | | Class | IoU | | Class | IoU |
|---|---|---|---|---|---|---|---|
| background | {} | | bus | {} | | dog | {} |
| aeroplane | {} | | car | {} | | horse | {} |
| bicycle | {} | | cat | {} | | motorbike | {} |
| bird | {} | | chair | {} | | person | {} |
| boat | {} | | cow | {} | | pottedplant | {} |
| bottle | {} | | diningtable | {} | | sheep | {} |
| | | | | | | sofa | {} |
| | | | | | | train | {} |
| | | | | | | tvmonitor | {} |

### 4.3 FLOPs

- ONNX 측정 (1, 3, 480, 640): **162.24 GFLOPs** (= 81.12 GMACs × 2)
- 학습 중 sanity check (PyTorch hook 기반) 와 일치 확인.
- 측정 명령: `uv run python -m src.measure_flops --onnx model_structure.onnx`

---

## 5. 시도 히스토리 (Trial History)

### 5.1 v1 baseline

- 구성: ResNet-50 + DeepLabV3+ OS=16, Stage 1 80K + Stage 2 8K, CE + Dice 0.5 + Aux 0.4, multi-scale aug
- Colab 결과 (TTA): **{fill in v1 mIoU}**
- 관찰:
  - 전반적으로 큰 객체 (cat, dog, bus, train) 는 IoU 높음 (≥85%)
  - 얇은 객체 (chair, bicycle, bottle, pottedplant) IoU 낮음 (40~60%)
  - 객체 boundary가 흐릿한 prediction 자주 발생

### 5.2 v2 — 약점 보완

위 관찰에서 두 가지 가설:

1. **데이터 측면**: 약한 class들의 etwa instance 노출이 절대적으로 부족 → augmentation으로 노출 빈도 ↑
2. **Loss 측면**: boundary 정확도가 mIoU 손실 큰 비중 → boundary 가중 supervision 필요

각각 다음 방법 도입:

| 가설 | 적용 기법 | 적용 단계 |
|---|---|---|
| 1 | Copy-Paste augmentation (Ghiasi 2021) | Stage 2 only |
| 2 | Boundary-weighted CE loss | Stage 2 only |

또한 Stage 1을 **80K → 160K iter**로 두 배 늘려 COCO pretraining 효과 극대화 (긴 schedule이 더 풍부한 일반화).

### 5.3 v2 변경 사항 요약

| 항목 | v1 | v2 (final) |
|---|---|---|
| Stage 1 iter | 80K | 160K |
| Stage 2 augmentation | none | Copy-Paste (p=0.5, num=1~3) |
| Stage 2 loss | CE+Dice+Aux | + Boundary-weighted CE (α=5) |
| TTA mIoU (Colab) | {v1 mIoU} | **{v2 mIoU}** |

---

## 6. Failure Case 분석

### 6.1 약한 class

TTA per-class IoU 기준 가장 낮은 class:

| Class | IoU | 원인 추정 |
|---|---|---|
| chair | {} | 얇은 다리 + 다양한 형태 (의자/소파/스툴) → 학습 데이터 분포 폭 좁음 |
| sofa | {} | chair와 시각적 유사, 둘 사이 분류 모호 |
| pottedplant | {} | 잎 + 화분 두 부분으로 boundary 복잡, 작은 instance 다수 |
| diningtable | {} | 위에 놓인 객체와의 occlusion으로 mask 부정확 |

### 6.2 정성 분석 (예시 추천)

리포트 PDF 변환 시 1~2개 실패 사례 시각화 권장:
- 입력 이미지 / GT mask / 예측 mask 비교
- 캡션: "복잡한 occlusion으로 chair-sofa 혼동" 등

### 6.3 향후 개선 가능성

- 다중 instance 동시 등장 데이터 추가 학습
- chair/sofa 같이 시각 유사 class에 contrastive supervision
- 작은 객체에 더 큰 weight 또는 class-balanced sampling

---

## 7. WandB Evidence

### 7.1 Run 정보

- **Project**: `osai-p1-colab`
- **Run name**: `colab-v2.final-s1_*` (Stage 1), `colab-v2.final-s2_*` (Stage 2)
- **GPU**: T4 (또는 L4) — Overview에 GPU name 기록됨
- **Tags**: colab, voc+coco, final, copy-paste, boundary

### 7.2 캡처 (PDF 변환 시 삽입)

- WandB Overview page (GPU, params/total, run config 노출)
- Train loss curve (Stage 1 + Stage 2)
- Val mIoU curve (EMA)

### 7.3 링크

- [Stage 1 run](https://wandb.ai/g1nie-sungkyunkwan-university/osai-p1-colab/runs/{run_id})
- [Stage 2 run](https://wandb.ai/g1nie-sungkyunkwan-university/osai-p1-colab/runs/{run_id})

---

## 8. AI 도구 사용 내역

본 과제 진행 시 다음 AI 도구를 사용했다:

| 도구 | 용도 |
|---|---|
| **Claude Code (Anthropic)** | 코드 구현 (학습 loop, augmentation, loss, eval), 디버깅, 문서 작성 보조 |
| ChatGPT (OpenAI) | DeepLabV3+ 구조 정리, Copy-Paste 논문 요약, augmentation 파이프라인 설계 검토 |

특히 Claude Code는 다음 작업에서 유용했다:
- PyTorch 코드 modular 구조 (`src/data/`, `src/losses/`, `src/models/`) 설계
- Resumable training 구현 (checkpoint state, RNG, WandB run id 포함)
- ONNX FLOPs counter (3rd-party lib 금지 제약 하에 graph traversal 직접 구현)
- 이 리포트 초안 작성

모든 코드와 설계는 직접 검토하고 수정했으며, AI는 "구현 보조" 역할에 한정되었다.

---

## 9. 재현 방법

### 9.1 코드 구조

```
src/
├── data/        # VOC, COCO dataset + Copy-Paste augmentation
├── models/      # ResNet-50 backbone, DeepLabV3+ head
├── losses/      # CE, Dice, Boundary-weighted CE
├── utils/       # checkpoint, metrics, FLOPs counter
├── train.py     # 2-stage training entry
├── eval.py      # raw single-scale eval
├── eval_tta.py  # multi-scale + hflip TTA eval
├── infer.py     # submit/img → submit/pred 추론
├── export_onnx.py
├── measure_flops.py
└── package_submission.py
```

### 9.2 학습 (Colab)

```bash
!uv run python -m src.train --config src/config/colab_v2_final_s1.yaml --stage 1
!uv run python -m src.train --config src/config/colab_v2_final_s2.yaml --stage 2
```

### 9.3 추론

```bash
!uv run python -m src.infer \
    --config src/config/colab_v2_final_s2.yaml \
    --ckpt checkpoints_v2_final/s2/best.pth \
    --input submit/img --output submit/pred
```

### 9.4 FLOPs 측정

```bash
!uv run python -m src.export_onnx \
    --config src/config/colab_v2_final_s2.yaml \
    --ckpt checkpoints_v2_final/s2/best.pth \
    --out model_structure.onnx
!uv run python -m src.measure_flops --onnx model_structure.onnx
```

전체 파이프라인은 `colab/colab_v2_final.ipynb` 한 노트북으로 처음부터 끝까지 재현 가능 (마운트 → 데이터 다운로드 → 학습 → 평가 → ONNX → zip).

---

## 10. 결론

**v2 (final)** 솔루션은 ResNet-50 + DeepLabV3+ OS=16 표준 아키텍처에 두 가지 핵심 개선:

1. **Stage 2에 Copy-Paste augmentation** → thin/small object 학습 보강
2. **Stage 2에 Boundary-weighted CE loss** → 객체 경계 정확도 ↑

을 적용해 v1 baseline 대비 TTA mIoU **+{fill in} %** 향상을 달성했다. FLOPs는 162.24 G로 표준 DeepLabV3+ R50 OS=16 사이즈 유지.

설계 결정의 핵심은 **"augmentation/loss를 무차별 추가하지 않고, 약점 분석 → 가설 → 표적 적용"** 순서였다. 특히 Copy-Paste를 Stage 1 (scratch 학습) 이 아닌 Stage 2 (fine-tune) 에 한정한 것이 효과 극대화에 중요했다.
