# OSAI Project 1 — Pascal-VOC Semantic Segmentation Report

**학번**: 2020314315  **이름**: 박지원  **마감**: 2026-05-05

---

## 1. 과제 개요

Pascal-VOC 21-class semantic segmentation을 수행하는 CNN 기반 모델을 설계·학습·평가했다. 평가 지표는 mIoU (ignore=255 제외) 이며, 채점은 `S = S_mIoU × S_FLOPs + S_Code + S_Report`로 mIoU와 FLOPs(@1×3×480×640)의 trade-off를 본다. 본 솔루션은 **DeepLabV3+ (ResNet-50, OS=16)** 에 COCO+VOC 2-stage 학습 + Copy-Paste augmentation + Boundary-weighted CE loss를 적용했다. 전체 파이프라인은 Colab L4 GPU에서 처음부터 학습·평가했다.

---

## 2. 모델 아키텍처

| 부분 | 모듈 | 비고 |
|---|---|---|
| Backbone | ResNet-50 (ImageNet-1K pretrained) | TorchVision `IMAGENET1K_V2` |
| Neck | ASPP (rates=[6,12,18], 256ch) | DeepLabV3 multi-scale context |
| Head | DeepLabV3+ decoder (256ch) | layer1 low-level skip (OS=4) |
| Aux | FCN aux head (layer3) | 학습 시에만 사용 |

**Output Stride 16**: layer4 stride=1 + dilation으로 OS=32→16 유지. OS=8 (≈600 GFLOPs) 대비 ¼ 비용으로 비슷한 mIoU 가능 → FLOPs 채점 유리. **DeepLabV3+ "+" decoder**는 ASPP 출력(OS=16)을 4× upsample 후 layer1 low-level feature와 concat하여 thin object boundary 정밀도 ↑ (chair leg, bicycle frame 등). **Aux head**는 backbone gradient flow 안정화로 수렴을 가속하고 `export_mode()`에서 제거되어 inference FLOPs에 영향 없다.

| 항목 | 값 |
|---|---|
| 총 parameters | **45.08M** |
| **GFLOPs @ (1, 3, 480, 640)** | **162.24** |

ONNX 그래프 traversal 기반 자체 counter로 측정 (Conv/Gemm/MatMul, MAC×2). 채점 사이트 측정값(162.46 GFLOPs)과 일치, PyTorch hook 기반 sanity check도 동일.

---

## 3. 학습 레시피

### 3.1 2-stage 학습

| Stage | 데이터셋 | Iter | Base LR | 목적 |
|---|---|---|---|---|
| **Stage 1** | COCO 2017 (VOC subset) + VOC train | 160K | 0.01 | Generalization 학습 |
| **Stage 2** | VOC train만 | 8K | 0.001 | VOC 분포 fine-tune |

COCO에서 VOC 20-class에 매핑되는 category만 사용 (~80K images), 나머지는 ignore_index=255. VOC 1.4K + COCO mapped → 데이터량 ~50배 확보. COCO mask는 instance 기반이라 노이즈가 있어 Stage 2에서 깨끗한 VOC로 정밀 적응하고, LR을 1/10로 낮춰 catastrophic forgetting을 방지했다.

### 3.2 Optimizer & Augmentation

```yaml
optimizer: SGD (momentum=0.9, weight_decay=5e-4)
backbone_lr_mult: 0.1   # backbone은 base_lr × 0.1, head는 base_lr
scheduler: poly (power=0.9), warmup 1000/500 iter
batch_size: 16, amp_dtype: fp16, ema_decay: 0.9999, grad_clip: 1.0
```

torchvision.transforms.v2 기반 image-mask joint pipeline: RandomResize(480×[0.5,2.0]) → RandomCrop 480×480 (pad=255) → HorizontalFlip 50% → ColorJitter(0.4/0.4/0.4/0.1) → RandomGrayscale 10% → GaussianBlur(σ∈[0.1,2.0]) → ImageNet Normalize → RandomErasing 25%. Mask는 NEAREST + ignore=255 fill로 라벨 보존. EMA는 학습 weight noise를 줄여 일반화에 기여하며 best mIoU는 EMA 모델 기준이다.

### 3.3 Stage 2 추가 기법

**Copy-Paste (Ghiasi et al. 2021)**: VOC SegmentationObject로부터 instance pool(~1700개)을 사전 추출, 학습 시 50% 확률로 1~3개를 다른 학습 이미지에 paste. VOC에서 chair, bottle, pottedplant 등 얇거나 작은 객체는 빈도가 낮아 학습 부족 → 강제로 다양한 context에 노출. Stage 1(scratch)에서는 데이터가 이미 풍부하고 fundamental feature 학습 중이라 추가 다양성이 수렴을 방해할 수 있어 Stage 2 한정 적용.

**Boundary-weighted CE Loss**: 3×3 max-pool 기반으로 class boundary 픽셀(이웃에 다른 class 존재)에 α=5.0 가중치를 부여한 CE. segmentation에서 가장 어려운 부분이 객체 경계인데 일반 CE는 픽셀 균등 처리라 boundary가 minority. 가중치를 두면 가늘거나 복잡한 윤곽(bicycle wheel, chair leg)의 정확도가 ↑.

### 3.4 Total Loss & TTA

```
Stage 1: L = CE(main) + 0.5·Dice(main) + 0.4·CE(aux)
Stage 2: L = CE(main) + 0.5·Dice(main) + 0.5·BoundaryCE(main) + 0.4·CE(aux)
```

모든 loss에서 ignore_index=255 픽셀은 제외. **TTA**: 5-scale [0.5, 0.75, 1.0, 1.25, 1.5] × hflip on/off = 10 forward의 softmax 확률 평균 후 argmax. Multi-scale로 작은 객체(low scale 우세)와 큰 객체(high scale 우세) 모두 커버.

---

## 4. 검증 결과 (Colab L4)

### 4.1 Validation mIoU (VOC 2012 val, 1449 images)

| 평가 | mIoU |
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
| | | | | sofa | 0.531 |
| | | | | train | 0.859 |
| | | | | tvmonitor | 0.802 |

**고성능** (IoU > 0.90): background, large/distinct object. **저성능** (IoU < 0.55): chair, bicycle, sofa — thin/occlusion-prone object (§5).

### 4.3 채점 사이트 결과 & FLOPs

| 항목 | 값 |
|---|---|
| Test set TTA mIoU (special, 1000장) | **0.785** |
| GFLOPs @ (1, 3, 480, 640) | **162.24** (= 81.12 GMACs × 2) |

Val 0.8103 → test 0.785 (-2.5%) 갭은 special test set의 분포 차이(난이도 ↑, occlusion 비율 ↑)로 추정. FLOPs는 채점 사이트 측정값 162.46과 일치.

---

## 5. Failure Case 분석

| Class | IoU | 원인 추정 |
|---|---|---|
| chair | 0.418 | 얇은 다리 + 다양한 형태, 사람과의 occlusion 빈번 |
| bicycle | 0.498 | 얇은 프레임/바퀴(sparse mask), 사람과 자주 겹침 |
| sofa | 0.531 | chair와 시각적 유사, 위에 쿠션/사람 occlusion |
| diningtable | 0.656 | 위 음식/그릇 occlusion으로 표면 mask 부정확 |
| pottedplant | 0.682 | 잎 + 화분 두 부분, boundary 복잡, instance 작음 |

이 5개 class만 평균 0.10 끌어올려도 mIoU +2.4%. 약한 class 개선이 mIoU의 dominant factor. **공통 어려움**: thin/sparse boundary (bike frame, chair leg) — Copy-Paste + Boundary loss로 일정 보완했으나 본질적 한계 존재; occlusion (사람이 의자에 앉음, 식탁 위 음식 등); visual ambiguity (chair-sofa 둘 다 좌석류). **향후 개선**: 약한 class 표적 데이터 추가, chair/sofa contrastive supervision, 작은 객체에 더 큰 weight 또는 class-balanced sampling.

---

## 6. WandB Evidence

- **Project**: `osai-p1-colab` · **GPU**: NVIDIA L4 (Overview `gpu_name`) · **Total params**: 45,076,170 (Overview `params/total`) · **Tags**: colab, voc+coco, final, copy-paste, boundary
- **Best step / val/best_mIoU**: Stage 1 = 155,000 / 0.7549, Stage 2 = 8,000 / 0.7776
- **Stage 1 run**: `colab-v2.final-s1_0428_resnet50_stage1`
- **Stage 2 run**: `colab-v2.final-s2_0428_resnet50_stage2`
- **Project URL**: <https://wandb.ai/g1nie-sungkyunkwan-university/osai-p1-colab>

*[WandB Overview 캡처 — GPU=L4, params/total=45M, run config 보이도록]*

*[Train loss curve (Stage 1 + Stage 2)]*

*[Val mIoU_ema curve]*

---

## 7. AI 도구 사용 내역

본 과제에서는 **Claude Code (Anthropic)** 를 코드 구현 및 디버깅 보조 용도로 사용했다. 모든 코드와 설계는 직접 검토·수정했으며 AI는 보조 역할에 한정했다.

---

## 8. 재현 방법

```
src/
├── data/        # VOC, COCO + Copy-Paste + transforms
├── models/      # ResNet-50, ASPP, DeepLabV3+ head, FCN aux
├── losses/      # CE, Dice, Boundary-weighted CE
├── utils/       # checkpoint, metrics, FLOPs counter
├── train.py / eval.py / eval_tta.py / infer.py
├── export_onnx.py / measure_flops.py / package_submission.py
```

```bash
# 학습 (Colab L4): Stage 1 ~10h, Stage 2 ~30분, ckpt가 Drive에 저장되어 자동 resume
!uv run python -m src.train --config src/config/colab_v2_final_s1.yaml --stage 1
!uv run python -m src.train --config src/config/colab_v2_final_s2.yaml --stage 2
# 평가 (TTA)
!uv run python -m src.eval_tta --config src/config/colab_v2_final_s2.yaml --ckpt checkpoints/best.pth
# 추론 (submit/img → submit/pred)
!uv run python -m src.infer --config src/config/colab_v2_final_s2.yaml --ckpt checkpoints/best.pth --input submit/img --output submit/pred
# FLOPs 측정
!uv run python -m src.export_onnx --ckpt checkpoints/best.pth --out model_structure.onnx
!uv run python -m src.measure_flops --onnx model_structure.onnx
```

전체 파이프라인은 `colab/colab_v2_final.ipynb` 한 노트북으로 처음부터 끝까지 재현 가능 (Drive 마운트 → 데이터 다운로드 → COCO mask cache → Stage 1 → Stage 2 → 평가 → ONNX → zip).

---

## 9. 결론

ResNet-50 + DeepLabV3+ (OS=16) 표준 segmentation 아키텍처에 2-stage 학습(COCO+VOC 160K + VOC fine-tune 8K)을 적용했고, Stage 2에 한정해 **Copy-Paste augmentation** (thin/small object 학습 보강) 과 **Boundary-weighted CE loss** (객체 경계 정확도 ↑) 를 추가했다. 설계 핵심은 augmentation/loss를 무차별 추가하지 않고 약점(thin object, boundary) 분석 → 가설 → 표적 적용이며, 두 기법을 Stage 1(scratch)이 아닌 Stage 2(fine-tune)에 한정한 것은 학습 안정성을 위한 의도적 결정이다. 최종 결과: VOC val TTA mIoU **0.8103**, special test set mIoU **0.785**, FLOPs **162.24 G**.
