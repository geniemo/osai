# Project 1 — Improvement Ablation Design Spec

> v1 baseline (75.82% raw / 79.47% TTA) 위에서 mIoU 추가 개선을 위한 ablation 전략. SBD는 PDF 정책 모호로 미사용 가정. `improve` 브랜치에서 진행.

---

## 1. Goal & Scope

### 1.1 목표

- **타겟**: TTA mIoU 79.47% → **~82%** (+2~3%)
- **수단**: 4 ablation 균형 (a/b/c/d 중 b 균형 옵션 선택)
- **이유**: paper baseline (with SBD: 80-82%) 매칭 + 4/29 threshold strict 시 5점 안전마진

### 1.2 시간 budget

- 마감 5/5. 차감: Colab 재현 (~1.5일) + 리포트 (1.5일) + 안전마진 (0.5일) = 3.5일.
- ablation 가용: ~5-6일.
- 외부 이벤트: 4/29 threshold 발표 (mid-ablation 시점).

### 1.3 Scope에 포함 안 됨

- Architecture 변경 (ResNet-50 유지, OS=16 유지, ResNet-101/ConvNeXt 시도 X — FLOPs 위험)
- CRF post-processing (PyDenseCRF 정책 회색지대로 skip)
- AutoAugment / RandAugment (효과 불확실)
- SBD 사용 (PDF 정책 미명시 → 4/29 답변 받으면 별도 고려)
- COCO를 Copy-Paste source로 사용 (구현 복잡도 vs 효과 trade-off에서 우선순위 낮음)

---

## 2. Locked Decisions (확정 사항)

| 항목 | 결정 | 근거 |
|---|---|---|
| Branch 전략 | `improve` 브랜치, v1을 `v1-baseline-75.82` git tag로 고정 | rollback 가능, baseline 보존 |
| Ckpt 백업 | `p1/checkpoints_v1_baseline/` (2.2GB, gitignored) | 학습 중 overwrite 방지 |
| 구조 | **Hybrid** (Stage 1 한 번 + 3 isolated ablation + 최종 통합) | rigorous attribution + 시간 효율 |
| Ablation 4개 | Stage 1 160K (v2.0), Copy-Paste (v2.A), Class-balanced sampling (v2.B), Lovász loss (v2.C) | 가능성 ★★★★ 이상만 선별 |
| 결정 기준 | TTA mIoU 기준 Δ vs v2.0 | 실제 채점 mIoU에 더 가까움 |
| WandB | 같은 project (`osai-p1-local`), tags로 구분 | 비교 표 자동 생성 |

---

## 3. Overall Structure (Hybrid)

```
v1 baseline (main, 75.82% raw / 79.47% TTA) — 보존
  ↓
[improve 브랜치]
  ↓
🔵 v2.0 = "improved baseline" (1회 학습)
   변경: stage1_iters 80K → 160K (다른 모든 것 v1 동일)
   학습: Stage 1 160K + Stage 2 8K
   GPU 시간: ~11.5h
   ckpt: ./checkpoints/v2.0/
   ↓
   v2.0의 Stage 1 best ckpt를 starting point로
   3개 ablation 독립 측정 (각 Stage 2 8K iter):
   ↓
   ├── 🟢 v2.A = v2.0 + Copy-Paste (Stage 2)
   ├── 🟢 v2.B = v2.0 + Class-balanced sampling (Stage 2)
   └── 🟢 v2.C = v2.0 + Lovász loss (Stage 2)
   ↓
   각 ablation Δ 측정 → positive ones 선별
   ↓
🟣 v2.final = v2.0 + (selected combination)
   학습: Stage 1 (160K) + Stage 2 (8K) 통째로 재실행
   GPU 시간: ~11.5h
   ckpt: ./checkpoints/v2.final/
```

### 3.1 Hybrid 구조의 장점

- 각 ablation 효과 **독립 측정** (모두 v2.0 base에서 시작 → 동일 출발점)
- Stage 1 비싼 학습 한 번만 (v2.0)
- 리포트 ablation 표가 깔끔
- 마지막 v2.final이 fail해도 v2.A/B/C 중 best로 fallback 가능

### 3.2 적용 범위 (Stage 1 vs Stage 2)

| Ablation | Stage 1 적용 | Stage 2 적용 | 이유 |
|---|---|---|---|
| Copy-Paste | ✅ | ✅ | data augmentation |
| Class-balanced sampling | ❌ | ✅ | Stage 1은 COCO 95K 다양성 충분, Stage 2 VOC만이 imbalanced |
| Lovász loss | ✅ | ✅ | loss 변경 |

→ ablation 측정 시점(Stage 2)에는 모두 적용. v2.final 시점에는 Class-balanced만 Stage 2 한정, 나머지는 두 stage 모두.

---

## 4. v2.0 — Stage 1 80K → 160K

### 4.1 변경

`src/config/v2_0_stage1_160k.yaml`:
- `training.stage1_iters`: 80000 → **160000**
- 다른 모든 항목 default.yaml 동일 (data, model, loss, optimizer 등)
- WandB tags: `[desktop, voc+coco, ablation, v2.0, stage1-160k]`
- ckpt 경로: `./checkpoints/v2.0/`

### 4.2 LR Schedule 영향

```
old: lr(t) = 0.01 × (1 - t/80000)^0.9
new: lr(t) = 0.01 × (1 - t/160000)^0.9
```

→ 같은 iter에서 new가 더 큰 lr (예: t=40K에서 old=0.0054, new=0.0078). 2x 길게 학습 + slower decay.

### 4.3 예상 효과

| 시나리오 | Stage 1 best | Stage 2 best | TTA |
|---|---|---|---|
| 비관 (Stage 1 underfit 아니었음) | 72.5% | 76.0% | 79.7% |
| 현실 | 73.5% | 76.7% | 80.5% |
| 낙관 | 75% | 78% | 82% |

### 4.4 GPU 시간

- Stage 1: 160K iter ÷ 4 it/s ≈ **11.1h**
- Stage 2: 8K iter ÷ 4 it/s ≈ **0.5h**
- **합계: ~11.5h**

### 4.5 리스크

- Diminishing returns 가능 — Stage 1 80K에서 이미 71.98%, 추가 80K로 +1~3%만일 수 있음
- Overfitting 위험 낮음 (COCO 95K 충분히 다양)

---

## 5. v2.A — Copy-Paste Augmentation

### 5.1 알고리즘 (Ghiasi 2021)

매 iter마다 (확률 p=0.5로):
1. 다른 VOC train 이미지에서 random 1-3 instance 선택
2. 그 instance를 현재 이미지 위에 paste (mask도 같이 update)
3. 기존 augmentation pipeline 적용

**Source pool**: VOC train 1,464장의 SegmentationObject 활용 (~5,000-7,000 instance 풀, 메모리 ~250-350MB)

### 5.2 Instance 추출 (학습 시작 시 1회)

```python
def build_instance_pool(voc_root, train_ids, min_area=64*64, max_area_ratio=0.25):
    pool = []  # list of (image_patch, instance_mask, class_id)
    for img_id in train_ids:
        seg_obj = np.array(Image.open(f"{voc_root}/SegmentationObject/{img_id}.png"))
        seg_cls = np.array(Image.open(f"{voc_root}/SegmentationClass/{img_id}.png"))
        img = np.array(Image.open(f"{voc_root}/JPEGImages/{img_id}.jpg"))
        H, W = seg_obj.shape
        for inst_id in np.unique(seg_obj):
            if inst_id in (0, 255):
                continue
            inst_mask = (seg_obj == inst_id)
            area = inst_mask.sum()
            if area < min_area or area > H * W * max_area_ratio:
                continue
            cls = int(seg_cls[inst_mask][0])
            if cls in (0, 255):
                continue
            ys, xs = np.where(inst_mask)
            y0, y1 = ys.min(), ys.max() + 1
            x0, x1 = xs.min(), xs.max() + 1
            patch_img = img[y0:y1, x0:x1].copy()
            patch_mask = inst_mask[y0:y1, x0:x1].copy()
            pool.append((patch_img, patch_mask, cls))
    return pool
```

Filter:
- min_area: 64×64 = 4096 px (너무 작은 instance 제외)
- max_area_ratio: 0.25 (target 이미지의 25% 초과 paste 방지)

### 5.3 CopyPasteDataset Wrapper

```python
class CopyPasteDataset(Dataset):
    def __init__(self, base, instance_pool, p=0.5, num_paste=(1, 3)):
        self.base = base
        self.pool = instance_pool
        self.p = p
        self.num_paste_range = num_paste

    def __getitem__(self, idx):
        img_pil, mask_pil = self.base.get_raw(idx)  # transform 전 raw PIL
        if random.random() < self.p:
            img_arr = np.array(img_pil)
            mask_arr = np.array(mask_pil)
            n = random.randint(*self.num_paste_range)
            for _ in range(n):
                patch_img, patch_mask, cls_id = random.choice(self.pool)
                img_arr, mask_arr = paste_instance(
                    img_arr, mask_arr, patch_img, patch_mask, cls_id
                )
            img_pil = Image.fromarray(img_arr)
            mask_pil = Image.fromarray(mask_arr)
        return self.base.apply_transform(img_pil, mask_pil)
```

→ VOCSegDataset에 `get_raw()` + `apply_transform()` 메서드 분리 필요.

### 5.4 Paste Logic

```python
def paste_instance(target_img, target_mask, patch_img, patch_mask, cls_id):
    Ht, Wt = target_img.shape[:2]
    Hp, Wp = patch_img.shape[:2]
    # patch가 target보다 크면 random scale down
    if Hp > Ht * 0.5 or Wp > Wt * 0.5:
        scale = min(Ht * 0.5 / Hp, Wt * 0.5 / Wp)
        new_H, new_W = int(Hp * scale), int(Wp * scale)
        patch_img = cv2.resize(patch_img, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
        patch_mask = cv2.resize(
            patch_mask.astype(np.uint8), (new_W, new_H), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        Hp, Wp = new_H, new_W
    # random hflip
    if random.random() < 0.5:
        patch_img = patch_img[:, ::-1].copy()
        patch_mask = patch_mask[:, ::-1].copy()
    # random center position
    cy = random.randint(Hp // 2, Ht - Hp // 2)
    cx = random.randint(Wp // 2, Wt - Wp // 2)
    y0, y1 = cy - Hp // 2, cy - Hp // 2 + Hp
    x0, x1 = cx - Wp // 2, cx - Wp // 2 + Wp
    region = target_img[y0:y1, x0:x1]
    region[patch_mask] = patch_img[patch_mask]
    target_img[y0:y1, x0:x1] = region
    target_mask[y0:y1, x0:x1][patch_mask] = cls_id
    return target_img, target_mask
```

핵심:
- Hard paste (no blending) — paper 검증
- 새 instance mask가 기존 라벨 위에 덮어씀 (overlap 시 paste 우선)
- 최대 크기 target의 50% (너무 큰 paste 방지)

### 5.5 Hyperparameters

| 항목 | 값 |
|---|---|
| `p` (apply prob) | 0.5 |
| `num_paste` | 1-3 random |
| `min_area` | 64×64 = 4096 px |
| `max_area_ratio` | 0.25 |
| `hflip prob` | 0.5 |
| Source | VOC train only |

### 5.6 yaml 옵션

```yaml
data:
  copy_paste:
    enabled: true
    p: 0.5
    num_paste: [1, 3]
```

### 5.7 예상 효과

- 전체 mIoU: +1~3% (paper 보고)
- Weak class (chair, bicycle, sofa): +3~7% 가능
- 가장 큰 단일 ablation 효과

### 5.8 리스크

- 초기 instance pool 구축 ~10-30초 (1회만)
- DataLoader worker가 pool 공유 → fork 후 메모리 ~2.8GB 추가 (5070 Ti CPU RAM 32GB+에 OK)
- Synthetic image (paste된 거)가 실제 분포와 다를 가능성 — paper들이 효과 검증
- 구현 버그 (mask alignment, position bound)

### 5.9 Test 전략

1. `build_instance_pool` 단위 테스트: VOC train 일부에서 추출, instance 개수 + bbox 검증
2. `paste_instance` 단위 테스트: toy image로 paste 결과 검증
3. `CopyPasteDataset` smoke test: 10 sample 추출, mask 값 [0..20]+255 범위 검증
4. **시각화**: `viz.py`로 paste 전/후 비교 (몇 개 샘플 PNG 직접 확인)

---

## 6. v2.B — Class-Balanced Sampling

### 6.1 목표

VOC train 1,464장 class imbalanced → weak class 포함 이미지 더 자주 sample.

### 6.2 알고리즘

`WeightedRandomSampler` (PyTorch built-in) image-level oversampling.

#### Step 1: Class pixel count (1회)

```python
def compute_voc_class_counts(voc_root, train_ids):
    counts = np.zeros(21, dtype=np.int64)
    for img_id in train_ids:
        mask = np.array(Image.open(f"{voc_root}/SegmentationClass/{img_id}.png"))
        for c in range(21):
            counts[c] += (mask == c).sum()
    return counts
```

#### Step 2: Class weight (inverse sqrt frequency)

```python
class_weights = 1.0 / torch.sqrt(class_pixel_counts.float() + 1)
class_weights = class_weights / class_weights.sum()
```

→ inverse (1/x)는 너무 극단적 (bg 거의 0, bicycle 거대), inverse sqrt가 적당한 boost.

예상 weight:
- background ≈ 0.01 (낮음)
- person ≈ 0.05
- bicycle, chair ≈ 0.18~0.20 (높음)

#### Step 3: Per-image weight

```python
def compute_image_weights(train_ids, voc_root, class_weights):
    image_weights = []
    for img_id in train_ids:
        mask = np.array(Image.open(f"{voc_root}/SegmentationClass/{img_id}.png"))
        present = np.unique(mask)
        present = present[(present != 255) & (present < 21)]
        w = class_weights[present].sum().item()
        image_weights.append(w)
    return torch.tensor(image_weights)
```

#### Step 4: Sampler 적용

```python
sampler = WeightedRandomSampler(
    weights=image_weights,
    num_samples=len(image_weights),
    replacement=True,
)
train_loader = DataLoader(train_ds, sampler=sampler, ...)  # shuffle=False
```

### 6.3 적용 범위

**Stage 2 (VOC train only)에만 적용.** Stage 1은 COCO 95K로 다양성 확보 → 의미 X.

### 6.4 Implementation

```
src/data/sampler.py (새 파일, ~50줄)
  compute_voc_class_counts()
  compute_image_weights()
  build_balanced_sampler()
```

builder.py에 `data.class_balanced` flag 추가.

### 6.5 예상 효과

- 전체 mIoU: +0.5~1.5% TTA
- chair (40%): +4~8%
- bicycle (47%): +4~7%
- sofa (50%): +4~7%
- person (88%): -0~1% (약간 trade-off)

### 6.6 리스크

- WeightedRandomSampler가 같은 이미지 반복 → augmentation diversity 의존 (우리 강한 aug로 mitigated)
- DataLoader iter 수 동일하지만 unique sample 적음 → 학습 일관성 OK
- Stage 1 mixed 효과 없음 → Stage 2 한정

---

## 7. v2.C — Lovász-Softmax Loss

### 7.1 배경

Lovász-Softmax (Berman et al. 2018) = mIoU의 differentiable smooth surrogate.
- CE: pixel-wise log-likelihood 최적화 (mIoU와 다름)
- Lovász: IoU 자체를 직접 최적화

### 7.2 알고리즘 (간단히)

```
각 class c에 대해:
  errors = |target_c - prob_c|  (per pixel)
  errors descending 정렬
  Lovász extension gradient 적용
  dot product (errors_sorted, lovasz_grad(target_sorted))
전체 class 평균
```

### 7.3 구현

```python
# src/losses/lovasz.py (~80줄)

def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probs, labels, classes="present", ignore_index=255):
    valid = labels != ignore_index
    probs = probs[valid]
    labels = labels[valid]
    if probs.numel() == 0:
        return probs.sum() * 0.0
    losses = []
    class_set = (
        torch.unique(labels).tolist() if classes == "present" else range(probs.size(1))
    )
    for c in class_set:
        if c >= probs.size(1):
            continue
        fg = (labels == c).float()
        if classes == "present" and fg.sum() == 0:
            continue
        errors = (fg - probs[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    return torch.stack(losses).mean() if losses else probs.sum() * 0.0


def lovasz_softmax(logits, labels, ignore_index=255):
    probs = F.softmax(logits, dim=1)
    N, C, H, W = probs.shape
    probs = probs.permute(0, 2, 3, 1).reshape(-1, C)
    labels = labels.reshape(-1)
    return lovasz_softmax_flat(probs, labels, ignore_index=ignore_index)
```

### 7.4 기존 Loss와 결합 (Option A: Dice → Lovász 교체)

현재 v1: `L = CE + 0.5 × Dice + 0.4 × Aux_CE`

**v2.C**: `L = CE + 0.5 × Lovász + 0.4 × Aux_CE`

**왜 교체?**
- Dice와 Lovász 모두 region-based (IoU 최적화)
- 둘 다 쓰면 redundant + 균형 어려움
- Lovász가 IoU 직접 최적화라 mIoU 채점에 더 직접

**weight 0.5**: Dice와 동일한 weight로 controlled comparison.

**Aux head**: CE 그대로 (Lovász는 sorting으로 비싸서 main만).

### 7.5 yaml

```yaml
loss:
  ignore_index: 255
  dice_weight: 0.0       # disabled
  lovasz_weight: 0.5     # added
  aux_weight: 0.4
```

### 7.6 SegLoss 수정

```python
class SegLoss(nn.Module):
    def __init__(self, ..., dice_weight=0.5, lovasz_weight=0.0, aux_weight=0.4):
        ...
        self.lovasz_weight = lovasz_weight

    def forward(self, main, aux, target):
        loss = self._ce_safe(main, target)
        if self.dice_weight > 0:
            loss = loss + self.dice_weight * self.dice(main, target)
        if self.lovasz_weight > 0:
            from src.losses.lovasz import lovasz_softmax
            loss = loss + self.lovasz_weight * lovasz_softmax(main, target)
        if aux is not None and self.aux_weight > 0:
            loss = loss + self.aux_weight * self._ce_safe(aux, target)
        return loss
```

### 7.7 예상 효과

- Lovász가 Dice보다 일반적으로 +0.3~1% 더 (paper)
- 우리 baseline 강해서 marginal gain만 가능

### 7.8 리스크

- 계산 비용 ~2x Dice (sort + cumsum) → 학습 시간 6h → 7-8h
- 수치 안정성: cumsum overflow 가능 → float32 유지
- 구현 버그 (sorted index, perm tracking) → unit test 필수
- 효과 약할 가능성 (이미 Dice로 region 최적화 중)

### 7.9 Tests

1. `lovasz_grad` 단위 테스트: paper formula 일치
2. Perfect prediction → loss ≈ 0
3. Random prediction → loss in [0, 1]
4. Ignore label 검증
5. Comparison with Dice (대략 같은 magnitude)

---

## 8. v2.final — Combination

### 8.1 Decision Tree

각 ablation 측정 후:

```
v2.A, v2.B, v2.C 각각 v2.0 대비 Δ_TTA 계산
  ↓
Case 1: 모두 positive (Δ > 0)
  → v2.final = v2.0 + A + B + C
  → 예상: Δ_A + Δ_B + Δ_C의 70-80%

Case 2: 2개 positive
  → v2.final = v2.0 + 2 positive ones

Case 3: 1개 positive
  → v2.final = v2.0 + 1개

Case 4: 모두 0 또는 negative
  → v2.final = v2.0 그대로 (또는 v1 회귀)
  → negative result 리포트 기록
```

### 8.2 결정 기준

- **Δ > +1% TTA**: 확실히 keep
- **Δ ∈ [-0.5, +1] TTA**: 통계적 noise 가능성, drop
- **Δ < -0.5% TTA**: drop + 분석 (왜 hurt했는지 리포트)

### 8.3 학습 전략

**전체 Stage 1 (160K) + Stage 2 (8K) 재학습**.

이유:
- Copy-Paste, Lovász는 Stage 1에도 효과
- v2.0 ckpt에서 Stage 2만 추가 시 Stage 1의 Copy-Paste/Lovász 효과 누락
- 11.5h 한 번 더 투자할 가치

**대안**: v2.0 결과가 v1 대비 marginal (Δ < +1%)이라면 Stage 1 80K (=v1)로 회귀 → 6h 학습.

### 8.4 v2.final yaml 예시 (Case 1: 모두 positive)

```yaml
# v2_final.yaml
training:
  stage1_iters: 160000
  stage2_iters: 8000

data:
  copy_paste:
    enabled: true
    p: 0.5
    num_paste: [1, 3]
  class_balanced: true

loss:
  ignore_index: 255
  dice_weight: 0.0
  lovasz_weight: 0.5
  aux_weight: 0.4

wandb:
  run_name_prefix: v2.final
  tags: [desktop, voc+coco, ablation, v2.final, combined]

paths:
  ckpt_dir: ./checkpoints/v2.final
  ...
```

### 8.5 v2.final 필수 아님

- 만약 v2.A 단독으로 충분히 좋으면 (예: +2% TTA → 81.5%) v2.A를 final로 직접 사용
- 리포트에 "tried combining but A alone was best" 결정 기록

---

## 9. Reporting

### 9.1 비교 결과 표

| Variant | Description | Δ vs v1 | Δ vs prev | mIoU raw | mIoU TTA |
|---|---|---|---|---|---|
| v1 | baseline (Stage 1 80K, CE+Dice+Aux) | 0 | — | 75.82 | 79.47 |
| v2.0 | + Stage 1 160K | +Δ₀ | +Δ₀ | ? | ? |
| v2.A | v2.0 + Copy-Paste (S2) | +Δ₀+ΔA | +ΔA | ? | ? |
| v2.B | v2.0 + Class-balanced (S2) | +Δ₀+ΔB | +ΔB | ? | ? |
| v2.C | v2.0 + Lovász (S2) | +Δ₀+ΔC | +ΔC | ? | ? |
| v2.final | v2.0 + selected combo (full S1+S2) | +Δfinal | — | ? | ? |

→ 리포트 § Ablation Study에 그대로.

### 9.2 Per-class IoU 비교 (특히 weak classes)

```
Class            v1     v2.0   v2.A   v2.B   v2.C   v2.final
chair (40)       40     ?      ?      ?      ?      ?
bicycle (47)     47     ?      ?      ?      ?      ?
sofa (50)        50     ?      ?      ?      ?      ?
pottedplant (56) 56     ?      ?      ?      ?      ?
```

→ 어느 ablation이 어느 class에 효과인지 직접 확인. 리포트 §failure case analysis 활용.

### 9.3 WandB Tag 조직

| Run | Tags |
|---|---|
| v1 (Stage 1) | `dev`, `voc+coco` |
| v1 (Stage 2) | `dev`, `voc+coco` |
| v2.0 | `ablation`, `v2.0`, `stage1-160k` |
| v2.A | `ablation`, `v2.A`, `copy-paste` |
| v2.B | `ablation`, `v2.B`, `class-balanced` |
| v2.C | `ablation`, `v2.C`, `lovasz` |
| v2.final | `ablation`, `v2.final`, `combined` |
| Colab final | `colab`, `final` |

WandB 웹에서 tag filter로 비교 표 자동 생성.

---

## 10. Workflow & Phases

날짜 명시 X — 가능한 빨리 진행. 외부 이벤트 (4/29 threshold, 5/5 마감) 고정.

| Phase | 작업 |
|---|---|
| **P1** | v2.0 학습 시작 (background, ~11.5h). 동시에 Class-balanced + Lovász 구현 + 단위 테스트 |
| **P2** | v2.0 완료, eval. Copy-Paste 구현 (가장 큰 작업, 6-8h) |
| **P3** | v2.A (Copy-Paste) 학습 (6h) |
| **P4** | v2.A eval + v2.B (Class-balanced) 학습 (6h) |
| **P5** | v2.B eval + v2.C (Lovász) 학습 (7-8h, sort overhead) |
| **P6** | v2.C eval + 결과 종합 + v2.final config 결정 |
| **★ 4/29 Threshold** | 외부 이벤트. 점수 시뮬레이션 → ablation 더 할지 결정 |
| **P7** | v2.final 학습 (Stage 1 + 2 통째, 11.5h) |
| **P8** | v2.final eval + TTA eval + 비교 정리 |
| **P9** | Colab 재현 (winning config) |
| **P10** | Submission packaging + Report 작성 |
| **마감** | 5/5 23:59 |

---

## 11. Risk Management

| 위험 | 완화책 |
|---|---|
| ablation 모두 negative | v2.0 그대로 또는 v1 사용. 시간 손해 1-2일 |
| Copy-Paste 구현 버그 | smoke test + 시각화로 사전 검증 |
| 학습 중 OOM | batch_size 16 → 12 fallback |
| Lovász 학습 불안정 (NaN) | weight 0.5 → 0.3, 또는 fp16 → fp32 fallback |
| 시간 초과 | v2.final skip하고 winning ablation 중 best 1개로 직접 |
| 4/29 threshold strict | A 모델 (MobileNetV3+LR-ASPP, 7.75 GFLOPs) fallback (이미 v1 브랜치에 코드) |
| v2.0 효과 미미 (<+1%) | Stage 1 160K 폐기, v1 base에서 Copy-Paste/Lovász/Class-balanced 진행 |

---

## 12. Stop Conditions

학습 중 또는 ablation 후 다음 조건 충족 시 중단/조정:

- 어떤 ablation Δ TTA > +1% → 확실히 keep
- 어떤 ablation Δ TTA ∈ [-0.5, +1] → drop (효과 없음)
- 어떤 ablation Δ TTA < -0.5% → drop + 분석
- v2.final이 모든 individual ablation보다 낮음 → individual best 선택
- 4/29 threshold가 매우 lenient (예: 75% mIoU = 5점) → 추가 ablation 중단, Colab 재현 직행
- 4/29 threshold가 매우 strict (mIoU 85%+) → A 모델로 fallback (FLOPs 5점 + mIoU 3-4점)

---

## 13. 다음 단계

이 spec 기반으로 implementation plan 작성 (writing-plans 스킬). 각 ablation별 task 분해, TDD 또는 smoke test 패턴, commit 전략 명시.
