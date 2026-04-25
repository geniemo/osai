"""Class-balanced sampling for VOC train (Stage 2 ablation v2.B).

Inverse sqrt frequency 기반 image weight 계산 → WeightedRandomSampler.
weak class (chair, bicycle, sofa, pottedplant) 포함 이미지 더 자주 sample.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import WeightedRandomSampler

NUM_CLASSES = 21
IGNORE_INDEX = 255


def compute_class_pixel_counts(voc_root: str, train_ids: List[str]) -> np.ndarray:
    """VOC train 마스크에서 class별 픽셀 수 합계. Returns (21,) int64 array."""
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    seg_dir = Path(voc_root) / "VOCdevkit" / "VOC2012" / "SegmentationClass"
    for img_id in train_ids:
        mask = np.array(Image.open(seg_dir / f"{img_id}.png"))
        for c in range(NUM_CLASSES):
            counts[c] += int((mask == c).sum())
    return counts


def compute_image_weights(
    voc_root: str,
    train_ids: List[str],
    class_weights: np.ndarray,
) -> torch.Tensor:
    """이미지에 등장하는 class들의 class_weight 합 = sampling weight."""
    seg_dir = Path(voc_root) / "VOCdevkit" / "VOC2012" / "SegmentationClass"
    image_weights = []
    for img_id in train_ids:
        mask = np.array(Image.open(seg_dir / f"{img_id}.png"))
        present = np.unique(mask)
        present = present[(present != IGNORE_INDEX) & (present < NUM_CLASSES)]
        w = float(class_weights[present].sum())
        image_weights.append(w)
    return torch.tensor(image_weights, dtype=torch.float32)


def build_balanced_sampler(voc_root: str, train_ids: List[str]) -> WeightedRandomSampler:
    """Pipeline: counts → class_weights (inverse sqrt) → image_weights → Sampler."""
    counts = compute_class_pixel_counts(voc_root, train_ids)
    class_weights = 1.0 / np.sqrt(counts.astype(np.float64) + 1.0)
    class_weights = class_weights / class_weights.sum()
    image_weights = compute_image_weights(voc_root, train_ids, class_weights)
    return WeightedRandomSampler(
        weights=image_weights,
        num_samples=len(image_weights),
        replacement=True,
    )
