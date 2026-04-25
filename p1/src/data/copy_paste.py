"""Copy-Paste augmentation for segmentation (Ghiasi et al. 2021).

VOC SegmentationObject로 instance 추출 → 다른 학습 이미지 위에 paste.
v2.A ablation에 사용.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def build_instance_pool(
    voc_root: str,
    train_ids: List[str],
    min_area: int = 64 * 64,
    max_area_ratio: float = 0.25,
) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """VOC train의 SegmentationObject에서 모든 instance 추출.

    Args:
        voc_root: e.g. "./data/voc"
        train_ids: VOC train image IDs
        min_area: instance 최소 픽셀 수
        max_area_ratio: 이미지 대비 instance 최대 비율 (0~1)

    Returns:
        list of (patch_img: HxWx3 uint8, patch_mask: HxW bool, class_id: int).
    """
    devkit = Path(voc_root) / "VOCdevkit" / "VOC2012"
    seg_cls_dir = devkit / "SegmentationClass"
    seg_obj_dir = devkit / "SegmentationObject"
    img_dir = devkit / "JPEGImages"

    pool = []
    for img_id in train_ids:
        seg_obj_path = seg_obj_dir / f"{img_id}.png"
        if not seg_obj_path.exists():
            continue
        seg_obj = np.array(Image.open(seg_obj_path))
        seg_cls = np.array(Image.open(seg_cls_dir / f"{img_id}.png"))
        img = np.array(Image.open(img_dir / f"{img_id}.jpg").convert("RGB"))
        H, W = seg_obj.shape

        for inst_id in np.unique(seg_obj):
            if inst_id == 0 or inst_id == 255:
                continue
            inst_mask = seg_obj == inst_id
            area = int(inst_mask.sum())
            if area < min_area or area > int(H * W * max_area_ratio):
                continue
            cls = int(seg_cls[inst_mask][0])
            if cls == 0 or cls == 255:
                continue
            ys, xs = np.where(inst_mask)
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            patch_img = img[y0:y1, x0:x1].copy()
            patch_mask = inst_mask[y0:y1, x0:x1].copy()
            pool.append((patch_img, patch_mask, cls))

    return pool


def paste_instance(
    target_img: np.ndarray,
    target_mask: np.ndarray,
    patch_img: np.ndarray,
    patch_mask: np.ndarray,
    cls_id: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Paste single instance onto target. In-place modification."""
    Ht, Wt = target_img.shape[:2]
    Hp, Wp = patch_img.shape[:2]

    if Hp > Ht * 0.5 or Wp > Wt * 0.5:
        scale = min(Ht * 0.5 / Hp, Wt * 0.5 / Wp)
        new_H, new_W = max(1, int(Hp * scale)), max(1, int(Wp * scale))
        patch_img = cv2.resize(patch_img, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
        patch_mask = cv2.resize(
            patch_mask.astype(np.uint8), (new_W, new_H), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        Hp, Wp = new_H, new_W

    if random.random() < 0.5:
        patch_img = patch_img[:, ::-1].copy()
        patch_mask = patch_mask[:, ::-1].copy()

    if Ht <= Hp or Wt <= Wp:
        return target_img, target_mask
    cy = random.randint(Hp // 2, Ht - Hp + Hp // 2)
    cx = random.randint(Wp // 2, Wt - Wp + Wp // 2)
    y0, y1 = cy - Hp // 2, cy - Hp // 2 + Hp
    x0, x1 = cx - Wp // 2, cx - Wp // 2 + Wp

    y0, y1 = max(0, y0), min(Ht, y1)
    x0, x1 = max(0, x0), min(Wt, x1)
    Hp_clip, Wp_clip = y1 - y0, x1 - x0
    patch_img = patch_img[:Hp_clip, :Wp_clip]
    patch_mask = patch_mask[:Hp_clip, :Wp_clip]

    region = target_img[y0:y1, x0:x1]
    region[patch_mask] = patch_img[patch_mask]
    target_img[y0:y1, x0:x1] = region
    target_mask[y0:y1, x0:x1][patch_mask] = cls_id

    return target_img, target_mask


class CopyPasteDataset(Dataset):
    """Wraps VOCSegDataset; with prob p, paste 1-N instances from pool.

    class_weights={cls_id: weight}로 instance sampling 가중 가능 (v2.A.2).
    None이면 uniform sampling (v2.A 표준).
    """

    def __init__(
        self,
        base: Dataset,
        instance_pool: List[Tuple[np.ndarray, np.ndarray, int]],
        p: float = 0.5,
        num_paste: Tuple[int, int] = (1, 3),
        class_weights: Optional[dict] = None,
    ) -> None:
        self.base = base
        self.pool = instance_pool
        self.p = p
        self.num_paste_range = num_paste
        # Per-instance sampling weight (None → uniform)
        if class_weights and len(instance_pool) > 0:
            self.instance_weights = [
                float(class_weights.get(cls, 1.0)) for _, _, cls in instance_pool
            ]
        else:
            self.instance_weights = None

    def __len__(self) -> int:
        return len(self.base)

    def _sample_one(self):
        if self.instance_weights is not None:
            return random.choices(self.pool, weights=self.instance_weights, k=1)[0]
        return random.choice(self.pool)

    def __getitem__(self, idx: int):
        img_pil, mask_pil = self.base.get_raw(idx)
        if random.random() < self.p and len(self.pool) > 0:
            img_arr = np.array(img_pil)
            mask_arr = np.array(mask_pil)
            n = random.randint(*self.num_paste_range)
            for _ in range(n):
                patch_img, patch_mask, cls_id = self._sample_one()
                img_arr, mask_arr = paste_instance(
                    img_arr, mask_arr, patch_img, patch_mask, cls_id
                )
            img_pil = Image.fromarray(img_arr)
            mask_pil = Image.fromarray(mask_arr)
        return self.base.apply_transform(img_pil, mask_pil)
