"""COCO 2017 train (VOC subset) dataset.

핵심:
- 20-class hard-coded mapping (COCO id → VOC id)
- non-VOC class object → 255 (ignore)
- mask 사전 캐싱 (1회 ~30-60분)
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

COCO_TO_VOC = {
    1:15, 2:2, 3:7, 4:14, 5:1, 6:6, 7:19, 9:4, 16:3, 17:8,
    18:12, 19:13, 20:17, 21:10, 44:5, 62:9, 63:18, 64:16, 67:11, 72:20,
}


def build_voc_mask_from_anns(anns: List[dict], h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=np.uint8)
    for ann in anns:
        m = ann["binary_mask"]
        cat = ann["category_id"]
        if cat in COCO_TO_VOC:
            out[m == 1] = COCO_TO_VOC[cat]
        else:
            out[m == 1] = 255
    return out


class COCOSegDataset(Dataset):
    def __init__(
        self,
        coco_root: str,
        split: str = "train2017",
        transform: Optional[Callable] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        from pycocotools.coco import COCO

        self.coco_root = Path(coco_root)
        self.split = split
        self.transform = transform
        self.img_dir = self.coco_root / split
        ann_file = self.coco_root / "annotations" / f"instances_{split}.json"
        self.coco = COCO(str(ann_file))
        voc_cat_ids = list(COCO_TO_VOC.keys())
        keep_ids = set()
        for cid in voc_cat_ids:
            keep_ids.update(self.coco.getImgIds(catIds=[cid]))
        self.image_ids = sorted(keep_ids)
        self.cache_dir = Path(cache_dir) if cache_dir else (self.coco_root / "coco_voc_masks")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.image_ids)

    def _build_mask(self, image_id: int, h: int, w: int) -> np.ndarray:
        ann_ids = self.coco.getAnnIds(imgIds=[image_id])
        anns_raw = self.coco.loadAnns(ann_ids)
        anns = [{"category_id": a["category_id"], "binary_mask": self.coco.annToMask(a)} for a in anns_raw]
        return build_voc_mask_from_anns(anns, h, w)

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        info = self.coco.loadImgs([image_id])[0]
        img = Image.open(self.img_dir / info["file_name"]).convert("RGB")
        cache_path = self.cache_dir / f"{image_id}.png"
        if cache_path.exists():
            mask_arr = np.array(Image.open(cache_path))
        else:
            mask_arr = self._build_mask(image_id, info["height"], info["width"])
            Image.fromarray(mask_arr, mode="L").save(cache_path)
        mask = Image.fromarray(mask_arr, mode="L")
        if self.transform is not None:
            img, mask = self.transform(img, mask)
        return img, mask
