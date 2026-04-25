"""Pascal-VOC 2012 segmentation dataset (train + val).

PDF 정책: ImageNet/COCO/VOC만 허용. SBD는 명시 X → **사용 안 함**.
- split="train" → VOC 2012 train (1,464 images, SegmentationClass)
- split="val"   → VOC 2012 val (1,449 images)

만약 추후 교수님이 SBD 허용 확인 시: split="trainaug" 분기를 다시 추가하고
download.py에서 SBD merge 활성화.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset


class VOCSegDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        transform: Optional[Callable] = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.devkit = self.root / "VOCdevkit" / "VOC2012"

        if split == "train":
            id_file = self.devkit / "ImageSets" / "Segmentation" / "train.txt"
        elif split == "val":
            id_file = self.devkit / "ImageSets" / "Segmentation" / "val.txt"
        else:
            raise ValueError(f"Unknown split: {split} (allowed: 'train', 'val')")

        mask_dir = self.devkit / "SegmentationClass"

        with open(id_file) as f:
            self.ids = [line.strip() for line in f if line.strip()]
        self.img_dir = self.devkit / "JPEGImages"
        self.mask_dir = mask_dir

    def __len__(self) -> int:
        return len(self.ids)

    def get_raw(self, index: int) -> Tuple[Image.Image, Image.Image]:
        """Return raw PIL (image, mask) without transform.

        Used by Copy-Paste wrapper to manipulate before applying transforms.
        """
        img_id = self.ids[index]
        img = Image.open(self.img_dir / f"{img_id}.jpg").convert("RGB")
        mask = Image.open(self.mask_dir / f"{img_id}.png")
        return img, mask

    def apply_transform(self, img: Image.Image, mask: Image.Image):
        """Apply self.transform to (img, mask) tuple."""
        if self.transform is not None:
            return self.transform(img, mask)
        return img, mask

    def __getitem__(self, index: int):
        img, mask = self.get_raw(index)
        return self.apply_transform(img, mask)
