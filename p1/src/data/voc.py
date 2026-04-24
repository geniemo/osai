"""Pascal-VOC 2012 segmentation dataset (trainaug + val).

trainaug = VOC 2012 train (1464) + SBD extra (9118) – val overlap (= 10582 total).
SBD ID list는 download.py가 다운로드 후 'train_aug.txt'로 저장.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
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

        if split == "trainaug":
            id_file = self.devkit / "ImageSets" / "Segmentation" / "train_aug.txt"
            mask_dir = self.devkit / "SegmentationClassAug"
        elif split == "val":
            id_file = self.devkit / "ImageSets" / "Segmentation" / "val.txt"
            mask_dir = self.devkit / "SegmentationClass"
        else:
            raise ValueError(f"Unknown split: {split}")

        with open(id_file) as f:
            self.ids = [line.strip() for line in f if line.strip()]
        self.img_dir = self.devkit / "JPEGImages"
        self.mask_dir = mask_dir

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        img_id = self.ids[index]
        img = Image.open(self.img_dir / f"{img_id}.jpg").convert("RGB")
        mask = Image.open(self.mask_dir / f"{img_id}.png")
        if self.transform is not None:
            img, mask = self.transform(img, mask)
        return img, mask
