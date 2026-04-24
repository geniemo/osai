"""Joint image-mask transforms using torchvision.transforms.v2.

Mask는 tv_tensors.Mask로 wrap → NEAREST + ignore=255 fill 자동 처리.
Color/blur/erasing 등 image-only 변환은 자동으로 mask 영향 없음.

파이프라인 순서 주의:
- RandomResize → RandomCrop: ScaleJitter 대신 사용해 crop_size 이상으로 보장
- ColorJitter, RandomGrayscale, GaussianBlur는 uint8 또는 [0, 1] float에서 작동.
  따라서 ToDtype/Normalize 이전에 위치해야 올바른 HSV 변환이 이루어진다.
- RandomErasing은 정규화된 텐서에서도 OK (값을 0으로 erase).
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor
from torchvision import tv_tensors
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _PairWrapper:
    """Compose v2 transform이 (image, mask) tuple을 받도록 wrap."""

    def __init__(self, transform: v2.Transform) -> None:
        self.transform = transform

    def __call__(self, image, mask) -> Tuple[Tensor, Tensor]:
        img_tv = tv_tensors.Image(image)
        mask_tv = tv_tensors.Mask(mask)
        out_img, out_mask = self.transform(img_tv, mask_tv)
        # Mask는 (1, H, W)로 wrap되므로 (H, W)로 squeeze
        return out_img.as_subclass(torch.Tensor), out_mask.as_subclass(torch.Tensor).squeeze(0).long()


def build_train_transform(
    crop_size: int = 480,
    scale_range: Tuple[float, float] = (0.5, 2.0),
) -> _PairWrapper:
    # RandomResize: 짧은 변 기준으로 [min_size, max_size] 범위 랜덤 리사이즈.
    # ScaleJitter 대신 사용: crop_size * scale_range → both dims >= crop_size * scale_range[0].
    # scale_range[0] >= 1.0이면 패딩 불필요; < 1.0이면 pad_if_needed가 처리.
    min_s = int(scale_range[0] * crop_size)
    max_s = int(scale_range[1] * crop_size)
    if max_s <= min_s:
        max_s = min_s + 1
    pipeline = v2.Compose([
        v2.RandomResize(min_size=min_s, max_size=max_s, antialias=True),
        v2.RandomCrop(size=crop_size, pad_if_needed=True, fill={tv_tensors.Image: 0, tv_tensors.Mask: 255}),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        v2.RandomGrayscale(p=0.1),
        v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
        v2.RandomErasing(p=0.25),
    ])
    return _PairWrapper(pipeline)


def build_val_transform() -> _PairWrapper:
    pipeline = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
    ])
    return _PairWrapper(pipeline)
