"""LR-ASPP head (Searching for MobileNetV3 paper).

LR-ASPP는 backbone (low, high) 둘 다 사용하는 별도 구조 → SegmentationModel과 별개로
LRASPPModel class로 별도 구현. builder에서 head=='lraspp'면 이걸 직접 반환.
"""
from __future__ import annotations

import torch.nn as nn
from torch import Tensor
from torch.nn.functional import interpolate


class LRASPPHead(nn.Module):
    """Internal head — used by LRASPPModel."""

    def __init__(self, low_in: int, high_in: int, num_classes: int, mid: int = 128) -> None:
        super().__init__()
        self.cbr = nn.Sequential(
            nn.Conv2d(high_in, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(high_in, mid, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.high_classifier = nn.Conv2d(mid, num_classes, kernel_size=1)
        self.low_classifier = nn.Conv2d(low_in, num_classes, kernel_size=1)


class LRASPPModel(nn.Module):
    """LR-ASPP는 backbone (low, high) 둘 다 사용 → 별도 model class.

    SegmentationModel과 동일한 export_mode() 인터페이스 제공.
    """

    def __init__(self, backbone: nn.Module, low_in: int, high_in: int, num_classes: int, mid: int = 128) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = LRASPPHead(low_in, high_in, num_classes, mid=mid)
        self._export = False
        # placeholders for compatibility with SegmentationModel-like access
        self.aux_head = None

    def export_mode(self):
        self._export = True
        return self

    def forward(self, x: Tensor) -> Tensor:
        h, w = x.shape[2:]
        low, high = self.backbone(x)
        feat = self.head.cbr(high) * self.head.scale(high)
        high_cls = self.head.high_classifier(feat)
        low_cls = self.head.low_classifier(low)
        high_up = interpolate(high_cls, size=low.shape[2:], mode="bilinear", align_corners=False)
        out = high_up + low_cls
        return interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
