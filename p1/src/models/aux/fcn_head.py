"""FCN-style auxiliary head for deep supervision (학습 전용).

ONNX export 전 SegmentationModel.export_mode()가 이 head를 비활성화.
"""
from __future__ import annotations

import torch.nn as nn
from torch import Tensor
from torch.nn.functional import interpolate


class FCNHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, mid_channels: int = 256) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, num_classes, kernel_size=1),
        )

    def forward(self, x: Tensor, output_size: tuple) -> Tensor:
        y = self.conv(x)
        return interpolate(y, size=output_size, mode="bilinear", align_corners=False)
