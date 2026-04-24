"""Atrous Spatial Pyramid Pooling (DeepLabV3+ style).

5 branches (1×1, 3×3 d=6, d=12, d=18, GAP) → concat → 1×1 → BN → ReLU → Dropout.
Reference: w4/models/deeplab_v3.py.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import interpolate


def _conv1x1(c_in: int, c_out: int) -> nn.Conv2d:
    return nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)


def _conv3x3(c_in: int, c_out: int, dilation: int) -> nn.Conv2d:
    return nn.Conv2d(c_in, c_out, kernel_size=3, padding=dilation, dilation=dilation, bias=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 256, rates: List[int] = (6, 12, 18)) -> None:
        super().__init__()
        self.branches = nn.ModuleList()
        self.branches.append(nn.Sequential(_conv1x1(in_channels, out_channels), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)))
        for r in rates:
            self.branches.append(nn.Sequential(_conv3x3(in_channels, out_channels, r), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)))
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            _conv1x1(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        n_branches = 1 + len(rates) + 1
        self.project = nn.Sequential(
            _conv1x1(out_channels * n_branches, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: Tensor) -> Tensor:
        h, w = x.shape[2:]
        outs = [b(x) for b in self.branches]
        outs.append(interpolate(self.global_pool(x), size=(h, w), mode="bilinear", align_corners=False))
        return self.project(torch.cat(outs, dim=1))
