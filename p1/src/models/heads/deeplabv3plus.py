"""DeepLabV3+ decoder: low-level skip + ASPP feature fusion.

ASPP_out (H/16) ─upsample×4─→ + low_level (1×1 → 48ch) ─→ concat (304) ─→
3×3 conv → 256 → 3×3 conv → 256 → 1×1 conv → num_classes ─upsample×4─→ logits at input H,W.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import interpolate


class DeepLabV3PlusHead(nn.Module):
    def __init__(self, low_in_channels: int, aspp_out_channels: int, num_classes: int, low_proj_channels: int = 48) -> None:
        super().__init__()
        self.low_proj = nn.Sequential(
            nn.Conv2d(low_in_channels, low_proj_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_proj_channels),
            nn.ReLU(inplace=True),
        )
        merged = aspp_out_channels + low_proj_channels
        self.decoder = nn.Sequential(
            nn.Conv2d(merged, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, aspp_out: Tensor, low_level: Tensor, output_size: tuple) -> Tensor:
        low = self.low_proj(low_level)
        h_low, w_low = low.shape[2:]
        aspp_up = interpolate(aspp_out, size=(h_low, w_low), mode="bilinear", align_corners=False)
        x = torch.cat([aspp_up, low], dim=1)
        x = self.decoder(x)
        x = self.classifier(x)
        x = interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return x
