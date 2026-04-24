"""ResNet-50 backbone for segmentation.

TorchVision IMAGENET1K_V2 pretrained, layer4를 dilated conv로 OS=32→16.
forward는 (c2, c5) 반환 — c2는 DLv3+ decoder의 low-level skip용.
"""
from __future__ import annotations

from typing import Tuple

import torch.nn as nn
from torch import Tensor
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50Backbone(nn.Module):
    LOW_CHANNELS = 256
    HIGH_CHANNELS = 2048

    def __init__(self, output_stride: int = 16, pretrained: bool = True) -> None:
        super().__init__()
        if output_stride == 16:
            replace = [False, False, True]
        elif output_stride == 8:
            replace = [False, True, True]
        else:
            raise ValueError(f"output_stride must be 8 or 16, got {output_stride}")

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = resnet50(weights=weights, replace_stride_with_dilation=replace)

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c2, c5
