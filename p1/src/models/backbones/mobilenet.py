"""MobileNetV3-Large backbone for segmentation (LR-ASPP 기반).

dilated 마지막 inverted residual block들 → effective OS=16.
forward (low_level, high_level) 반환:
- low_level: features[~6] 출력 (40ch, H/8)
- high_level: features 끝 (960ch, H/16, dilated)
"""
from __future__ import annotations

from typing import Tuple

import torch.nn as nn
from torch import Tensor
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


class MobileNetV3LargeBackbone(nn.Module):
    LOW_CHANNELS = 40
    HIGH_CHANNELS = 960

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        net = mobilenet_v3_large(weights=weights)
        # OS=16 위해 features[7..16]에 dilation 2 적용 (stride 2 → 1)
        for i in range(7, len(net.features)):
            for m in net.features[i].modules():
                if isinstance(m, nn.Conv2d) and m.stride == (2, 2) and m.kernel_size != (1, 1):
                    m.stride = (1, 1)
                    m.dilation = (2, 2)
                    pad = m.kernel_size[0] // 2 * 2
                    m.padding = (pad, pad)
        self.features = net.features
        self.low_idx = 6   # features[0..6] → ~H/8, ~40ch
        self.high_idx = len(net.features) - 1

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        low = None
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == self.low_idx:
                low = x
        return low, x
