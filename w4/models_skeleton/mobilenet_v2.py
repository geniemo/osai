"""MobileNetV2 skeleton for practice."""

# import torch
import torch.nn as nn
from torch import Tensor


def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_ch: int, out_ch: int) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)


class InvertedResidual(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, expand_ratio: int) -> None:
        super().__init__()
        # TODO: implement inverted residual block

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C, H, W)
        # TODO: implement forward
        raise NotImplementedError("InvertedResidual forward is not implemented.")


class MobileNetV2(nn.Module):
    def __init__(self, num_classes: int = 1000, width_mult: float = 1.0) -> None:
        super().__init__()
        # TODO: implement MobileNetV2 architecture
        self.stem = nn.Identity()

        # Output projection
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, 3, H, W)
        # TODO: implement forward
        raise NotImplementedError("MobileNetV2 forward is not implemented.")


def mobilenet_v2(num_classes: int = 1000) -> MobileNetV2:
    return MobileNetV2(num_classes=num_classes)