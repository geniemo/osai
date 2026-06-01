"""StyleGAN2-style residual discriminator with MinibatchStd."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import EqualConv2d, EqualLinear


@dataclass
class DiscriminatorConfig:
    channels: dict[int, int]
    minibatch_std_group: int = 4

    def __post_init__(self):
        self.channels = {int(k): int(v) for k, v in self.channels.items()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DiscriminatorConfig":
        return cls(channels=d["channels"], minibatch_std_group=d.get("minibatch_std_group", 4))


class MinibatchStd(nn.Module):
    def __init__(self, group_size: int = 4):
        super().__init__()
        self.group_size = group_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        g = min(self.group_size, B)
        if B % g != 0:
            g = B
        y = x.view(g, B // g, C, H, W)
        y = y - y.mean(dim=0, keepdim=True)
        y = (y.pow(2).mean(dim=0) + 1e-8).sqrt()
        y = y.mean(dim=[1, 2, 3], keepdim=True)
        y = y.repeat(g, 1, H, W)
        return torch.cat([x, y], dim=1)


class ResBlockDown(nn.Module):
    """leaky_relu → conv 3×3 (same ch) → leaky_relu → conv 3×3 (out_ch) → avgpool 2.
    Skip: avgpool 2 → conv 1×1 (out_ch). Sum / sqrt(2).
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv0 = EqualConv2d(in_ch, in_ch, kernel=3)
        self.conv1 = EqualConv2d(in_ch, out_ch, kernel=3)
        self.skip = EqualConv2d(in_ch, out_ch, kernel=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv0(F.leaky_relu(x, 0.2)) * math.sqrt(2.0)
        h = self.conv1(F.leaky_relu(h, 0.2)) * math.sqrt(2.0)
        h = F.avg_pool2d(h, 2)
        skip = F.avg_pool2d(x, 2)
        skip = self.skip(skip)
        return (h + skip) / math.sqrt(2.0)


class Discriminator(nn.Module):
    def __init__(self, cfg: DiscriminatorConfig):
        super().__init__()
        self.cfg = cfg
        resolutions = sorted(cfg.channels.keys(), reverse=True)  # 256 → ... → 4
        self.resolutions = resolutions
        first_ch = cfg.channels[resolutions[0]]
        self.from_rgb = EqualConv2d(3, first_ch, kernel=1)

        blocks: list[nn.Module] = []
        in_ch = first_ch
        for res in resolutions[1:]:
            out_ch = cfg.channels[res]
            blocks.append(ResBlockDown(in_ch, out_ch))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        last_res = resolutions[-1]
        last_ch = cfg.channels[last_res]
        self.minibatch_std = MinibatchStd(cfg.minibatch_std_group)
        self.final_conv = EqualConv2d(last_ch + 1, last_ch, kernel=3)
        self.final_linear = EqualLinear(last_ch * last_res * last_res, last_ch)
        self.score_linear = EqualLinear(last_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.from_rgb(x), 0.2) * math.sqrt(2.0)
        h = self.blocks(h)
        h = self.minibatch_std(h)
        h = F.leaky_relu(self.final_conv(h), 0.2) * math.sqrt(2.0)
        h = h.flatten(1)
        h = F.leaky_relu(self.final_linear(h), 0.2) * math.sqrt(2.0)
        return self.score_linear(h)
