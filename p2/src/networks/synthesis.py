"""StyleGAN2 Synthesis network with skip-G structure.

Each resolution stage:
  ModulatedConv (up if not first) → AddNoise → BiasAct
  ModulatedConv (no up) → AddNoise → BiasAct
  ToRGB (modulated 1×1) → bilinear-upsample previous RGB → sum

Final RGB at target resolution.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import ModulatedConv2d, AddNoise, BiasAct


class ToRGB(nn.Module):
    def __init__(self, in_ch: int, w_dim: int):
        super().__init__()
        self.conv = ModulatedConv2d(in_ch, 3, kernel=1, style_dim=w_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(3))

    def forward(self, x: torch.Tensor, w: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        out = self.conv(x, w)
        out = out + self.bias.view(1, 3, 1, 1)
        if skip is not None:
            skip = F.interpolate(skip, scale_factor=2, mode="bilinear", align_corners=False)
            out = out + skip
        return out


class SynthesisBlock(nn.Module):
    """One resolution stage: upsample → conv → noise → act → conv → noise → act."""

    def __init__(self, in_ch: int, out_ch: int, w_dim: int, *, up: bool = True):
        super().__init__()
        self.up = up
        self.conv0 = ModulatedConv2d(in_ch, out_ch, kernel=3, style_dim=w_dim, up=up)
        self.noise0 = AddNoise(out_ch)
        self.act0 = BiasAct(out_ch)
        self.conv1 = ModulatedConv2d(out_ch, out_ch, kernel=3, style_dim=w_dim, up=False)
        self.noise1 = AddNoise(out_ch)
        self.act1 = BiasAct(out_ch)

    def forward(self, x: torch.Tensor, w0: torch.Tensor, w1: torch.Tensor) -> torch.Tensor:
        x = self.conv0(x, w0)
        x = self.noise0(x)
        x = self.act0(x)
        x = self.conv1(x, w1)
        x = self.noise1(x)
        x = self.act1(x)
        return x


class SynthesisNet(nn.Module):
    """Skip-G synthesis. Const 4×4 input → progressive stages → final RGB."""

    def __init__(self, channels: dict[int, int], w_dim: int):
        super().__init__()
        self.w_dim = w_dim
        resolutions = sorted(channels.keys())
        self.resolutions = resolutions
        first_res = resolutions[0]
        first_ch = channels[first_res]

        # Const 4×4 input (learnable)
        self.const = nn.Parameter(torch.randn(1, first_ch, first_res, first_res))

        # First stage: no upsample, single conv path
        self.first_conv = ModulatedConv2d(first_ch, first_ch, kernel=3, style_dim=w_dim, up=False)
        self.first_noise = AddNoise(first_ch)
        self.first_act = BiasAct(first_ch)
        self.first_to_rgb = ToRGB(first_ch, w_dim)

        # Subsequent stages
        self.blocks = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        in_ch = first_ch
        for res in resolutions[1:]:
            out_ch = channels[res]
            self.blocks.append(SynthesisBlock(in_ch, out_ch, w_dim, up=True))
            self.to_rgbs.append(ToRGB(out_ch, w_dim))
            in_ch = out_ch

        # w layer count: first stage uses 2 slices (1 conv + 1 ToRGB)
        # each subsequent block uses 3 slices (2 convs + 1 ToRGB)
        self.num_w_layers = 2 + len(self.blocks) * 3

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """w shape: (B, num_w_layers, w_dim). Each layer consumes one w slice."""
        B = w.shape[0]
        idx = 0
        x = self.const.expand(B, -1, -1, -1).contiguous()
        x = self.first_conv(x, w[:, idx]); idx += 1
        x = self.first_noise(x)
        x = self.first_act(x)
        rgb = self.first_to_rgb(x, w[:, idx], skip=None); idx += 1

        for block, to_rgb in zip(self.blocks, self.to_rgbs):
            x = block(x, w[:, idx], w[:, idx + 1]); idx += 2
            rgb = to_rgb(x, w[:, idx], skip=rgb); idx += 1
        return rgb
