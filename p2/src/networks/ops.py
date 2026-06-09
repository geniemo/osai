"""Core StyleGAN2 operations: equalized LR + modulated conv + noise/bias act."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EqualLinear(nn.Module):
    """Linear with equalized learning rate.

    Init weight ~ N(0, 1) (raw), runtime multiplied by 1/sqrt(in_dim) * lr_mul.
    Bias init 0 (raw), runtime multiplied by lr_mul.
    """

    def __init__(self, in_dim: int, out_dim: int, *, bias: bool = True, lr_mul: float = 1.0, bias_init: float = 0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim) / lr_mul)
        self.bias = nn.Parameter(torch.full((out_dim,), bias_init / lr_mul)) if bias else None
        self.runtime_scale = (1.0 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight * self.runtime_scale
        b = self.bias * self.lr_mul if self.bias is not None else None
        return F.linear(x, w, b)


class EqualConv2d(nn.Module):
    """Conv2d with equalized LR. Stride/padding fixed to stride=1, padding=k//2."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, *, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, kernel, kernel))
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None
        self.runtime_scale = 1.0 / math.sqrt(in_ch * kernel * kernel)
        self.padding = kernel // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight * self.runtime_scale
        return F.conv2d(x, w, self.bias, padding=self.padding)


class ModulatedConv2d(nn.Module):
    """Per-sample weight-modulated convolution with optional demodulation and upsample.

    StyleGAN2 §B (Karras 2020). Implementation via group conv trick:
      - Weights are reshaped to (B*out, in, k, k); input to (1, B*in, H, W).
      - F.conv2d with groups=B applies a separate filter per sample.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int, style_dim: int,
                 *, demodulate: bool = True, up: bool = False):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel = kernel
        self.demodulate = demodulate
        self.up = up
        self.padding = kernel // 2
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, kernel, kernel))
        self.runtime_scale = 1.0 / math.sqrt(in_ch * kernel * kernel)
        # affine projection from style w → per-input-channel scale
        self.affine = EqualLinear(style_dim, in_ch, bias=True, bias_init=1.0)
        # _w_var_cache: registered as a buffer (None by default), populated by
        # prepare_for_onnx_export() before ONNX trace
        self.register_buffer("_w_var_cache", None, persistent=False)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        style = self.affine(w)  # (B, in_ch)
        weight = self.weight * self.runtime_scale  # (out, in, k, k)
        weight = weight.unsqueeze(0)  # (1, out, in, k, k)
        weight = weight * style.view(B, 1, C, 1, 1)  # (B, out, in, k, k)
        if self.demodulate:
            d = torch.rsqrt(weight.square().sum(dim=[2, 3, 4]) + 1e-8)  # (B, out)
            weight = weight * d.view(B, self.out_ch, 1, 1, 1)
        # group conv
        weight = weight.view(B * self.out_ch, C, self.kernel, self.kernel)
        x = x.view(1, B * C, H, W)
        if self.up:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            H_out, W_out = H * 2, W * 2
        else:
            H_out, W_out = H, W
        x = F.conv2d(x, weight, padding=self.padding, groups=B)
        x = x.view(B, self.out_ch, H_out, W_out)
        return x

    def forward_onnx(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Mathematically equivalent forward without the groups=B trick.

        Reformulation: apply `runtime_scale` to `style` (input side) instead of
        to `weight`, and read the demod-side `sum_k weight^2` from a buffer
        (`_w_var_cache`) populated by `prepare_for_onnx_export()`. With this:
          - `self.weight` is referenced exactly once (the conv), so ONNX
            exporter emits it as a single initializer instead of duplicating.
          - `_w_var_cache` is a small (out, in) buffer — much smaller than
            the full weight tensor.

        Identity used for the conv:
            conv(input * (rs * style[i]), weight)
              == conv(input, weight * rs * style[i])         (linearity in i)
              == per-sample-modulated conv  (original group-conv trick output)
        """
        assert self._w_var_cache is not None, (
            "Call prepare_for_onnx_export() before using forward_onnx() "
            "(needed to dedup self.weight in the ONNX graph)."
        )
        B, C, H, W = x.shape
        style_scaled = self.affine(w) * self.runtime_scale  # (B, in_ch)

        x_in = x * style_scaled.view(B, C, 1, 1)
        if self.up:
            x_in = F.interpolate(x_in, scale_factor=2, mode="bilinear", align_corners=False)
        x_out = F.conv2d(x_in, self.weight, padding=self.padding)

        if self.demodulate:
            style_sq = style_scaled.square()                                       # (B, in)
            per_out = (style_sq.unsqueeze(1) * self._w_var_cache.unsqueeze(0)).sum(dim=2)
            d = torch.rsqrt(per_out + 1e-8)
            x_out = x_out * d.view(B, self.out_ch, 1, 1)

        return x_out

    def prepare_for_onnx_export(self) -> None:
        """Cache `sum_k weight^2` in the existing buffer so forward_onnx can
        reference self.weight exactly once (avoids ONNX initializer duplication)."""
        with torch.no_grad():
            self._w_var_cache = self.weight.square().sum(dim=[2, 3]).detach()  # (out, in)


class AddNoise(nn.Module):
    """Add learnable-scale per-pixel Gaussian noise."""

    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            B, C, H, W = x.shape
            noise = torch.randn(B, 1, H, W, device=x.device, dtype=x.dtype)
        return x + self.weight * noise


class BiasAct(nn.Module):
    """Bias add + leaky ReLU (0.2) + sqrt(2) gain (StyleGAN2 act_gain)."""

    GAIN = math.sqrt(2.0)

    def __init__(self, channels: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.bias.view(1, -1, *([1] * (x.ndim - 2)))
        x = F.leaky_relu(x, negative_slope=0.2)
        return x * self.GAIN
