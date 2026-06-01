"""MappingNet — z → w with PixelNorm + N FC layers + lr_mul scaling."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import EqualLinear, BiasAct


def pixel_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize per sample: x / sqrt(mean(x^2) + eps)."""
    return x * torch.rsqrt(x.square().mean(dim=1, keepdim=True) + eps)


class MappingNet(nn.Module):
    """z ~ N(0, I) → PixelNorm → FC×N (with lr_mul) → w."""

    def __init__(self, z_dim: int = 512, w_dim: int = 512,
                 num_layers: int = 8, lr_mul: float = 0.01,
                 w_avg_beta: float = 0.995):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        layers: list[nn.Module] = []
        in_dim = z_dim
        for _ in range(num_layers):
            layers.append(EqualLinear(in_dim, w_dim, lr_mul=lr_mul, bias=True))
            layers.append(BiasAct(w_dim))
            in_dim = w_dim
        self.net = nn.Sequential(*layers)
        self.register_buffer("w_avg", torch.zeros(w_dim))

    def forward(self, z: torch.Tensor, *, update_w_avg: bool = False) -> torch.Tensor:
        z = pixel_norm(z)
        x = self.net(z)
        if update_w_avg and self.training:
            with torch.no_grad():
                self.w_avg.copy_(x.detach().mean(0).lerp(self.w_avg, self.w_avg_beta))
        return x
