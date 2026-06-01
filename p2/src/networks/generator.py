"""Generator = MappingNet + SynthesisNet with optional style mixing."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
import torch.nn as nn

from .mapping import MappingNet
from .synthesis import SynthesisNet


@dataclass
class GeneratorConfig:
    z_dim: int
    w_dim: int
    channels: dict[int, int]
    mapping_layers: int = 8
    mapping_lr_mul: float = 0.01
    style_mixing_prob: float = 0.9

    def __post_init__(self):
        self.channels = {int(k): int(v) for k, v in self.channels.items()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GeneratorConfig":
        return cls(
            z_dim=d["z_dim"],
            w_dim=d.get("w_dim", d["z_dim"]),
            channels=d["channels"],
            mapping_layers=d.get("mapping_layers", 8),
            mapping_lr_mul=d.get("mapping_lr_mul", 0.01),
            style_mixing_prob=d.get("style_mixing_prob", 0.9),
        )


class Generator(nn.Module):
    def __init__(self, cfg: GeneratorConfig):
        super().__init__()
        self.cfg = cfg
        self.z_dim = cfg.z_dim
        self.w_dim = cfg.w_dim
        self.mapping = MappingNet(z_dim=cfg.z_dim, w_dim=cfg.w_dim,
                                  num_layers=cfg.mapping_layers, lr_mul=cfg.mapping_lr_mul)
        self.synthesis = SynthesisNet(channels=cfg.channels, w_dim=cfg.w_dim)
        self.num_w_layers = self.synthesis.num_w_layers
        self.style_mixing_prob = cfg.style_mixing_prob

    def forward(self, z: torch.Tensor, *, update_w_avg: bool = False,
                style_mixing: bool | None = None) -> torch.Tensor:
        B = z.shape[0]
        w = self.mapping(z, update_w_avg=update_w_avg)
        # Broadcast w to all layers
        ws = w.unsqueeze(1).expand(-1, self.num_w_layers, -1).contiguous()

        do_mix = style_mixing if style_mixing is not None else (self.training and self.style_mixing_prob > 0)
        if do_mix and torch.rand(()) < self.style_mixing_prob:
            z2 = torch.randn(B, self.z_dim, device=z.device, dtype=z.dtype)
            w2 = self.mapping(z2)
            cutoff = int(torch.randint(1, self.num_w_layers, ()).item())
            ws[:, cutoff:] = w2.unsqueeze(1).expand(-1, self.num_w_layers - cutoff, -1)
        return self.synthesis(ws)

    def forward_w(self, w: torch.Tensor) -> torch.Tensor:
        """Skip mapping — for PL regularization which needs w directly."""
        if w.ndim == 2:
            w = w.unsqueeze(1).expand(-1, self.num_w_layers, -1).contiguous()
        return self.synthesis(w)
