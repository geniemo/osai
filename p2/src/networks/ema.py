"""Exponential moving average of Generator weights, image-count half-life."""
from __future__ import annotations
import copy
import torch
import torch.nn as nn


class EMA:
    """decay = 0.5 ** (batch_size / half_life).
    Call `update(G, batch_size)` after every G step. Shadow is eval+no_grad."""

    def __init__(self, G: nn.Module, half_life: int = 20_000):
        self.shadow = copy.deepcopy(G).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self.half_life = max(half_life, 1)

    @torch.no_grad()
    def update(self, G: nn.Module, batch_size: int) -> None:
        decay = 0.5 ** (batch_size / self.half_life)
        for sp, p in zip(self.shadow.parameters(), G.parameters()):
            sp.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
        for sb, b in zip(self.shadow.buffers(), G.buffers()):
            sb.copy_(b)

    def state_dict(self) -> dict:
        return self.shadow.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.shadow.load_state_dict(state)
