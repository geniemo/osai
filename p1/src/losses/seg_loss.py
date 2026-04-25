"""Segmentation loss = CE(main) + dice_weight × Dice(main) + aux_weight × CE(aux).

ignore_index=255는 CE는 자동, Dice는 수동 mask로 제외.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Per-class dice (background 포함), ignore mask 적용, 평균."""

    def __init__(self, num_classes: int, ignore_index: int = 255, eps: float = 1e-6) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = (target != self.ignore_index).unsqueeze(1).float()
        target_clamped = target.clone()
        target_clamped[target == self.ignore_index] = 0
        target_oh = F.one_hot(target_clamped, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        target_oh = target_oh * valid
        probs = F.softmax(logits, dim=1) * valid

        dims = (0, 2, 3)
        intersect = (probs * target_oh).sum(dim=dims)
        denom = probs.sum(dim=dims) + target_oh.sum(dim=dims)
        dice = (2 * intersect + self.eps) / (denom + self.eps)
        return 1.0 - dice.mean()


class SegLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 21,
        ignore_index: int = 255,
        dice_weight: float = 0.5,
        aux_weight: float = 0.4,
        lovasz_weight: float = 0.0,
        boundary_weight: float = 0.0,
        boundary_alpha: float = 5.0,
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="sum")
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.dice_weight = dice_weight
        self.aux_weight = aux_weight
        self.lovasz_weight = lovasz_weight
        self.boundary_weight = boundary_weight
        self.boundary_alpha = boundary_alpha

    def _ce(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_valid = (target != self.ignore_index).sum().clamp(min=1)
        return self.ce(logits, target) / n_valid

    def forward(self, main_logits: torch.Tensor, aux_logits: Optional[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        loss = self._ce(main_logits, target)
        if self.dice_weight > 0:
            loss = loss + self.dice_weight * self.dice(main_logits, target)
        if self.lovasz_weight > 0:
            from src.losses.lovasz import lovasz_softmax
            loss = loss + self.lovasz_weight * lovasz_softmax(
                main_logits, target, ignore_index=self.ignore_index
            )
        if self.boundary_weight > 0:
            from src.losses.boundary import boundary_weighted_ce
            loss = loss + self.boundary_weight * boundary_weighted_ce(
                main_logits, target,
                ignore_index=self.ignore_index,
                boundary_alpha=self.boundary_alpha,
            )
        if aux_logits is not None and self.aux_weight > 0:
            loss = loss + self.aux_weight * self._ce(aux_logits, target)
        return loss
