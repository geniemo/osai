"""SegmentationModel: backbone + neck + head + (optional) aux head.

forward 반환:
- training: (main_logits, aux_logits) tuple
- inference (after export_mode()): main_logits only
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import torch.nn as nn
from torch import Tensor


class SegmentationModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        head: nn.Module,
        aux_head: Optional[nn.Module] = None,
        aux_in_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.aux_head = aux_head
        self._export = False

    def export_mode(self) -> "SegmentationModel":
        """ONNX export 직전 호출. aux head 비활성화 + main logits만 반환."""
        self._export = True
        return self

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        h, w = x.shape[2:]
        low, high = self.backbone(x)
        feat = self.neck(high)
        main_logits = self.head(feat, low, output_size=(h, w))
        if self._export or self.aux_head is None or not self.training:
            return main_logits
        aux_logits = self.aux_head(high, output_size=(h, w))
        return main_logits, aux_logits
