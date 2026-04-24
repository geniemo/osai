"""Segmentation mIoU via confusion matrix accumulation.

Standard formula: IoU_c = TP_c / (TP_c + FP_c + FN_c).
- ignore_index 픽셀은 누적에서 제외 (loss와 일치).
- denom=0 클래스는 NaN → nanmean으로 평균에서 제외.
"""
from __future__ import annotations

from typing import Tuple, List

import torch


class SegMetric:
    def __init__(self, num_classes: int = 21, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self.cm = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred = pred.flatten()
        target = target.flatten()
        valid = target != self.ignore_index
        pred = pred[valid].long()
        target = target[valid].long()
        idx = target * self.num_classes + pred
        binc = torch.bincount(idx, minlength=self.num_classes ** 2)
        self.cm += binc.view(self.num_classes, self.num_classes).to(self.cm.device)

    def compute(self) -> Tuple[float, List[float]]:
        cm = self.cm.float()
        tp = cm.diag()
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp
        denom = tp + fp + fn
        iou = torch.where(denom > 0, tp / denom, torch.tensor(float("nan")))
        miou = torch.nanmean(iou).item()
        return miou, iou.tolist()
