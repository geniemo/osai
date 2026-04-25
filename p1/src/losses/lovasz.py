"""Lovász-Softmax loss (Berman et al. 2018).

mIoU의 differentiable smooth surrogate. CE와 결합해 IoU를 직접 최적화.
Reference: https://arxiv.org/abs/1705.08790
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Lovász extension gradient (paper Algorithm 1)."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(
    probs: torch.Tensor,
    labels: torch.Tensor,
    classes: str = "present",
    ignore_index: int = 255,
) -> torch.Tensor:
    """Lovász-Softmax on flattened (P, C) probs + (P,) labels."""
    valid = labels != ignore_index
    probs = probs[valid]
    labels = labels[valid]
    if probs.numel() == 0:
        return probs.sum() * 0.0

    losses = []
    if classes == "present":
        class_set = torch.unique(labels).tolist()
    else:
        class_set = list(range(probs.size(1)))

    for c in class_set:
        if c >= probs.size(1):
            continue
        fg = (labels == c).float()
        if classes == "present" and fg.sum() == 0:
            continue
        errors = (fg - probs[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))

    if not losses:
        return probs.sum() * 0.0
    return torch.stack(losses).mean()


def lovasz_softmax(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Lovász-Softmax loss for segmentation.

    Args:
        logits: (N, C, H, W) raw logits.
        labels: (N, H, W) long, class IDs in [0, C-1] or ignore_index.
    """
    probs = F.softmax(logits, dim=1)
    N, C, H, W = probs.shape
    probs = probs.permute(0, 2, 3, 1).reshape(-1, C)
    labels = labels.reshape(-1)
    return lovasz_softmax_flat(probs, labels, classes="present", ignore_index=ignore_index)
