"""Boundary-aware Cross-Entropy loss.

객체 경계 픽셀에 더 큰 weight를 줘서 thin objects (chair, bicycle 등)에서
boundary 정확도 개선. v2.D ablation에 사용.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def boundary_mask(target: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Detect class boundary pixels via dilation/erosion.

    Args:
        target: (N, H, W) long tensor (class IDs).
        kernel_size: neighborhood size.

    Returns:
        (N, H, W) bool tensor — True at class boundaries.
    """
    t = target.float().unsqueeze(1)  # (N, 1, H, W)
    pad = kernel_size // 2
    pool_max = F.max_pool2d(t, kernel_size=kernel_size, stride=1, padding=pad)
    pool_min = -F.max_pool2d(-t, kernel_size=kernel_size, stride=1, padding=pad)
    return (pool_max != pool_min).squeeze(1)


def boundary_weighted_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 255,
    boundary_alpha: float = 5.0,
    kernel: int = 3,
) -> torch.Tensor:
    """CE with higher weight at class boundary pixels.

    Args:
        logits: (N, C, H, W).
        target: (N, H, W) long.
        ignore_index: pixel value to skip.
        boundary_alpha: weight multiplier for boundary pixels.
        kernel: boundary detection neighborhood.
    """
    N, C, H, W = logits.shape
    log_probs = F.log_softmax(logits, dim=1)
    valid = target != ignore_index
    target_safe = target.clone()
    target_safe[~valid] = 0
    target_oh = F.one_hot(target_safe, num_classes=C).permute(0, 3, 1, 2).float()
    nll = -(target_oh * log_probs).sum(dim=1)  # (N, H, W)

    bmask = boundary_mask(target, kernel_size=kernel)
    alpha = torch.tensor(float(boundary_alpha), device=logits.device, dtype=nll.dtype)
    one = torch.tensor(1.0, device=logits.device, dtype=nll.dtype)
    weights = torch.where(bmask, alpha, one) * valid.to(nll.dtype)

    loss_sum = (nll * weights).sum()
    weight_sum = weights.sum().clamp(min=1.0)
    return loss_sum / weight_sum
