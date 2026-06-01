"""GAN losses for StyleGAN2-style training: NS logistic + R1 + Path Length."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def ns_logistic_d(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    return F.softplus(-d_real).mean() + F.softplus(d_fake).mean()


def ns_logistic_g(d_fake: torch.Tensor) -> torch.Tensor:
    return F.softplus(-d_fake).mean()


def r1_penalty(D: nn.Module, x_real: torch.Tensor, gamma: float = 10.0) -> torch.Tensor:
    x = x_real.detach().requires_grad_(True)
    d = D(x).sum()
    (grad,) = torch.autograd.grad(d, x, create_graph=True)
    return (gamma / 2.0) * grad.pow(2).flatten(1).sum(dim=1).mean()


def path_length_penalty(
    G,
    w: torch.Tensor,
    pl_mean: torch.Tensor,
    *,
    decay: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """StyleGAN2 path length regularization.

    Args:
        G: Generator (callable as G.forward_w(w) → image).
        w: w broadcasted to all layers, shape (B, num_w, w_dim). Must require_grad after detach.
        pl_mean: scalar buffer, EMA of path lengths.
        decay: EMA decay for pl_mean.

    Returns:
        (penalty, pl_mean_updated).
    """
    w = w.detach().requires_grad_(True)
    img = G.forward_w(w)
    noise = torch.randn_like(img) / math.sqrt(img.shape[-1] * img.shape[-2])
    out_sum = (img * noise).sum()
    (pl_grads,) = torch.autograd.grad(out_sum, w, create_graph=True)
    pl_lengths = pl_grads.square().sum(dim=2).mean(dim=1).sqrt()  # (B,)
    pl_mean_new = pl_mean.lerp(pl_lengths.detach().mean(), decay)
    penalty = (pl_lengths - pl_mean_new).square().mean()
    return penalty, pl_mean_new
