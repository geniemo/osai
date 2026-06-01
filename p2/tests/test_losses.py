import torch
import torch.nn as nn
from p2.src.losses import ns_logistic_g, ns_logistic_d, r1_penalty, path_length_penalty


def test_ns_logistic_g_finite():
    d_fake = torch.randn(4)
    loss = ns_logistic_g(d_fake)
    assert torch.isfinite(loss)


def test_ns_logistic_d_finite():
    d_real = torch.randn(4)
    d_fake = torch.randn(4)
    loss = ns_logistic_d(d_real, d_fake)
    assert torch.isfinite(loss)


def test_r1_penalty_runs():
    class TinyD(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 1, 3, padding=1)
        def forward(self, x):
            return self.conv(x).mean(dim=[1, 2, 3], keepdim=False)
    D = TinyD()
    x = torch.randn(4, 3, 16, 16)
    loss = r1_penalty(D, x, gamma=10.0)
    assert torch.isfinite(loss)
    assert loss.requires_grad
