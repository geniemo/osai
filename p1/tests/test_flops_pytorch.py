import torch
import torch.nn as nn

from src.utils.flops import count_pytorch_flops


def test_single_conv_flops():
    m = nn.Conv2d(3, 8, kernel_size=1, bias=False)
    flops = count_pytorch_flops(m, input_size=(1, 3, 4, 4))
    assert flops == 384


def test_3x3_conv_with_groups():
    m = nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=8, bias=False)
    flops = count_pytorch_flops(m, input_size=(1, 8, 4, 4))
    assert flops == 1152


def test_linear_flops():
    m = nn.Linear(16, 4, bias=False)
    flops = count_pytorch_flops(m, input_size=(1, 16))
    assert flops == 64
