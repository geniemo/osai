import math
import torch
from p2.src.networks.ops import (
    EqualLinear,
    EqualConv2d,
    ModulatedConv2d,
    AddNoise,
    BiasAct,
)


def test_equal_linear_runtime_scale():
    """Linear weight is N(0, 1/sqrt(in)) at init, runtime scale equalizes LR."""
    layer = EqualLinear(in_dim=64, out_dim=32, lr_mul=1.0)
    x = torch.randn(8, 64)
    y = layer(x)
    assert y.shape == (8, 32)
    expected_runtime_scale = 1.0 / math.sqrt(64)
    assert math.isclose(layer.runtime_scale, expected_runtime_scale, rel_tol=1e-5)


def test_equal_linear_lr_mul_scaling():
    """lr_mul scales both runtime weight scale and bias scale."""
    layer = EqualLinear(in_dim=64, out_dim=32, lr_mul=0.01)
    assert math.isclose(layer.runtime_scale, (1.0 / math.sqrt(64)) * 0.01, rel_tol=1e-5)


def test_equal_conv2d_forward_shape():
    layer = EqualConv2d(in_ch=16, out_ch=32, kernel=3)
    x = torch.randn(4, 16, 8, 8)
    y = layer(x)
    assert y.shape == (4, 32, 8, 8)


def test_modulated_conv_forward_shape():
    layer = ModulatedConv2d(in_ch=16, out_ch=32, kernel=3, style_dim=64, demodulate=True)
    x = torch.randn(4, 16, 8, 8)
    w = torch.randn(4, 64)
    y = layer(x, w)
    assert y.shape == (4, 32, 8, 8)


def test_modulated_conv_upsample_shape():
    layer = ModulatedConv2d(in_ch=16, out_ch=32, kernel=3, style_dim=64, up=True)
    x = torch.randn(4, 16, 8, 8)
    w = torch.randn(4, 64)
    y = layer(x, w)
    assert y.shape == (4, 32, 16, 16)


def test_modulated_conv_demodulation_normalizes():
    """After demodulation, per-sample output channels should have ~unit variance."""
    torch.manual_seed(0)
    layer = ModulatedConv2d(in_ch=32, out_ch=32, kernel=3, style_dim=64, demodulate=True)
    x = torch.randn(16, 32, 4, 4)
    w = torch.randn(16, 64)
    y = layer(x, w)
    var = y.var(dim=[0, 2, 3]).mean()
    assert 0.3 < var.item() < 3.0


def test_add_noise_per_pixel_learnable():
    layer = AddNoise(channels=16)
    x = torch.zeros(4, 16, 8, 8)
    y = layer(x)
    assert y.shape == x.shape
    assert layer.weight.shape == (1, 16, 1, 1) or layer.weight.numel() == 1


def test_bias_act_leaky_relu_gain():
    """BiasAct = (x + bias) → leakyReLU(0.2) → * sqrt(2)."""
    layer = BiasAct(channels=4)
    layer.bias.data.zero_()
    x = torch.tensor([[1.0, -1.0, 0.5, -0.5]]).view(1, 4, 1, 1)
    y = layer(x)
    expected = torch.tensor([[1.0 * math.sqrt(2), -1.0 * 0.2 * math.sqrt(2),
                              0.5 * math.sqrt(2), -0.5 * 0.2 * math.sqrt(2)]]).view(1, 4, 1, 1)
    assert torch.allclose(y, expected, atol=1e-5)
