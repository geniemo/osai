import torch
from p2.src.networks.mapping import MappingNet
from p2.src.networks.synthesis import SynthesisNet


def test_mapping_net_forward_shape():
    net = MappingNet(z_dim=512, w_dim=512, num_layers=8, lr_mul=0.01)
    z = torch.randn(4, 512)
    w = net(z)
    assert w.shape == (4, 512)


def test_mapping_net_w_avg_buffer():
    """w_avg should be a persistent buffer for truncation trick."""
    net = MappingNet(z_dim=512, w_dim=512, num_layers=8, lr_mul=0.01)
    assert hasattr(net, "w_avg")
    assert net.w_avg.shape == (512,)


def test_mapping_net_normalizes_z():
    """Input z should be PixelNorm-normalized before mapping."""
    net = MappingNet(z_dim=512, w_dim=512, num_layers=2, lr_mul=0.01)
    z1 = torch.randn(4, 512)
    z2 = z1 * 10.0  # scaled
    w1 = net(z1)
    w2 = net(z2)
    # PixelNorm normalizes per-sample → scaled input should map to ~same w
    assert torch.allclose(w1, w2, atol=1e-4)


def test_synthesis_forward_shape():
    channels = {4: 512, 8: 512, 16: 512, 32: 512, 64: 512, 128: 256, 256: 128}
    net = SynthesisNet(channels=channels, w_dim=512)
    num_blocks = net.num_w_layers
    w = torch.randn(2, num_blocks, 512)
    rgb = net(w)
    assert rgb.shape == (2, 3, 256, 256)


def test_synthesis_w_layers_count():
    channels = {4: 512, 8: 512, 16: 512, 32: 512, 64: 512, 128: 256, 256: 128}
    net = SynthesisNet(channels=channels, w_dim=512)
    assert net.num_w_layers >= 7  # at least one w per resolution
