import torch
from p2.src.networks.mapping import MappingNet


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
