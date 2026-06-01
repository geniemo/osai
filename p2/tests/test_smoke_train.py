"""End-to-end smoke: G/D forward + losses + 1 D step + 1 G step on tiny inputs."""
import torch
from p2.src.networks.generator import Generator, GeneratorConfig
from p2.src.networks.discriminator import Discriminator, DiscriminatorConfig
from p2.src.networks.ema import EMA
from p2.src.losses import ns_logistic_g, r1_penalty, path_length_penalty
import torch.nn.functional as F


def test_smoke_one_step():
    torch.manual_seed(0)
    device = "cpu"
    g_cfg = GeneratorConfig(
        z_dim=512, w_dim=512,
        channels={4: 32, 8: 32, 16: 32, 32: 32, 64: 32, 128: 32, 256: 32},  # tiny for speed
        mapping_layers=2, mapping_lr_mul=0.01, style_mixing_prob=0.5,
    )
    d_cfg = DiscriminatorConfig(
        channels={256: 32, 128: 32, 64: 32, 32: 32, 16: 32, 8: 32, 4: 32},
        minibatch_std_group=4,
    )
    G = Generator(g_cfg).to(device)
    D = Discriminator(d_cfg).to(device)
    G_ema = EMA(G, half_life=20_000)
    optG = torch.optim.Adam(G.parameters(), lr=2e-3, betas=(0.0, 0.99))
    optD = torch.optim.Adam(D.parameters(), lr=2e-3, betas=(0.0, 0.99))
    pl_mean = torch.zeros(1, device=device)

    real = torch.randn(8, 3, 256, 256, device=device) * 0.5

    # D step
    z = torch.randn(8, 512, device=device)
    with torch.no_grad():
        fake = G(z)
    d_real = D(real)
    d_fake = D(fake.detach())
    l_d = F.softplus(-d_real).mean() + F.softplus(d_fake).mean()
    optD.zero_grad()
    l_d.backward()
    l_r1 = r1_penalty(D, real, gamma=10.0)
    l_r1.backward()
    optD.step()

    # G step
    z = torch.randn(8, 512, device=device)
    fake = G(z)
    d_fake_g = D(fake)
    l_g = ns_logistic_g(d_fake_g)
    optG.zero_grad()
    l_g.backward()
    # PL
    z_pl = torch.randn(4, 512, device=device)
    with torch.no_grad():
        w_pl = G.mapping(z_pl)
    w_pl_ws = w_pl.unsqueeze(1).expand(-1, G.num_w_layers, -1).contiguous()
    pl_penalty, pl_mean_new = path_length_penalty(G, w_pl_ws, pl_mean, decay=0.01)
    pl_mean.copy_(pl_mean_new.detach())
    (8 * 2.0 * pl_penalty).backward()
    optG.step()
    G_ema.update(G, 8)

    assert torch.isfinite(l_d) and torch.isfinite(l_g) and torch.isfinite(l_r1) and torch.isfinite(pl_penalty)
