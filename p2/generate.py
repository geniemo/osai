"""Generate sample grid from a checkpoint."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
import torchvision.utils as vutils
from p2.src.networks.generator import Generator, GeneratorConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--out", default=Path("samples.png"), type=Path)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--nrow", type=int, default=8)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    g_cfg = GeneratorConfig.from_dict(ckpt["meta"]["generator_config"])
    G = Generator(g_cfg).to(args.device)
    state = ckpt["G_state"] if args.no_ema else ckpt["G_ema_state"]
    G.load_state_dict(state)
    G.eval()

    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    z = torch.randn(args.n, G.z_dim, generator=gen).to(args.device)
    with torch.no_grad():
        fake = G(z, style_mixing=False)
    x = ((fake + 1.0) / 2.0).clamp(0.0, 1.0)
    grid = vutils.make_grid(x, nrow=args.nrow, padding=2)
    vutils.save_image(grid, args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
