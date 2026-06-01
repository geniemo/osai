"""Compute FID of a checkpoint's G_ema against cached real stats.

Usage:
    python eval_fid.py --ckpt runs/d256_main/ckpt_xxx.pt \
                       --stats checkpoints/fid_stats_256.npz \
                       --n 8000
"""
from __future__ import annotations
import argparse
import tempfile
from pathlib import Path
import torch
import yaml
from p2.src.networks.generator import Generator, GeneratorConfig
from p2.src.fid import dump_samples, compute_fid_against_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--stats", required=True, type=Path)
    parser.add_argument("--n", type=int, default=8000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-ema", action="store_true", default=True)
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    g_cfg_dict = ckpt["meta"]["generator_config"]
    g_cfg = GeneratorConfig.from_dict(g_cfg_dict)
    G = Generator(g_cfg).to(args.device)
    state = ckpt["G_ema_state"] if args.use_ema else ckpt["G_state"]
    G.load_state_dict(state)
    G.eval()

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        print(f"Dumping {args.n} samples to {out}...")
        dump_samples(G, args.n, G.z_dim, out, batch=args.batch, device=args.device, seed=args.seed)
        print("Computing FID...")
        fid = compute_fid_against_stats(out, args.stats)
        print(f"FID: {fid:.3f}")


if __name__ == "__main__":
    main()
