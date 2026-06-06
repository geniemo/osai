"""Train StyleGAN2 D-256.

Modes:
    python train.py --config configs/d256.yaml
    python train.py --config configs/d256.yaml --resume runs/d256_main/ckpt_xxx.pt
"""
from __future__ import annotations
import argparse
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import yaml
from torch.utils.data import DataLoader

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    wandb = None
    _HAS_WANDB = False

from p2.src.augment import diff_augment
from p2.src.dataset import ZipImageDataset, infinite_loader
from p2.src.losses import ns_logistic_g, r1_penalty, path_length_penalty
from p2.src.networks.discriminator import Discriminator, DiscriminatorConfig
from p2.src.networks.ema import EMA
from p2.src.networks.generator import Generator, GeneratorConfig
from p2.src.utils import async_save_checkpoint, save_checkpoint_atomic, set_seed, ThroughputMeter


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def save_sample_grid(G: torch.nn.Module, sample_z: torch.Tensor, out_path: Path, nrow: int = 8) -> None:
    G.eval()
    fake = G(sample_z)
    x = ((fake + 1.0) / 2.0).clamp(0.0, 1.0)
    grid = vutils.make_grid(x, nrow=nrow, padding=2)
    vutils.save_image(grid, out_path)


def build_checkpoint(*, images_seen, step, G, D, G_ema, optG, optD, pl_mean,
                    g_cfg, d_cfg, training_cfg, wandb_run_id):
    return {
        "images_seen": images_seen,
        "step": step,
        "G_state": G.state_dict(),
        "D_state": D.state_dict(),
        "G_ema_state": G_ema.state_dict(),
        "optG_state": optG.state_dict(),
        "optD_state": optD.state_dict(),
        "pl_mean": pl_mean.cpu().clone(),
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
        },
        "wandb_run_id": wandb_run_id,
        "meta": {
            "generator_config": asdict(g_cfg),
            "discriminator_config": asdict(d_cfg),
            "training_config": training_cfg,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--total-images", type=int, default=None)
    parser.add_argument("--train-zip", type=Path, default=None,
                        help="Override cfg['training']['train_zip'].")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Override cfg['out']['run_dir'].")
    parser.add_argument("--new-wandb-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    if args.total_images is not None:
        train_cfg["total_images"] = args.total_images
    if args.train_zip is not None:
        train_cfg["train_zip"] = str(args.train_zip)
    if args.run_dir is not None:
        cfg["out"]["run_dir"] = str(args.run_dir)

    set_seed(train_cfg["seed"])
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    g_cfg = GeneratorConfig.from_dict(cfg["generator"])
    d_cfg = DiscriminatorConfig.from_dict(cfg["discriminator"])
    G = Generator(g_cfg).to(device)
    D = Discriminator(d_cfg).to(device)
    n_g = sum(p.numel() for p in G.parameters())
    n_d = sum(p.numel() for p in D.parameters())
    print(f"Generator: {n_g/1e6:.2f}M params")
    print(f"Discriminator: {n_d/1e6:.2f}M params")
    assert n_g < 40_000_000, f"G has {n_g/1e6:.2f}M params > 40M hard threshold"

    optG = torch.optim.Adam(G.parameters(), lr=train_cfg["lr_g"],
                            betas=(train_cfg["beta1"], train_cfg["beta2"]), eps=1e-8,
                            weight_decay=train_cfg["weight_decay"])
    optD = torch.optim.Adam(D.parameters(), lr=train_cfg["lr_d"],
                            betas=(train_cfg["beta1"], train_cfg["beta2"]), eps=1e-8,
                            weight_decay=train_cfg["weight_decay"])

    G_ema = EMA(G, half_life=train_cfg["ema_kimg"] * 1000)
    G_ema.shadow.to(device)

    dataset = ZipImageDataset(train_cfg["train_zip"], flip=train_cfg["flip"])
    print(f"Dataset: {len(dataset)} images")
    loader = DataLoader(dataset, batch_size=train_cfg["batch_size"], shuffle=True,
                        num_workers=train_cfg["num_workers"], pin_memory=device == "cuda",
                        persistent_workers=train_cfg["num_workers"] > 0,
                        prefetch_factor=2 if train_cfg["num_workers"] > 0 else None,
                        drop_last=True)
    inf_loader = infinite_loader(loader)

    sample_gen = torch.Generator(device="cpu").manual_seed(train_cfg["sample_seed"])
    sample_z = torch.randn(train_cfg["sample_n"], g_cfg.z_dim, generator=sample_gen).to(device)

    run_dir = Path(cfg["out"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    pl_mean = torch.zeros(1, device=device)

    images_seen = 0
    step = 0
    wandb_run_id = None

    if args.resume is not None:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        G.load_state_dict(ckpt["G_state"])
        D.load_state_dict(ckpt["D_state"])
        G_ema.load_state_dict(ckpt["G_ema_state"])
        optG.load_state_dict(ckpt["optG_state"])
        optD.load_state_dict(ckpt["optD_state"])
        pl_mean.copy_(ckpt["pl_mean"].to(device))
        images_seen = ckpt["images_seen"]
        step = ckpt["step"]
        wandb_run_id = None if args.new_wandb_run else ckpt.get("wandb_run_id")
        rng = ckpt.get("rng_state", {})
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"].cpu())
        if torch.cuda.is_available() and rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
        if rng.get("numpy") is not None:
            np.random.set_state(rng["numpy"])

    wandb_cfg = cfg.get("wandb", {})
    wandb_mode = wandb_cfg.get("mode", "online") if _HAS_WANDB else "disabled"
    run = None
    if wandb_mode != "disabled":
        init_kwargs = {"project": wandb_cfg["project"], "name": wandb_cfg.get("name"),
                       "mode": wandb_mode, "config": cfg}
        if wandb_run_id is not None:
            init_kwargs["id"] = wandb_run_id
            init_kwargs["resume"] = "must"
        run = wandb.init(**init_kwargs)
        wandb_run_id = run.id

    total_images = train_cfg["total_images"]
    z_dim = g_cfg.z_dim
    r1_gamma = train_cfg["r1_gamma"]
    r1_lazy = train_cfg["r1_lazy_every"]
    pl_weight = train_cfg["pl_weight"]
    pl_lazy = train_cfg["pl_lazy_every"]
    pl_decay = train_cfg["pl_decay"]
    log_every = train_cfg.get("log_every", 50)
    ckpt_every = train_cfg["ckpt_every"]
    augment_policy = train_cfg.get("augment", "")
    precision = train_cfg.get("precision", "fp32")
    use_amp = precision in ("bf16", "fp16")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
    print(f"Precision: {precision}, augment: {augment_policy!r}")

    last_ckpt = images_seen
    save_threads: list[threading.Thread] = []
    meter = ThroughputMeter()
    last_r1 = None
    last_pl = None

    print(f"Training: images_seen={images_seen} → {total_images} (bs={train_cfg['batch_size']}, device={device})")

    while images_seen < total_images:
        real = next(inf_loader).to(device, non_blocking=True)
        b = real.size(0)

        # D step
        z = torch.randn(b, z_dim, device=device)
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
            with torch.no_grad():
                fake = G(z)
            d_real = D(diff_augment(real, augment_policy))
            d_fake = D(diff_augment(fake.detach(), augment_policy))
            l_d_real = F.softplus(-d_real).mean()
            l_d_fake = F.softplus(d_fake).mean()
            l_d = l_d_real + l_d_fake
        optD.zero_grad(set_to_none=True)
        l_d.backward()

        if (step + 1) % r1_lazy == 0:
            # R1 memory peak: halve batch to fit small GPUs (e.g. L4 22GB).
            # StyleGAN2-ADA does the same; expectation is unchanged (mean over batch).
            r1_b = max(1, b // 2)
            l_r1 = r1_lazy * r1_penalty(D, diff_augment(real.float()[:r1_b], augment_policy), gamma=r1_gamma)
            l_r1.backward()
            last_r1 = float(l_r1.item()) / r1_lazy

        optD.step()

        # G step
        z = torch.randn(b, z_dim, device=device)
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
            fake = G(z)
            d_fake_g = D(diff_augment(fake, augment_policy))
            l_g = ns_logistic_g(d_fake_g)
        optG.zero_grad(set_to_none=True)
        l_g.backward()

        if (step + 1) % pl_lazy == 0:
            # PL on fresh w batch (fp32 for stability)
            z_pl = torch.randn(b // 2, z_dim, device=device)
            with torch.no_grad():
                w_pl = G.mapping(z_pl)
            w_pl_ws = w_pl.unsqueeze(1).expand(-1, G.num_w_layers, -1).contiguous()
            pl_penalty, pl_mean_new = path_length_penalty(G, w_pl_ws, pl_mean, decay=pl_decay)
            pl_mean.copy_(pl_mean_new.detach())
            (pl_lazy * pl_weight * pl_penalty).backward()
            last_pl = float(pl_penalty.item())

        optG.step()
        G_ema.update(G, b)

        images_seen += b
        meter.add(b)
        step += 1

        if step % log_every == 0:
            log = {
                "images_seen": images_seen,
                "throughput/imgs_per_sec": meter.rate(),
                "loss/D_total": float(l_d.item()),
                "loss/D_real": float(l_d_real.item()),
                "loss/D_fake": float(l_d_fake.item()),
                "loss/G": float(l_g.item()),
                "D_out/real_mean": float(d_real.float().mean().item()),
                "D_out/fake_mean": float(d_fake.float().mean().item()),
                "pl_mean_path_length": float(pl_mean.item()),
            }
            if last_r1 is not None:
                log["loss/R1"] = last_r1
            if last_pl is not None:
                log["loss/PL"] = last_pl
            if wandb_mode != "disabled":
                wandb.log(log, step=step)
            else:
                print(f"step={step} imgs={images_seen} thr={meter.rate():.1f}img/s "
                      f"l_d={l_d.item():.3f} l_g={l_g.item():.3f}")
            meter.reset()

        if images_seen - last_ckpt >= ckpt_every:
            ckpt = build_checkpoint(
                images_seen=images_seen, step=step, G=G, D=D, G_ema=G_ema,
                optG=optG, optD=optD, pl_mean=pl_mean,
                g_cfg=g_cfg, d_cfg=d_cfg, training_cfg=train_cfg, wandb_run_id=wandb_run_id,
            )
            ckpt_path = run_dir / f"ckpt_{images_seen:09d}.pt"
            grid_path = samples_dir / f"grid_{images_seen:09d}.png"
            save_threads = [t for t in save_threads if t.is_alive()]
            save_threads.append(async_save_checkpoint(ckpt_path, ckpt))
            save_sample_grid(G_ema.shadow, sample_z, grid_path, nrow=8)
            if wandb_mode != "disabled":
                wandb.log({"samples/grid": wandb.Image(str(grid_path))}, step=step)
            print(f"[ckpt+grid] {ckpt_path.name} / {grid_path.name}")
            last_ckpt = images_seen

    print("Training complete. Saving final ckpt...")
    final = build_checkpoint(
        images_seen=images_seen, step=step, G=G, D=D, G_ema=G_ema,
        optG=optG, optD=optD, pl_mean=pl_mean,
        g_cfg=g_cfg, d_cfg=d_cfg, training_cfg=train_cfg, wandb_run_id=wandb_run_id,
    )
    save_checkpoint_atomic(run_dir / "final.pt", final)
    for t in save_threads:
        t.join()
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
