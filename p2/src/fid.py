"""FID measurement wrapper around pytorch-fid."""
from __future__ import annotations
import subprocess
import tempfile
import shutil
from pathlib import Path
import torch
from PIL import Image
import numpy as np


@torch.no_grad()
def dump_samples(G, n_samples: int, z_dim: int, out_dir: Path,
                 *, batch: int = 16, device: str = "cuda", seed: int = 0) -> None:
    """Generate n_samples images via G(z) and save as PNG in out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    G.eval()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    idx = 0
    while idx < n_samples:
        b = min(batch, n_samples - idx)
        z = torch.randn(b, z_dim, generator=gen).to(device)
        fake = G(z)
        fake = ((fake + 1.0) / 2.0).clamp(0.0, 1.0).cpu().mul(255).byte()
        fake = fake.permute(0, 2, 3, 1).numpy()
        for i in range(b):
            Image.fromarray(fake[i]).save(out_dir / f"{idx + i:06d}.png")
        idx += b


def compute_fid_against_stats(samples_dir: Path, stats_path: Path) -> float:
    """Run pytorch-fid CLI to compute FID(samples_dir, stats_path).

    Raises RuntimeError with full stdout+stderr if pytorch-fid exits non-zero
    or if FID line is missing (helps debug "subprocess returned 1" silently).
    """
    result = subprocess.run(
        ["python", "-m", "pytorch_fid", str(samples_dir), str(stats_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"pytorch-fid exited {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    for line in result.stdout.splitlines():
        if line.startswith("FID:"):
            return float(line.split()[1])
    raise RuntimeError(f"FID parse failed:\n{result.stdout}")


def build_stats_cache(real_dir: Path, stats_path: Path) -> None:
    """Cache Inception stats for real images (run once).

    pytorch-fid CLI quirk: --save-stats must come *after* both positional
    paths. The form `python -m pytorch_fid <dir> --save-stats <stats>` errors
    with 'the following arguments are required: path'.
    """
    subprocess.run(
        ["python", "-m", "pytorch_fid", str(real_dir), str(stats_path), "--save-stats"],
        check=True,
    )
