"""Eval entrypoint: model.pth (or ckpt) → val mIoU + per-class IoU."""
from __future__ import annotations

import argparse

import torch

from src.data.builder import build_dataloaders
from src.models.builder import build_model
from src.train import evaluate, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    cfg["data"]["stage"] = 2  # val만 필요
    _, val_loader = build_dataloaders(cfg)

    model = build_model(cfg).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    state_dict = state.get("ema_state", state) if isinstance(state, dict) else state
    model.load_state_dict(state_dict)
    miou, per_class = evaluate(model, val_loader, device, num_classes=cfg["model"]["num_classes"])
    print(f"mIoU: {miou:.4f}")
    for i, v in enumerate(per_class):
        print(f"  class {i:2d}: {v:.4f}" if v == v else f"  class {i:2d}: NaN")


if __name__ == "__main__":
    main()
