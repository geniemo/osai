"""Eval with TTA on VOC val. 실제 inference 성능 측정 (multi-scale + hflip).

Usage:
    python -m src.eval_tta --config src/config/default.yaml --ckpt checkpoints/best.pth
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.infer import tta_predict
from src.models.builder import build_model
from src.train import load_config
from src.utils.metrics import SegMetric


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--scales", nargs="+", type=float,
                        default=[0.5, 0.75, 1.0, 1.25, 1.5])
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    model = build_model(cfg).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    state_dict = state.get("ema_state", state) if isinstance(state, dict) else state
    model.load_state_dict(state_dict)
    model.eval()
    model.export_mode()

    voc_root = Path(cfg["data"]["voc_root"]) / "VOCdevkit" / "VOC2012"
    val_ids = (voc_root / "ImageSets/Segmentation/val.txt").read_text().split()

    metric = SegMetric(num_classes=cfg["model"]["num_classes"], ignore_index=255)
    print(f"[info] {len(val_ids)} val images, scales={args.scales} + hflip")
    for vid in tqdm(val_ids, desc="TTA eval"):
        img = Image.open(voc_root / "JPEGImages" / f"{vid}.jpg").convert("RGB")
        mask = np.array(Image.open(voc_root / "SegmentationClass" / f"{vid}.png"))
        pred = tta_predict(model, img, device, scales=tuple(args.scales))
        metric.update(torch.from_numpy(pred).long(), torch.from_numpy(mask).long())

    miou, per_class = metric.compute()
    print(f"\nTTA mIoU: {miou:.4f}")
    for i, v in enumerate(per_class):
        print(f"  class {i:2d}: {v:.4f}" if v == v else f"  class {i:2d}: NaN")


if __name__ == "__main__":
    main()
