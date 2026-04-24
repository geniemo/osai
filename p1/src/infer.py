"""Inference with multi-scale TTA + hflip. input_dir/*.jpg → output_dir/*.png."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.models.builder import build_model
from src.train import load_config


def _round_up(x: float, multiple: int = 32) -> int:
    return int(math.ceil(x / multiple) * multiple)


def _preprocess(img: Image.Image, target_size: tuple[int, int]) -> torch.Tensor:
    img = img.resize(target_size, Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()


@torch.no_grad()
def tta_predict(model, img_pil: Image.Image, device, scales=(0.5, 0.75, 1.0, 1.25, 1.5)) -> np.ndarray:
    W, H = img_pil.size
    accum = torch.zeros(21, H, W, device=device)
    for scale in scales:
        Hs = _round_up(H * scale, 32); Ws = _round_up(W * scale, 32)
        img_t = _preprocess(img_pil, (Ws, Hs)).to(device)
        for hflip in (False, True):
            x = torch.flip(img_t, dims=[-1]) if hflip else img_t
            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = model(x)
            if isinstance(logits, tuple): logits = logits[0]
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
            logits = logits.float().softmax(dim=1)
            if hflip: logits = torch.flip(logits, dims=[-1])
            accum += logits.squeeze(0)
    pred = accum.argmax(dim=0).to(torch.uint8).cpu().numpy()
    return pred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-tta", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    state_dict = state.get("ema_state", state) if isinstance(state, dict) else state
    model.load_state_dict(state_dict)
    model.eval(); model.export_mode()

    inp = Path(args.input); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    files = sorted(inp.glob("*.jpg"))
    scales = (1.0,) if args.no_tta else (0.5, 0.75, 1.0, 1.25, 1.5)
    for f in tqdm(files):
        img = Image.open(f).convert("RGB")
        pred = tta_predict(model, img, device, scales=scales)
        Image.fromarray(pred, mode="L").save(out / f"{f.stem}.png", format="PNG")


if __name__ == "__main__":
    main()
