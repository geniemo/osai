"""ONNX export entrypoint. EMA model → 가중치 제거된 ONNX (10MB 이하)."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import onnx
import torch

from src.models.builder import build_model
from src.train import load_config


def export(cfg: dict, ckpt_path: str, out_path: str, opset: int = 17) -> None:
    device = torch.device("cpu")
    model = build_model(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = state.get("ema_state", state) if isinstance(state, dict) else state
    model.load_state_dict(state_dict)
    model.eval(); model.export_mode()
    dummy = torch.zeros(1, 3, 480, 640)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["input"], output_names=["logits"],
        dynamic_axes=None, opset_version=opset, do_constant_folding=True,
    )

    m = onnx.load(out_path)
    for init in m.graph.initializer:
        init.ClearField("raw_data")
        init.ClearField("float_data")
        init.ClearField("int32_data")
        init.ClearField("int64_data")
    onnx.save(m, out_path)

    size_mb = os.path.getsize(out_path) / 1e6
    assert size_mb <= 10.0, f"ONNX too large: {size_mb:.2f}MB"
    print(f"[ok] {out_path} ({size_mb:.2f}MB)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    export(cfg, args.ckpt, args.out)


if __name__ == "__main__":
    main()
