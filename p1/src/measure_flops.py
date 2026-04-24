"""FLOPs 측정. PyTorch counter (학습 중 sanity) + ONNX counter (채점 ground truth)."""
from __future__ import annotations

import argparse

from src.train import load_config
from src.utils.flops import count_pytorch_flops, count_onnx_flops


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--onnx", default=None)
    args = p.parse_args()

    if args.onnx:
        total, breakdown = count_onnx_flops(args.onnx, (1, 3, 480, 640))
        print(f"[ONNX] {args.onnx}: {total/1e9:.2f} GFLOPs")
        for op, f in sorted(breakdown.items(), key=lambda kv: -kv[1]):
            if f > 0:
                print(f"    {op}: {f/1e9:.3f} G")

    if args.config and args.ckpt:
        import torch
        from src.models.builder import build_model
        cfg = load_config(args.config)
        m = build_model(cfg)
        state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        sd = state.get("ema_state", state) if isinstance(state, dict) else state
        m.load_state_dict(sd)
        m.export_mode().eval()
        f_py = count_pytorch_flops(m, (1, 3, 480, 640))
        print(f"[PyTorch] {f_py/1e9:.2f} GFLOPs (sanity check)")


if __name__ == "__main__":
    main()
