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
        total_mac, breakdown = count_onnx_flops(args.onnx, (1, 3, 480, 640))
        total_flop = total_mac * 2
        print(f"[ONNX] {args.onnx}: {total_flop/1e9:.2f} GFLOPs (= {total_mac/1e9:.2f} GMACs ×2)")
        for op, f in sorted(breakdown.items(), key=lambda kv: -kv[1]):
            if f > 0:
                print(f"    {op}: {f*2/1e9:.3f} GFLOP")

    if args.config and args.ckpt:
        import torch
        from src.models.builder import build_model
        cfg = load_config(args.config)
        m = build_model(cfg)
        state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        sd = state.get("ema_state", state) if isinstance(state, dict) else state
        m.load_state_dict(sd)
        m.export_mode().eval()
        f_py_mac = count_pytorch_flops(m, (1, 3, 480, 640))
        print(f"[PyTorch] {f_py_mac*2/1e9:.2f} GFLOPs (= {f_py_mac/1e9:.2f} GMACs ×2, sanity check)")


if __name__ == "__main__":
    main()
