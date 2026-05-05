"""사전 FLOPs 검증 — yaml의 모델 빌드해 ONNX export 후 FLOPs 측정.

Usage:
    uv run python -m src.check_flops --config src/config/colab_mn3_s2.yaml --max-gflops 15.0
"""
from __future__ import annotations

import argparse
import sys

import torch
import yaml

from src.models.builder import build_model
from src.utils.flops import count_onnx_flops


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--max-gflops", type=float, default=None,
                   help="이 값 초과 시 비정상 종료 (S_FLOPs 임계값 검증).")
    p.add_argument("--input-shape", nargs=4, type=int, default=[1, 3, 480, 640])
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    m = build_model(cfg)
    m.eval()
    m.export_mode()

    onnx_path = "/tmp/check_flops.onnx"
    dummy = torch.zeros(*args.input_shape)
    torch.onnx.export(
        m, dummy, onnx_path,
        opset_version=18, do_constant_folding=True,
        input_names=["input"], output_names=["output"],
    )

    mac, _ = count_onnx_flops(onnx_path, tuple(args.input_shape))
    gflops = mac * 2 / 1e9
    n_params = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"GFLOPs: {gflops:.3f}")
    print(f"GMACs:  {mac/1e9:.3f}")
    print(f"Params: {n_params:.2f}M")

    if args.max_gflops is not None:
        if gflops > args.max_gflops:
            print(f"FAIL: {gflops:.3f} > {args.max_gflops} (S_FLOPs threshold violated)")
            sys.exit(1)
        else:
            print(f"OK: {gflops:.3f} <= {args.max_gflops} (margin {args.max_gflops - gflops:.3f})")


if __name__ == "__main__":
    main()
