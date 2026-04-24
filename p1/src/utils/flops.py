"""FLOPs counters: PyTorch (forward hooks) + ONNX (graph traversal).

PyTorch counter는 Conv2d/Linear의 MAC만 카운트 (×2 안 함).
w4/utils/compute_utils.py 기반, w4 컨벤션 보존.

ONNX counter (count_onnx_flops)는 다음 task에서 추가.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


def _conv2d_flops(layer: nn.Conv2d, output: torch.Tensor) -> int:
    n, cout, hout, wout = output.shape
    kernel_ops = layer.kernel_size[0] * layer.kernel_size[1] * (layer.in_channels // layer.groups)
    return int(n * cout * hout * wout * kernel_ops)


def _linear_flops(layer: nn.Linear, output: torch.Tensor) -> int:
    n = output.shape[0]
    return int(n * layer.in_features * layer.out_features)


def count_pytorch_flops(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 3, 480, 640),
    device: str = "cpu",
) -> int:
    """Conv2d + Linear FLOPs (MAC). bias 무시. w4 컨벤션."""
    flops: Dict[int, int] = {}
    hooks = []

    def conv_hook(layer, _inp, out):
        flops[id(layer)] = _conv2d_flops(layer, out)

    def linear_hook(layer, _inp, out):
        flops[id(layer)] = _linear_flops(layer, out)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(input_size, device=device)
        model.to(device)(dummy)
    for h in hooks:
        h.remove()
    if was_training:
        model.train()

    return int(sum(flops.values()))
