"""FLOPs counters: PyTorch (forward hooks) + ONNX (graph traversal).

내부 카운터는 MAC을 측정하고, 사용처에서 ×2를 곱해 GFLOPs로 표기한다.
1 MAC (a*b+c) = 2 FLOP. 채점 사이트도 ×2 컨벤션.
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


# === ONNX FLOPs counter (custom, no 3rd-party FLOPs lib) ===

from collections import defaultdict
from typing import Optional, Tuple as _T

import onnx
from onnx import shape_inference, numpy_helper


def _get_attr(node, name: str, default):
    for a in node.attribute:
        if a.name == name:
            if a.type == onnx.AttributeProto.INT:
                return a.i
            if a.type == onnx.AttributeProto.INTS:
                return list(a.ints)
            if a.type == onnx.AttributeProto.FLOAT:
                return a.f
            if a.type == onnx.AttributeProto.STRING:
                return a.s.decode()
    return default


def _build_shape_map(model: onnx.ModelProto) -> dict:
    shapes = {}
    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        dims = []
        for d in vi.type.tensor_type.shape.dim:
            dims.append(d.dim_value if d.dim_value > 0 else None)
        shapes[vi.name] = dims
    for init in model.graph.initializer:
        shapes[init.name] = list(init.dims)
    return shapes


def _conv_flops(node, shape_map) -> int:
    out = shape_map.get(node.output[0])
    w = shape_map.get(node.input[1])
    if out is None or w is None or any(x is None for x in out + w):
        return 0
    n, c_out, h_out, w_out = out
    _, c_in_per_g, k_h, k_w = w
    return n * c_out * h_out * w_out * c_in_per_g * k_h * k_w


def _gemm_flops(node, shape_map) -> int:
    a = shape_map.get(node.input[0])
    out = shape_map.get(node.output[0])
    if a is None or out is None or any(x is None for x in a + out):
        return 0
    m, k = a[-2], a[-1]
    n = out[-1]
    return m * k * n


def _matmul_flops(node, shape_map) -> int:
    return _gemm_flops(node, shape_map)


def count_onnx_flops(
    onnx_path: str,
    input_shape: _T[int, int, int, int] = (1, 3, 480, 640),
) -> _T[int, dict]:
    """ONNX 그래프 FLOPs (MAC). Conv/Gemm/MatMul만. BN/ReLU/Add/Resize는 0.

    가중치 제거된 model_structure.onnx도 OK — initializer.dims는 유지됨.
    """
    model = onnx.load(onnx_path)

    inp = model.graph.input[0]
    for i, val in enumerate(input_shape):
        inp.type.tensor_type.shape.dim[i].dim_value = val

    inferred = shape_inference.infer_shapes(model, strict_mode=False)
    shape_map = _build_shape_map(inferred)

    total = 0
    breakdown: dict = defaultdict(int)
    for node in inferred.graph.node:
        op = node.op_type
        if op == "Conv":
            f = _conv_flops(node, shape_map)
        elif op == "Gemm":
            f = _gemm_flops(node, shape_map)
        elif op == "MatMul":
            f = _matmul_flops(node, shape_map)
        else:
            f = 0
        total += f
        breakdown[op] += f

    return total, dict(breakdown)
