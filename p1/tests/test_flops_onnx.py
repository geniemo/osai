import torch
import torch.nn as nn

from src.utils.flops import count_onnx_flops, count_pytorch_flops


def _export_to_temp(model: nn.Module, input_size, path):
    model.eval()
    dummy = torch.zeros(input_size)
    torch.onnx.export(
        model, dummy, str(path),
        input_names=["input"], output_names=["output"],
        opset_version=17, dynamic_axes=None,
    )


def test_single_conv_onnx_matches_pytorch(tmp_path):
    m = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
    onnx_path = tmp_path / "single_conv.onnx"
    _export_to_temp(m, (1, 3, 16, 16), onnx_path)

    py_flops = count_pytorch_flops(m, (1, 3, 16, 16))
    onnx_flops, breakdown = count_onnx_flops(str(onnx_path), input_shape=(1, 3, 16, 16))
    assert py_flops == onnx_flops
    assert breakdown["Conv"] == py_flops


def test_resnet_block_onnx_within_5pct_of_pytorch(tmp_path):
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
            self.b1 = nn.BatchNorm2d(16)
            self.c2 = nn.Conv2d(16, 8, 1, bias=False)
        def forward(self, x):
            return self.c2(torch.relu(self.b1(self.c1(x))))

    m = Tiny()
    onnx_path = tmp_path / "tiny.onnx"
    _export_to_temp(m, (1, 3, 32, 32), onnx_path)

    py = count_pytorch_flops(m, (1, 3, 32, 32))
    onx, _ = count_onnx_flops(str(onnx_path), input_shape=(1, 3, 32, 32))
    rel_diff = abs(py - onx) / py
    assert rel_diff < 0.05, f"PyTorch={py} ONNX={onx} diff={rel_diff:.3f}"
