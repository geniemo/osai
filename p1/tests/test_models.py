import torch
import pytest

from src.models.builder import build_model


def test_build_resnet_deeplabv3plus_b_forward_shape():
    cfg = {
        "model": {
            "backbone": "resnet50",
            "head": "deeplabv3plus",
            "num_classes": 21,
            "output_stride": 16,
            "pretrained": False,
            "use_aux": True,
        }
    }
    m = build_model(cfg).eval()
    x = torch.zeros(1, 3, 480, 640)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (1, 21, 480, 640)


def test_build_with_aux_returns_tuple_in_training():
    cfg = {
        "model": {
            "backbone": "resnet50",
            "head": "deeplabv3plus",
            "num_classes": 21,
            "output_stride": 16,
            "pretrained": False,
            "use_aux": True,
        }
    }
    m = build_model(cfg).train()
    x = torch.zeros(1, 3, 480, 640)
    out = m(x)
    assert isinstance(out, tuple) and len(out) == 2
    assert out[0].shape == (1, 21, 480, 640)
    assert out[1].shape == (1, 21, 480, 640)


def test_export_mode_removes_aux_output():
    cfg = {
        "model": {
            "backbone": "resnet50", "head": "deeplabv3plus",
            "num_classes": 21, "output_stride": 16, "pretrained": False, "use_aux": True,
        }
    }
    m = build_model(cfg).eval().export_mode()
    x = torch.zeros(1, 3, 480, 640)
    with torch.no_grad():
        y = m(x)
    assert isinstance(y, torch.Tensor)
