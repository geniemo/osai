"""Export Generator to ONNX with 1024 bilinear wrapper.

Submission contract:
    input  z      (B, 512), float32
    output image  (B, 3, 1024, 1024), float32, range [-1, 1]

NOTE: StyleGAN2's ModulatedConv uses a group-conv trick (groups=B) that does
not trace cleanly with dynamic batch axes. By default we export with a static
batch size (no dynamic_axes); pass --dynamic-batch only if you've verified
the resulting ONNX graph works with variable batch.

Final wrapper clamps to [-1, 1] for spec compliance.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from p2.src.networks.generator import Generator, GeneratorConfig


TARGET_RES = 1024


class SubmissionWrapper(nn.Module):
    def __init__(self, G):
        super().__init__()
        self.G = G

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.G(z, style_mixing=False)
        x = F.interpolate(x, size=(TARGET_RES, TARGET_RES), mode="bilinear", align_corners=False)
        return x.clamp(-1.0, 1.0)


def export_to_onnx(G: nn.Module, out_path: Path, *, opset: int = 17,
                   batch_size: int = 1, dynamic_batch: bool = False) -> None:
    assert getattr(G, "z_dim", None) == 512, "G.z_dim must be 512"
    G.eval()
    wrapper = SubmissionWrapper(G).eval()
    dummy_z = torch.randn(batch_size, 512)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        input_names=["z"], output_names=["image"],
        opset_version=opset,
        dynamo=False,
    )
    if dynamic_batch:
        kwargs["dynamic_axes"] = {"z": {0: "batch"}, "image": {0: "batch"}}
    torch.onnx.export(wrapper, dummy_z, str(out_path), **kwargs)
    with torch.no_grad():
        ref = wrapper(dummy_z)
    print(f"Saved {out_path}, output shape {tuple(ref.shape)}, range [{ref.min():.3f}, {ref.max():.3f}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--out", default=Path("checkpoints/model.onnx"), type=Path)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dynamic-batch", action="store_true",
                        help="Try exporting with dynamic batch axes (may fail due to ModulatedConv group-conv).")
    parser.add_argument("--no-ema", action="store_true")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    g_cfg = GeneratorConfig.from_dict(ckpt["meta"]["generator_config"])
    G = Generator(g_cfg)
    state = ckpt["G_state"] if args.no_ema else ckpt["G_ema_state"]
    G.load_state_dict(state)
    export_to_onnx(G, args.out, opset=args.opset, batch_size=args.batch_size, dynamic_batch=args.dynamic_batch)


if __name__ == "__main__":
    main()
