"""Export Generator to ONNX with 1024 bilinear wrapper.

Submission contract:
    input  z      (B, 512), float32
    output image  (B, 3, 1024, 1024), float32, range [-1, 1]

For dynamic batch support we monkey-patch every `ModulatedConv2d.forward` to
its mathematically equivalent `forward_onnx` variant, which uses standard
F.conv2d (no groups=B) so the trace doesn't bake the batch dim into the graph.
The two variants are bit-equivalent (verified on real ckpt); the swap only
matters for ONNX export.

Final wrapper clamps to [-1, 1] for spec compliance.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from p2.src.networks.generator import Generator, GeneratorConfig
from p2.src.networks.ops import ModulatedConv2d


TARGET_RES = 1024


class SubmissionWrapper(nn.Module):
    def __init__(self, G):
        super().__init__()
        self.G = G

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.G(z, style_mixing=False)
        x = F.interpolate(x, size=(TARGET_RES, TARGET_RES), mode="bilinear", align_corners=False)
        return x.clamp(-1.0, 1.0)


def swap_modconv_to_onnx(G: nn.Module) -> None:
    """In-place: prepare each ModulatedConv2d for ONNX-friendly tracing.

    1) `prepare_for_onnx_export()` caches sum_k(weight^2) into a buffer so the
       demod path doesn't reference self.weight a second time (which would
       double the ONNX initializer).
    2) Swap .forward to .forward_onnx so the conv uses standard F.conv2d
       (groups=1) instead of the groups=B trick → dynamic batch traces cleanly.
    """
    for m in G.modules():
        if isinstance(m, ModulatedConv2d):
            m.prepare_for_onnx_export()
            m.forward = m.forward_onnx


def verify_numeric_equivalence(G_orig: nn.Module, G_onnx: nn.Module, *, batch: int = 2,
                               atol: float = 1e-3, seed: int = 0) -> None:
    """Sanity-check that swapped-forward G produces near-identical output to original.

    AddNoise.forward(noise=None) draws fresh torch.randn each call. Without resetting
    the global RNG between the two forwards, G_orig and G_onnx consume different noise
    streams → spurious diff that grows as AddNoise.weight learns away from zero.
    Reset before each call so both consume the same noise.
    """
    G_orig.eval(); G_onnx.eval()
    gen = torch.Generator().manual_seed(seed)
    z = torch.randn(batch, 512, generator=gen)
    with torch.no_grad():
        torch.manual_seed(seed)
        a = G_orig(z, style_mixing=False)
        torch.manual_seed(seed)
        b = G_onnx(z, style_mixing=False)
    diff = (a - b).abs().max().item()
    print(f"verify_numeric_equivalence: max |diff| = {diff:.6f} (atol={atol})")
    assert diff < atol, f"ONNX-forward output diverges from original (max diff {diff})"


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
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Tracing batch size. With --dynamic-batch this is just a dummy.")
    parser.add_argument("--dynamic-batch", action="store_true",
                        help="Export with dynamic batch axis (requires ModulatedConv ONNX swap).")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip numeric equivalence check between orig vs ONNX-friendly forward.")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    g_cfg = GeneratorConfig.from_dict(ckpt["meta"]["generator_config"])
    state = ckpt["G_state"] if args.no_ema else ckpt["G_ema_state"]

    G = Generator(g_cfg)
    G.load_state_dict(state)

    if args.dynamic_batch:
        if not args.skip_verify:
            G_ref = Generator(g_cfg)
            G_ref.load_state_dict(state)
            swap_modconv_to_onnx(G)
            verify_numeric_equivalence(G_ref, G)
        else:
            swap_modconv_to_onnx(G)

    export_to_onnx(G, args.out, opset=args.opset, batch_size=args.batch_size,
                   dynamic_batch=args.dynamic_batch)


if __name__ == "__main__":
    main()
