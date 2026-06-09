"""Build the submission zip per project02.pdf spec.

Output zip structure (root = SID_project02/):
    SID_project02/
    ├── src/
    ├── configs/
    ├── train.py, generate.py, export_onnx.py, eval_fid.py
    ├── checkpoints/model.pth
    ├── SID_project02_report.pdf
    ├── pyproject.toml
    └── README.md

Usage:
    python package_submission.py \\
        --ckpt checkpoints/model.pth \\
        --report 2020314315_project02_report.pdf \\
        --out 2020314315_project02.zip
"""
from __future__ import annotations
import argparse
import shutil
import zipfile
from pathlib import Path


SID = "2020314315"
PROJECT_ROOT = Path(__file__).resolve().parent

INCLUDE_DIRS = ["src", "configs"]
INCLUDE_FILES = [
    "train.py",
    "generate.py",
    "export_onnx.py",
    "eval_fid.py",
    "pyproject.toml",
    "README.md",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=Path,
                        help="Slim model.pth (G_ema_state + meta).")
    parser.add_argument("--report", required=True, type=Path,
                        help="Report PDF (named {SID}_project02_report.pdf).")
    parser.add_argument("--out", default=Path(f"{SID}_project02.zip"), type=Path)
    args = parser.parse_args()

    ckpt = args.ckpt.resolve()
    report = args.report.resolve()
    out = args.out.resolve()

    assert ckpt.exists(), f"ckpt not found: {ckpt}"
    assert report.exists(), f"report not found: {report}"
    assert report.suffix.lower() == ".pdf", f"report must be PDF, got {report.suffix}"

    root_name = f"{SID}_project02"
    staging = Path("/tmp") / root_name
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    # Copy source directories
    for d in INCLUDE_DIRS:
        src_dir = PROJECT_ROOT / d
        assert src_dir.is_dir(), f"missing dir: {src_dir}"
        shutil.copytree(
            src_dir, staging / d,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
        )

    # Copy individual files
    for f in INCLUDE_FILES:
        src_file = PROJECT_ROOT / f
        assert src_file.is_file(), f"missing file: {src_file}"
        shutil.copy(src_file, staging / f)

    # checkpoints/model.pth
    (staging / "checkpoints").mkdir()
    shutil.copy(ckpt, staging / "checkpoints" / "model.pth")

    # report
    shutil.copy(report, staging / f"{SID}_project02_report.pdf")

    # zip
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(staging.rglob("*")):
            if p.is_file():
                arcname = p.relative_to(staging.parent)
                zf.write(p, arcname)

    print(f"Created: {out}")
    print(f"  size: {out.stat().st_size / 1e6:.2f} MB")

    # Verify contents
    with zipfile.ZipFile(out) as zf:
        names = sorted(zf.namelist())
    print(f"\nContents ({len(names)} entries):")
    for n in names:
        print(f"  {n}")

    shutil.rmtree(staging)


if __name__ == "__main__":
    main()
