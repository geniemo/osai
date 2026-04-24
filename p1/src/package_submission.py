"""채점용 PNG zip 검증/패키징.

요구:
- 정확히 1000 파일
- 이름 ^\\d{3}\\.png$
- flat root (하위 폴더 없음)
- 픽셀 값 [0, 20] 정수
- 압축해제 ≤500MB
"""
from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image


_PNG_RE = re.compile(r"^\d{3}\.png$")


def validate_pred_dir(pred_dir: Union[str, Path], sample_check: int = 50) -> None:
    pred_dir = Path(pred_dir)
    files = sorted(pred_dir.iterdir())
    files = [f for f in files if f.is_file() and f.suffix == ".png"]
    assert len(files) == 1000, f"expected 1000 files, got {len(files)}"
    for f in files:
        assert _PNG_RE.match(f.name), f"bad name: {f.name}"
    targets = files if sample_check >= len(files) else files[:: max(1, len(files) // sample_check)]
    for f in targets:
        arr = np.array(Image.open(f))
        assert arr.dtype == np.uint8, f"{f.name}: dtype {arr.dtype}"
        assert arr.min() >= 0 and arr.max() <= 20, f"{f.name}: pixel out of [0, 20]"
    total = sum(f.stat().st_size for f in files)
    assert total <= 500_000_000, f"total size {total/1e6:.1f}MB > 500MB"


def package_zip(pred_dir: Union[str, Path], out_zip: Union[str, Path]) -> None:
    pred_dir = Path(pred_dir); out_zip = Path(out_zip)
    files = sorted([f for f in pred_dir.iterdir() if f.is_file() and f.suffix == ".png"])
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=f.name)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    validate_pred_dir(args.pred)
    package_zip(args.pred, args.out)
    print(f"[ok] {args.out}")


if __name__ == "__main__":
    main()
