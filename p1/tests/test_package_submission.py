import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.package_submission import validate_pred_dir, package_zip


def _make_pred_dir(d: Path, n: int, valid: bool = True) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        arr = np.zeros((100, 100), dtype=np.uint8)
        if not valid:
            arr[0, 0] = 99
        Image.fromarray(arr, mode="L").save(d / f"{i:03d}.png")


def test_validate_accepts_correct_dir(tmp_path):
    d = tmp_path / "pred"
    _make_pred_dir(d, 1000)
    validate_pred_dir(d)


def test_validate_rejects_wrong_count(tmp_path):
    d = tmp_path / "pred"
    _make_pred_dir(d, 999)
    with pytest.raises(AssertionError, match="1000"):
        validate_pred_dir(d)


def test_validate_rejects_invalid_pixel(tmp_path):
    d = tmp_path / "pred"
    _make_pred_dir(d, 1000, valid=False)
    with pytest.raises(AssertionError, match=r"\[0, 20\]"):
        validate_pred_dir(d, sample_check=10)


def test_package_zip_flat_and_correct_files(tmp_path):
    d = tmp_path / "pred"
    _make_pred_dir(d, 1000)
    out_zip = tmp_path / "submission.zip"
    package_zip(d, out_zip)
    with zipfile.ZipFile(out_zip) as zf:
        names = zf.namelist()
    assert len(names) == 1000
    assert all("/" not in n for n in names)
    assert names[0] == "000.png"
    assert names[-1] == "999.png"
