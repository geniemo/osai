import os
import random
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data.copy_paste import build_instance_pool


VOC_ROOT = Path(os.environ.get("VOC_ROOT", "data/voc"))
HAS_VOC = (VOC_ROOT / "VOCdevkit/VOC2012/SegmentationObject").exists()


def _make_synthetic_voc(tmp_path):
    """Create minimal VOC-like structure with 1 image + 2 instances."""
    devkit = tmp_path / "VOCdevkit" / "VOC2012"
    (devkit / "JPEGImages").mkdir(parents=True)
    (devkit / "SegmentationClass").mkdir(parents=True)
    (devkit / "SegmentationObject").mkdir(parents=True)

    img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    Image.fromarray(img).save(devkit / "JPEGImages" / "fake1.jpg")

    seg_cls = np.zeros((200, 200), dtype=np.uint8)
    seg_cls[20:80, 20:80] = 8     # cat region
    seg_cls[100:160, 100:160] = 12  # dog region
    Image.fromarray(seg_cls, mode="L").save(devkit / "SegmentationClass" / "fake1.png")

    seg_obj = np.zeros((200, 200), dtype=np.uint8)
    seg_obj[20:80, 20:80] = 1
    seg_obj[100:160, 100:160] = 2
    Image.fromarray(seg_obj, mode="L").save(devkit / "SegmentationObject" / "fake1.png")

    return tmp_path, ["fake1"]


def test_build_instance_pool_returns_correct_count(tmp_path):
    voc_root, ids = _make_synthetic_voc(tmp_path)
    pool = build_instance_pool(str(voc_root), ids, min_area=100, max_area_ratio=0.99)
    assert len(pool) == 2
    classes = sorted([cls for _, _, cls in pool])
    assert classes == [8, 12]


def test_build_instance_pool_respects_min_area(tmp_path):
    voc_root, ids = _make_synthetic_voc(tmp_path)
    pool = build_instance_pool(str(voc_root), ids, min_area=4096, max_area_ratio=0.99)
    assert len(pool) == 0


def test_build_instance_pool_respects_max_area(tmp_path):
    voc_root, ids = _make_synthetic_voc(tmp_path)
    pool = build_instance_pool(str(voc_root), ids, min_area=100, max_area_ratio=0.05)
    assert len(pool) == 0


def test_build_instance_pool_patch_shape(tmp_path):
    voc_root, ids = _make_synthetic_voc(tmp_path)
    pool = build_instance_pool(str(voc_root), ids, min_area=100, max_area_ratio=0.99)
    for patch_img, patch_mask, cls in pool:
        assert patch_img.ndim == 3 and patch_img.shape[2] == 3
        assert patch_mask.shape == patch_img.shape[:2]
        assert patch_mask.dtype == bool
        assert 1 <= cls <= 20


@pytest.mark.skipif(not HAS_VOC, reason="VOC SegmentationObject not present")
def test_build_instance_pool_real_voc():
    train_ids_file = VOC_ROOT / "VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt"
    with open(train_ids_file) as f:
        train_ids = [line.strip() for line in f][:50]
    pool = build_instance_pool(str(VOC_ROOT), train_ids)
    assert len(pool) > 50
    for _, _, cls in pool:
        assert 1 <= cls <= 20
