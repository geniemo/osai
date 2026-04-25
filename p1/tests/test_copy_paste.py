import os
import random
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data.copy_paste import build_instance_pool, paste_instance, CopyPasteDataset


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


def test_paste_instance_modifies_target():
    target_img = np.full((100, 100, 3), 255, dtype=np.uint8)
    target_mask = np.zeros((100, 100), dtype=np.uint8)

    patch_img = np.zeros((20, 20, 3), dtype=np.uint8)
    patch_mask = np.ones((20, 20), dtype=bool)
    cls_id = 8

    np.random.seed(0)
    random.seed(0)
    out_img, out_mask = paste_instance(
        target_img.copy(), target_mask.copy(), patch_img, patch_mask, cls_id
    )
    assert (out_img == 0).any()
    assert (out_mask == cls_id).any()


def test_paste_instance_resizes_too_large_patch():
    target_img = np.full((100, 100, 3), 255, dtype=np.uint8)
    target_mask = np.zeros((100, 100), dtype=np.uint8)
    patch_img = np.zeros((90, 90, 3), dtype=np.uint8)
    patch_mask = np.ones((90, 90), dtype=bool)

    np.random.seed(0)
    random.seed(0)
    out_img, out_mask = paste_instance(
        target_img.copy(), target_mask.copy(), patch_img, patch_mask, 8
    )
    # paste된 영역 50×50 이하 (resize)
    assert (out_mask == 8).sum() <= 50 * 50


def test_copy_paste_dataset_preserves_length(tmp_path):
    from src.data.voc import VOCSegDataset
    voc_root, ids = _make_synthetic_voc(tmp_path)
    seg_dir = voc_root / "VOCdevkit/VOC2012/ImageSets/Segmentation"
    seg_dir.mkdir(parents=True, exist_ok=True)
    (seg_dir / "train.txt").write_text("\n".join(ids))
    base = VOCSegDataset(root=str(voc_root), split="train", transform=None)

    pool = build_instance_pool(str(voc_root), ids, min_area=100, max_area_ratio=0.99)
    cp_ds = CopyPasteDataset(base, pool, p=1.0, num_paste=(1, 2))
    assert len(cp_ds) == len(base)
