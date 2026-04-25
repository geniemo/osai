import numpy as np
import pytest
import torch

from src.data.sampler import compute_class_pixel_counts, compute_image_weights


def _create_fake_voc_data(tmp_path):
    """Create minimal VOC-like structure with 4 fake masks."""
    voc_dir = tmp_path / "VOCdevkit" / "VOC2012" / "SegmentationClass"
    voc_dir.mkdir(parents=True)
    from PIL import Image
    Image.fromarray(np.zeros((100, 100), dtype=np.uint8), mode="L").save(voc_dir / "0.png")
    arr1 = np.zeros((100, 100), dtype=np.uint8); arr1[:, 50:] = 1
    Image.fromarray(arr1, mode="L").save(voc_dir / "1.png")
    arr2 = np.zeros((100, 100), dtype=np.uint8); arr2[0:10, 0:10] = 9
    Image.fromarray(arr2, mode="L").save(voc_dir / "2.png")
    arr3 = np.full((100, 100), 255, dtype=np.uint8); arr3[0, 0] = 1
    Image.fromarray(arr3, mode="L").save(voc_dir / "3.png")
    return tmp_path, ["0", "1", "2", "3"]


def test_compute_class_pixel_counts(tmp_path):
    voc_root, ids = _create_fake_voc_data(tmp_path)
    counts = compute_class_pixel_counts(str(voc_root), ids)
    assert counts[0] == 24900   # bg: 10000 + 5000 + 9900 + 0
    assert counts[1] == 5001    # person: 5000 + 1
    assert counts[9] == 100     # chair: 100
    assert len(counts) == 21


def test_compute_image_weights_weak_class_higher(tmp_path):
    voc_root, ids = _create_fake_voc_data(tmp_path)
    counts = compute_class_pixel_counts(str(voc_root), ids)
    class_weights = 1.0 / np.sqrt(counts.astype(float) + 1)
    class_weights = class_weights / class_weights.sum()
    image_weights = compute_image_weights(str(voc_root), ids, class_weights)
    # Image 2 (chair) > Image 1 (person) — weak class higher weight
    assert image_weights[2] > image_weights[1]


def test_image_weights_skip_ignore(tmp_path):
    voc_root, ids = _create_fake_voc_data(tmp_path)
    counts = compute_class_pixel_counts(str(voc_root), ids)
    class_weights = 1.0 / np.sqrt(counts.astype(float) + 1)
    class_weights = class_weights / class_weights.sum()
    image_weights = compute_image_weights(str(voc_root), ids, class_weights)
    # Image 3: 255 ignore + 1 px class 1 → weight = class_weights[1]
    assert abs(image_weights[3].item() - class_weights[1].item()) < 1e-6
