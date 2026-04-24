import torch
from PIL import Image
import numpy as np

from src.data.transforms import build_train_transform, build_val_transform


def _fake_pair(h=300, w=400):
    img = Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))
    mask = Image.fromarray(np.random.randint(0, 21, (h, w), dtype=np.uint8))
    return img, mask


def test_train_transform_returns_tensors_with_correct_dtypes():
    t = build_train_transform(crop_size=128, scale_range=(0.5, 2.0))
    img, mask = _fake_pair()
    out_img, out_mask = t(img, mask)
    assert isinstance(out_img, torch.Tensor)
    assert out_img.dtype == torch.float32
    assert out_img.shape == (3, 128, 128)
    assert out_mask.dtype == torch.long
    assert out_mask.shape == (128, 128)


def test_mask_values_in_valid_range_after_transform():
    t = build_train_transform(crop_size=128, scale_range=(0.5, 2.0))
    img, mask = _fake_pair()
    _, out_mask = t(img, mask)
    unique = set(out_mask.unique().tolist())
    valid = set(range(21)) | {255}
    assert unique <= valid


def test_val_transform_normalizes_to_imagenet_stats():
    """val_transform (no aug) gives correct Normalize output for uniform image at ImageNet mean."""
    t = build_val_transform()
    arr = np.full((300, 400, 3), int(0.485 * 255), dtype=np.uint8)
    arr[..., 1] = int(0.456 * 255)
    arr[..., 2] = int(0.406 * 255)
    img = Image.fromarray(arr)
    mask = Image.fromarray(np.zeros((300, 400), dtype=np.uint8))
    out_img, _ = t(img, mask)
    # Normalize on uniform ImageNet-mean image → all pixels ~0
    assert abs(out_img.mean().item()) < 0.05


def test_val_transform_preserves_size_and_no_aug():
    t = build_val_transform()
    img, mask = _fake_pair(h=200, w=300)
    out_img, out_mask = t(img, mask)
    assert out_img.shape == (3, 200, 300)
    assert out_mask.shape == (200, 300)
