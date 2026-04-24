import os
from pathlib import Path

import pytest
import torch

from src.data.voc import VOCSegDataset
from src.data.transforms import build_val_transform

VOC_ROOT = Path(os.environ.get("VOC_ROOT", "data/voc"))
HAS_VOC = (VOC_ROOT / "VOCdevkit/VOC2012/JPEGImages").exists()


@pytest.mark.skipif(not HAS_VOC, reason="VOC data not present")
def test_voc_val_loads_one_sample():
    ds = VOCSegDataset(root=str(VOC_ROOT), split="val", transform=build_val_transform())
    assert len(ds) == 1449
    img, mask = ds[0]
    assert isinstance(img, torch.Tensor) and img.shape[0] == 3
    assert mask.dtype == torch.long
    unique = set(mask.unique().tolist())
    assert unique <= set(range(21)) | {255}
