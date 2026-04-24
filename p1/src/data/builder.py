"""DataLoader builder + test set 격리 가드 (개발 안전망)."""
from __future__ import annotations

from pathlib import PurePath
from typing import Tuple

from torch.utils.data import ConcatDataset, DataLoader

from src.data.coco import COCOSegDataset
from src.data.transforms import build_train_transform, build_val_transform
from src.data.voc import VOCSegDataset
from src.utils.seed import worker_init_fn


def _assert_not_test_path(path: str, *, role: str) -> None:
    """train data root는 'input/' 디렉토리를 포함해선 안 됨 (test set 격리)."""
    parts = PurePath(path).parts
    assert "input" not in parts, (
        f"{role}={path!r} must not include 'input/' — test set 격리 위반. "
        f"data 경로는 'data/voc' 등 input/ 외부에."
    )


def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader]:
    """cfg["data"]에서 path/배치/워커 읽어 train+val DataLoader 생성.

    cfg["data"]["stage"] = 1: COCO+VOC mixed
    cfg["data"]["stage"] = 2: VOC train only (1,464 images, no SBD per PDF policy)
    """
    data_cfg = cfg["data"]
    voc_root = data_cfg["voc_root"]
    coco_root = data_cfg["coco_root"]
    _assert_not_test_path(voc_root, role="voc_root")
    _assert_not_test_path(coco_root, role="coco_root")

    train_t = build_train_transform(
        crop_size=data_cfg["crop_size"],
        scale_range=tuple(data_cfg["scale_range"]),
    )
    val_t = build_val_transform()

    voc_train = VOCSegDataset(root=voc_root, split="train", transform=train_t)
    voc_val = VOCSegDataset(root=voc_root, split="val", transform=val_t)

    stage = data_cfg.get("stage", 1)
    if stage == 1:
        coco_train = COCOSegDataset(coco_root=coco_root, split="train2017", transform=train_t)
        train_ds = ConcatDataset([coco_train, voc_train])
    else:
        train_ds = voc_train

    common = dict(
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg["pin_memory"],
        persistent_workers=data_cfg.get("persistent_workers", True),
        worker_init_fn=worker_init_fn,
    )
    train_loader = DataLoader(train_ds, batch_size=data_cfg["batch_size"], shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(voc_val, batch_size=1, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader
