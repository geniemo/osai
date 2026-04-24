"""자동 다운로드: VOC 2012 + COCO 2017 (VOC subset).

PDF 정책: ImageNet/COCO/VOC만 허용 → **SBD 사용 안 함**.
SBD 허용 확인 시 download_sbd_and_merge 함수 다시 추가하고 main에서 호출.

VOC 2012: 공식 mirror (~2GB).
COCO: cocodataset.org에서 train2017 + annotations 다운 (~25GB).

호출:
    python -m src.data.download --voc-root data/voc --coco-root data/coco
"""
from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


VOC_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
COCO_TRAIN_URL = "http://images.cocodataset.org/zips/train2017.zip"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[skip] {dest} exists")
        return
    print(f"[get] {url} → {dest}")
    urlretrieve(url, dest)


def _extract(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {archive} → {target}")
    if archive.suffix == ".tar" or str(archive).endswith(".tgz") or str(archive).endswith(".tar.gz"):
        with tarfile.open(archive) as tf:
            tf.extractall(target)
    elif archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)


def download_voc(voc_root: Path) -> None:
    archive = voc_root / "VOCtrainval_11-May-2012.tar"
    _download(VOC_URL, archive)
    _extract(archive, voc_root)


def download_coco(coco_root: Path) -> None:
    train_zip = coco_root / "train2017.zip"
    ann_zip = coco_root / "annotations_trainval2017.zip"
    _download(COCO_TRAIN_URL, train_zip)
    _download(COCO_ANN_URL, ann_zip)
    if not (coco_root / "train2017").exists():
        _extract(train_zip, coco_root)
    if not (coco_root / "annotations").exists():
        _extract(ann_zip, coco_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voc-root", default="data/voc")
    parser.add_argument("--coco-root", default="data/coco")
    parser.add_argument("--skip-voc", action="store_true")
    parser.add_argument("--skip-coco", action="store_true")
    args = parser.parse_args()
    voc_root = Path(args.voc_root)
    coco_root = Path(args.coco_root)
    if not args.skip_voc:
        download_voc(voc_root)
    if not args.skip_coco:
        download_coco(coco_root)
    print("[done]")


if __name__ == "__main__":
    main()
