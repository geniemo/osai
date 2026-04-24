"""자동 다운로드 + SBD merge + COCO mask cache 생성.

VOC 2012: TorchVision auto download.
SBD: berkeley mirror에서 수동 다운 + train_aug.txt 생성.
COCO: cocodataset.org에서 train2017 + annotations 다운 (~25GB).

호출:
    python -m src.data.download --voc-root data/voc --coco-root data/coco
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


SBD_URL = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz"
VOC_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
COCO_TRAIN_URL = "http://images.cocodataset.org/zips/train2017.zip"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
TRAINAUG_URL = "https://www.dropbox.com/scl/fi/8gqxa8wxwa0jpe5xkcjzz/train_aug.txt?rlkey=ssvd0w6jrhqg5sl4hh7m6n9bp&dl=1"


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


def download_sbd_and_merge(voc_root: Path) -> None:
    sbd_archive = voc_root / "benchmark.tgz"
    _download(SBD_URL, sbd_archive)
    sbd_extract = voc_root / "sbd"
    if not (sbd_extract / "benchmark_RELEASE").exists():
        _extract(sbd_archive, sbd_extract)

    out_mask_dir = voc_root / "VOCdevkit" / "VOC2012" / "SegmentationClassAug"
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    src_voc_mask = voc_root / "VOCdevkit" / "VOC2012" / "SegmentationClass"
    if src_voc_mask.exists():
        for p in src_voc_mask.iterdir():
            shutil.copy2(p, out_mask_dir / p.name)

    sbd_cls = sbd_extract / "benchmark_RELEASE" / "dataset" / "cls"
    if sbd_cls.exists():
        from scipy.io import loadmat
        from PIL import Image
        for mat_file in sbd_cls.glob("*.mat"):
            png_path = out_mask_dir / f"{mat_file.stem}.png"
            if png_path.exists():
                continue
            data = loadmat(mat_file)
            mask = data["GTcls"][0][0][1].astype("uint8")
            Image.fromarray(mask, mode="L").save(png_path)

    seg_dir = voc_root / "VOCdevkit" / "VOC2012" / "ImageSets" / "Segmentation"
    train_aug_path = seg_dir / "train_aug.txt"
    if not train_aug_path.exists():
        try:
            _download(TRAINAUG_URL, train_aug_path)
        except Exception as e:
            print(f"[warn] train_aug.txt download failed: {e}", file=sys.stderr)
            print("[warn] Build manually from SBD train + VOC train, excluding VOC val")


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
        download_sbd_and_merge(voc_root)
    if not args.skip_coco:
        download_coco(coco_root)
    print("[done]")


if __name__ == "__main__":
    main()
