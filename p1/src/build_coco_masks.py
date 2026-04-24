"""COCO VOC-subset mask cache pre-generation.

COCO annotation은 polygon/RLE라 학습 시 매번 binary mask로 변환 필요.
이 script는 모든 mask를 PNG로 미리 저장 → 학습 시 cache hit (빠름).

단일 프로세스: ~95-160분 (95K images × 60-100ms each).
DataLoader 8 worker: ~15-25분.

호출:
    python -m src.build_coco_masks --coco-root data/coco --num-workers 8
"""
from __future__ import annotations

import argparse
import time

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.coco import COCOSegDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-root", default="data/coco")
    parser.add_argument("--split", default="train2017")
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    ds = COCOSegDataset(coco_root=args.coco_root, split=args.split, transform=None)
    print(f"[info] COCO {args.split} VOC subset: {len(ds)} images")
    print(f"[info] cache dir: {ds.cache_dir}")
    print(f"[info] using {args.num_workers} workers")

    loader = DataLoader(
        ds,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=lambda x: None,
        persistent_workers=False,
    )
    start = time.time()
    for _ in tqdm(loader, total=len(ds), desc="building masks"):
        pass
    elapsed = time.time() - start
    print(f"[done] {len(ds)} masks in {elapsed:.1f}s ({elapsed/len(ds)*1000:.1f}ms/img)")


if __name__ == "__main__":
    main()
