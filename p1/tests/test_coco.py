import numpy as np
import pytest

from src.data.coco import COCO_TO_VOC, build_voc_mask_from_anns


def test_coco_to_voc_mapping_complete():
    assert len(COCO_TO_VOC) == 20
    voc_ids = set(COCO_TO_VOC.values())
    assert voc_ids == set(range(1, 21))


def test_build_voc_mask_voc_class_painted():
    h, w = 4, 4
    cat_mask = np.zeros((h, w), dtype=np.uint8)
    cat_mask[:2, :2] = 1
    anns = [{"category_id": 17, "binary_mask": cat_mask}]
    out = build_voc_mask_from_anns(anns, h, w)
    assert out[0, 0] == 8
    assert out[3, 3] == 0


def test_build_voc_mask_non_voc_class_becomes_ignore():
    h, w = 4, 4
    bowl_mask = np.zeros((h, w), dtype=np.uint8)
    bowl_mask[:2, :2] = 1
    anns = [{"category_id": 51, "binary_mask": bowl_mask}]
    out = build_voc_mask_from_anns(anns, h, w)
    assert out[0, 0] == 255
    assert out[3, 3] == 0
