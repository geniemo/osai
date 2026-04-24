import pytest
from src.data.builder import build_dataloaders, _assert_not_test_path


def test_isolation_guard_rejects_input_path():
    with pytest.raises(AssertionError, match="must not include 'input/'"):
        _assert_not_test_path("p1/input/test_public", role="voc_root")


def test_isolation_guard_accepts_data_path():
    _assert_not_test_path("data/voc", role="voc_root")
    _assert_not_test_path("./data/coco", role="coco_root")


def test_isolation_guard_blocks_relative_paths_too():
    with pytest.raises(AssertionError):
        _assert_not_test_path("./input/anywhere", role="coco_root")
