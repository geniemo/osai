import torch
from src.utils.metrics import SegMetric


def test_perfect_prediction_yields_iou_one():
    m = SegMetric(num_classes=3, ignore_index=255)
    pred = torch.tensor([[0, 1, 2], [0, 1, 2]])
    target = torch.tensor([[0, 1, 2], [0, 1, 2]])
    m.update(pred, target)
    miou, per_class = m.compute()
    assert miou == 1.0
    assert per_class == [1.0, 1.0, 1.0]


def test_ignore_label_excluded_from_iou():
    m = SegMetric(num_classes=2, ignore_index=255)
    pred = torch.tensor([[0, 1, 0], [1, 0, 1]])
    target = torch.tensor([[0, 1, 255], [1, 0, 255]])
    m.update(pred, target)
    miou, _ = m.compute()
    assert miou == 1.0


def test_class_absent_from_gt_excluded_from_mean():
    m = SegMetric(num_classes=3)
    pred = torch.tensor([[0, 1], [0, 1]])
    target = torch.tensor([[0, 1], [0, 1]])
    m.update(pred, target)
    miou, per_class = m.compute()
    assert miou == 1.0
    assert per_class[0] == 1.0
    assert per_class[1] == 1.0
    assert per_class[2] != per_class[2]  # NaN check


def test_partial_overlap_iou():
    m = SegMetric(num_classes=2)
    pred = torch.tensor([0, 0, 1, 1])
    target = torch.tensor([0, 1, 0, 1])
    m.update(pred, target)
    miou, per_class = m.compute()
    assert abs(per_class[0] - 1/3) < 1e-6
    assert abs(per_class[1] - 1/3) < 1e-6
    assert abs(miou - 1/3) < 1e-6
