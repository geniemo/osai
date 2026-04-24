import torch
from src.losses.seg_loss import SegLoss


def test_perfect_prediction_yields_low_loss():
    crit = SegLoss(num_classes=21, ignore_index=255, dice_weight=0.5, aux_weight=0.4)
    target = torch.randint(0, 21, (2, 32, 32))
    logits = torch.zeros(2, 21, 32, 32)
    logits.scatter_(1, target.unsqueeze(1), 10.0)
    loss = crit(logits, None, target)
    assert loss.item() < 0.5


def test_ignore_pixels_excluded_from_loss():
    crit = SegLoss(num_classes=2, ignore_index=255, dice_weight=0.0, aux_weight=0.0)
    logits = torch.randn(1, 2, 4, 4)
    target = torch.full((1, 4, 4), 255)
    loss = crit(logits, None, target)
    assert torch.isfinite(loss)


def test_aux_loss_added_when_provided():
    crit = SegLoss(num_classes=2, ignore_index=255, dice_weight=0.0, aux_weight=1.0)
    target = torch.randint(0, 2, (1, 4, 4))
    main = torch.randn(1, 2, 4, 4)
    aux = torch.randn(1, 2, 4, 4)
    l_with = crit(main, aux, target)
    l_without = crit(main, None, target)
    assert l_with > l_without
