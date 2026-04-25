import torch

from src.losses.lovasz import lovasz_grad, lovasz_softmax


def test_lovasz_grad_known_input():
    """Lovász gradient on toy input."""
    gt = torch.tensor([1.0, 1.0, 0.0, 0.0])
    grad = lovasz_grad(gt)
    expected = torch.tensor([0.5, 0.5, 0.0, 0.0])
    assert torch.allclose(grad, expected, atol=1e-6)


def test_lovasz_softmax_perfect_prediction():
    N, C, H, W = 1, 21, 16, 16
    target = torch.randint(0, C, (N, H, W))
    logits = torch.zeros(N, C, H, W)
    logits.scatter_(1, target.unsqueeze(1), 100.0)
    loss = lovasz_softmax(logits, target)
    assert loss.item() < 0.01


def test_lovasz_softmax_random_in_range():
    N, C, H, W = 2, 21, 32, 32
    logits = torch.randn(N, C, H, W)
    target = torch.randint(0, C, (N, H, W))
    loss = lovasz_softmax(logits, target)
    assert 0.0 < loss.item() < 1.0


def test_lovasz_softmax_ignore_index():
    N, C, H, W = 1, 21, 8, 8
    logits = torch.randn(N, C, H, W)
    target_all_ignore = torch.full((N, H, W), 255)
    loss_ignored = lovasz_softmax(logits, target_all_ignore, ignore_index=255)
    assert loss_ignored.item() == 0.0


def test_lovasz_softmax_gradient_flow():
    N, C, H, W = 1, 21, 8, 8
    logits = torch.randn(N, C, H, W, requires_grad=True)
    target = torch.randint(0, C, (N, H, W))
    loss = lovasz_softmax(logits, target)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
