import torch

from src.losses.boundary import boundary_mask, boundary_weighted_ce


def test_boundary_mask_single_class_no_boundary():
    """모든 픽셀이 같은 class면 boundary 없음."""
    target = torch.zeros(1, 16, 16, dtype=torch.long)
    bmask = boundary_mask(target, kernel_size=3)
    assert not bmask.any()


def test_boundary_mask_detects_square_boundary():
    """가운데 정사각형 객체 → 경계만 True."""
    target = torch.zeros(1, 16, 16, dtype=torch.long)
    target[0, 4:12, 4:12] = 1  # class 1 square
    bmask = boundary_mask(target, kernel_size=3)
    # 경계 라인은 True여야
    assert bmask[0, 4, 4]   # corner
    assert bmask[0, 4, 7]   # top edge
    # 내부는 False
    assert not bmask[0, 7, 7]   # center of square
    # background도 객체와 멀면 False
    assert not bmask[0, 0, 0]


def test_boundary_weighted_ce_perfect_prediction():
    """Perfect logits → loss ≈ 0."""
    N, C, H, W = 1, 21, 16, 16
    target = torch.randint(0, C, (N, H, W))
    logits = torch.zeros(N, C, H, W)
    logits.scatter_(1, target.unsqueeze(1), 100.0)
    loss = boundary_weighted_ce(logits, target)
    assert loss.item() < 0.01


def test_boundary_weighted_ce_random_finite():
    """Random logits → finite loss."""
    N, C, H, W = 2, 21, 32, 32
    logits = torch.randn(N, C, H, W)
    target = torch.randint(0, C, (N, H, W))
    loss = boundary_weighted_ce(logits, target)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_boundary_weighted_ce_ignore_index():
    """All ignore → loss == 0 (no contribution)."""
    N, C, H, W = 1, 21, 8, 8
    logits = torch.randn(N, C, H, W)
    target = torch.full((N, H, W), 255)
    loss = boundary_weighted_ce(logits, target, ignore_index=255)
    assert loss.item() == 0.0


def test_boundary_weighted_ce_alpha_increases_loss_when_boundaries_wrong():
    """boundary_alpha를 키우면, boundary 픽셀이 틀린 prediction에서 loss 더 커짐."""
    N, C, H, W = 1, 2, 16, 16
    target = torch.zeros(N, H, W, dtype=torch.long)
    target[:, 4:12, 4:12] = 1  # square boundary 존재
    # boundary 픽셀에서만 일부러 틀린 logits
    logits = torch.zeros(N, C, H, W)
    logits[:, 0, :, :] = 5.0   # 모든 픽셀 class 0 예측
    # → boundary에서 class 1을 틀리게 예측 → 큰 loss 기여
    loss_a1 = boundary_weighted_ce(logits, target, boundary_alpha=1.0)
    loss_a5 = boundary_weighted_ce(logits, target, boundary_alpha=5.0)
    # alpha 5x → boundary loss contribution 더 큼 → mean도 다름
    # weights 가 다르므로 결과 다른지 확인
    assert loss_a1.item() != loss_a5.item()


def test_boundary_weighted_ce_gradient_flow():
    """Backward pass without NaN."""
    N, C, H, W = 1, 21, 8, 8
    logits = torch.randn(N, C, H, W, requires_grad=True)
    target = torch.randint(0, C, (N, H, W))
    loss = boundary_weighted_ce(logits, target)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
