"""
pr3_freeze.py
Practice 3: Freezing parts of a ResNet34 model from torchvision.

Tasks:
  (1) Freeze the first 2 stages (layer1 and layer2) — the layers closest
      to the input — by setting requires_grad=False on their parameters.
  (2) Additionally freeze all 1-D parameters (e.g., BN weight/bias, Conv bias)
      across the entire model, then report the number of trainable parameters.
"""

import torch.nn as nn
from torchvision.models import resnet34


def load_model() -> nn.Module:
    """
    Load a ResNet34 model from torchvision without pretrained weights.

    Returns:
        model (nn.Module): ResNet34 model with random initialization.
    """
    model = resnet34(weights=None)
    return model


def freeze_first_two_stages(model: nn.Module, verbose: bool = True):
    """
    (1) Freeze the first 2 residual stages (layer1, layer2).

    In torchvision's ResNet34 the stages closest to the input are:
        conv1, bn1  — stem (not a "stage" per se)
        layer1      — stage 1
        layer2      — stage 2
        layer3      — stage 3
        layer4      — stage 4

    We freeze layer1 and layer2 by setting requires_grad=False on all
    their parameters, which prevents gradient computation and weight updates.

    Args:
        model (nn.Module): The model to modify in-place.
    """
    frozen_count = 0
    for name, param in model.named_parameters():
        if param is None:
            continue
        if ("layer1" in name) or ("layer2" in name):
            param.requires_grad = False
            frozen_count += 1

    if verbose:
        print("=" * 60)
        print("[Task 1] Freeze layer1 and layer2")
        print("=" * 60)
        print(f"  Frozen parameter tensors: {frozen_count}")
        print()


def freeze_1d_parameters(model: nn.Module, verbose: bool = True):
    """
    (2) Additionally freeze all 1-D parameters across the entire model.

    1-D parameters include:
        - BatchNorm weight (gamma) and bias (beta)
        - Conv2d / Linear bias vectors

    After freezing, count and report the number of still-trainable parameters.

    Args:
        model (nn.Module): The model to modify in-place.
    """
    frozen_count = 0
    for name, param in model.named_parameters():
        if param.dim() == 1:
            param.requires_grad = False
            frozen_count += 1

    if verbose:
        print("=" * 60)
        print("[Task 2] Freeze all 1-D parameters")
        print("=" * 60)
        print(f"  Frozen 1-D parameter tensors: {frozen_count}")
        print()


def count_trainable_parameters(model: nn.Module) -> int:
    """
    (3) Count the number of trainable parameters after freezing.

    Args:
        model (nn.Module): Target model.

    Returns:
        count (int): The number of trainable parameters.
    """
    count = 0
    for _, param in model.named_parameters():
        if param.requires_grad:
            count += param.numel()
    return count


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Load ResNet34 without pretrained weights
    model = load_model()

    # (1) Freeze the first 2 stages (layer1, layer2)
    freeze_first_two_stages(model)

    # (2) Freeze all 1-D parameters and report trainable parameter count
    freeze_1d_parameters(model)

    # (3) Count the number of trainable parameters.
    count_trainable_parameters(model)
