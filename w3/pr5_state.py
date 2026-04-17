"""
pr5_state.py
Practice 5: Saving, loading, and remapping state_dicts for ResNet34 from torchvision.

Tasks:
  (1) Save the model's state_dict to "model.pt".
  (2) Load the state_dict back with load_state_dict and verify the outputs match.
  (3) Wrap the model in a new nn.Module (self.module = net).
  (4) Remap the saved state_dict keys so they match the wrapped model's
      parameter names, then load it with load_state_dict.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet34


# ---------------------------------------------------------------------------
# Wrapper model
# ---------------------------------------------------------------------------

class WrappedModel(nn.Module):
    """
    A thin wrapper that stores the original network as self.module.

    When a model is wrapped this way (e.g., by nn.DataParallel or custom
    wrappers), its state_dict keys are prefixed with "module." instead of
    referring directly to the sub-layers. For example:
        Original key  : "layer1.0.conv1.weight"
        Wrapped key   : "module.layer1.0.conv1.weight"
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.module = net  # store the original model under the name "module"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_model() -> nn.Module:
    """
    Load a ResNet34 model from torchvision without pretrained weights.

    Returns:
        model (nn.Module): ResNet34 model with random initialization.
    """
    model = resnet34(weights=None)
    return model


def save_state_dict(model: nn.Module, path: str = "model.pt"):
    """
    (1) Save the model's state_dict to a file.

    state_dict() returns an OrderedDict mapping parameter/buffer names
    to their tensor values. Only parameters and buffers are saved,
    NOT the model architecture.

    Args:
        model (nn.Module): The model whose state_dict will be saved.
        path  (str)      : File path for saving (default: "model.pt").
    """
    torch.save(model.state_dict(), path)


def load_state_dict(model: nn.Module, path: str = "model.pt"):
    """
    (2) Load the saved state_dict into a fresh model and verify that
    the outputs for a fixed input are identical to the original model.

    Args:
        model          (nn.Module): The original model to compare against.
        path           (str)      : File path of the saved state_dict.
    """
    new_model = resnet34(weights=None)
    state_dict = torch.load(path, weights_only=True)
    new_model.load_state_dict(state_dict)

    dummy = torch.randn(1, 3, 224, 224)
    model.eval()
    new_model.eval()

    with torch.no_grad():
        out_original = model(dummy)
        out_loaded = new_model(dummy)

    match = torch.allclose(out_original, out_loaded)


def wrap_model(net: nn.Module) -> WrappedModel:
    """
    (3) Wrap the given model in a WrappedModel so its parameters live
    under the "module.*" namespace.

    Args:
        net (nn.Module): The model to wrap.

    Returns:
        wrapped (WrappedModel): The wrapped model.
    """
    wrapped = WrappedModel(net)
    return wrapped


def remap_and_load(wrapped_model: WrappedModel, path: str = "model.pt"):
    """
    (4) Remap the keys in the saved state_dict by prepending "module."
    to each key, then load it into the wrapped model.

    The original state_dict has keys like "layer1.0.conv1.weight".
    The wrapped model expects keys like "module.layer1.0.conv1.weight".

    Args:
        wrapped_model (WrappedModel): The target model to load weights into.
        path          (str)         : File path of the original state_dict.
    """
    state_dict = torch.load(path, weights_only=True)
    new_state_dict = {f"module.{k}": v for k, v in state_dict.items()}
    wrapped_model.load_state_dict(new_state_dict)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SAVE_PATH = "model.pt"

    # Load a ResNet34 model
    model = load_model()

    # (1) Save state_dict
    save_state_dict(model, path=SAVE_PATH)

    # (2) Load state_dict and verify outputs match
    load_state_dict(model, path=SAVE_PATH)

    # (3) Wrap the loaded model under self.module
    wrapped = wrap_model(model)

    # (4) Remap keys and load into the wrapped model
    remap_and_load(wrapped, path=SAVE_PATH)