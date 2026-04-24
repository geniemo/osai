"""Checkpoint helpers: full training state vs model-only."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.utils.seed import get_rng_state, set_rng_state


def save_full(
    *,
    path: str,
    iter_count: int,
    stage: int,
    model: nn.Module,
    ema_model: Optional[nn.Module],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    best_miou: float,
    wandb_run_id: Optional[str],
    config: Dict[str, Any],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "iter": iter_count,
        "stage": stage,
        "model_state": model.state_dict(),
        "ema_state": ema_model.state_dict() if ema_model is not None else None,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_miou": best_miou,
        "rng_state": get_rng_state(),
        "wandb_run_id": wandb_run_id,
        "config": config,
    }
    torch.save(state, path)


def load_full(
    *,
    path: str,
    model: nn.Module,
    ema_model: Optional[nn.Module],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model_state"])
    if ema_model is not None and state.get("ema_state") is not None:
        ema_model.load_state_dict(state["ema_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    scheduler.load_state_dict(state["scheduler_state"])
    scaler.load_state_dict(state["scaler_state"])
    set_rng_state(state["rng_state"])
    return {
        "iter": state["iter"],
        "stage": state["stage"],
        "best_miou": state["best_miou"],
        "wandb_run_id": state["wandb_run_id"],
        "config": state["config"],
    }


def save_model_only(path: str, model: nn.Module) -> None:
    """제출용: model_state만 저장 (가벼움, EMA model로 호출)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
