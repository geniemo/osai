"""Utilities: seed, timing, checkpoint IO."""
from __future__ import annotations
import os
import random
import threading
import time
from pathlib import Path
import numpy as np
import torch


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_checkpoint_atomic(path: Path, state: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def async_save_checkpoint(path: Path, state: dict) -> threading.Thread:
    t = threading.Thread(target=save_checkpoint_atomic, args=(path, state), daemon=False)
    t.start()
    return t


class ThroughputMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.t0 = time.perf_counter()
        self.n = 0

    def add(self, n: int):
        self.n += n

    def rate(self) -> float:
        dt = max(time.perf_counter() - self.t0, 1e-6)
        return self.n / dt
