import random
import numpy as np
import torch
from src.utils import seed as seed_mod


def test_set_seed_makes_random_deterministic():
    seed_mod.set_seed(42)
    a = (random.random(), np.random.rand(), torch.rand(1).item())
    seed_mod.set_seed(42)
    b = (random.random(), np.random.rand(), torch.rand(1).item())
    assert a == b


def test_get_set_rng_state_roundtrip():
    seed_mod.set_seed(0)
    state = seed_mod.get_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(1).item())
    seed_mod.set_seed(123)
    seed_mod.set_rng_state(state)
    actual = (random.random(), np.random.rand(), torch.rand(1).item())
    assert actual == expected


def test_worker_init_seeds_workers_distinctly():
    seed_mod.set_seed(0)
    seed_mod.worker_init_fn(0)
    s0 = (random.random(), np.random.rand())
    seed_mod.set_seed(0)
    seed_mod.worker_init_fn(1)
    s1 = (random.random(), np.random.rand())
    assert s0 != s1
