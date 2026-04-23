from __future__ import annotations

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def stage_seed(global_seed: int, offset: int, extra: int = 0) -> int:
    return int(global_seed + offset + extra)


def stage_rng(global_seed: int, offset: int, extra: int = 0) -> np.random.Generator:
    return make_rng(stage_seed(global_seed, offset, extra))
