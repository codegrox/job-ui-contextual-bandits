from __future__ import annotations

import numpy as np


def clip01(x):
    return np.clip(x, 0.0, 1.0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x, temperature: float = 1.0):
    x = np.asarray(x, dtype=float) / max(temperature, 1e-8)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def cosine_similarity(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)
    if nx < 1e-12 or ny < 1e-12:
        return 0.0
    return float(np.dot(x, y) / (nx * ny))


def normalized_index(idx: int, n_items: int) -> float:
    if n_items <= 1:
        return 0.0
    return idx / (n_items - 1)
