from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseBanditAgent(ABC):
    def __init__(self, arm_names: list[str], random_state: int | None = None):
        self.arm_names = arm_names
        self.n_arms = len(arm_names)
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.reset()

    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def select_arm(self, context: np.ndarray | None = None) -> int:
        ...

    @abstractmethod
    def update(self, arm: int, reward: float, context: np.ndarray | None = None) -> None:
        ...

    @abstractmethod
    def predict_expected_rewards(self, context: np.ndarray | None = None) -> np.ndarray:
        ...

    def greedy_arm(self, context: np.ndarray | None = None) -> int:
        return int(np.argmax(self.predict_expected_rewards(context)))

    def get_state_summary(self) -> dict[str, Any]:
        return {"n_arms": self.n_arms}
