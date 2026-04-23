from __future__ import annotations

import numpy as np

from src.bandit_base import BaseBanditAgent


class LinUCB(BaseBanditAgent):
    def __init__(self, arm_names: list[str], context_dim: int, alpha: float = 0.5, l2_reg: float = 1.0, random_state: int | None = None):
        self.context_dim = context_dim
        self.alpha = alpha
        self.l2_reg = l2_reg
        super().__init__(arm_names, random_state=random_state)

    def reset(self) -> None:
        d = self.context_dim
        self.A = [np.eye(d) * self.l2_reg for _ in range(self.n_arms)]
        self.b = [np.zeros(d) for _ in range(self.n_arms)]
        self.counts = np.zeros(self.n_arms, dtype=int)

    def _theta(self, arm: int) -> np.ndarray:
        return np.linalg.solve(self.A[arm], self.b[arm])

    def select_arm(self, context: np.ndarray | None = None) -> int:
        if context is None:
            raise ValueError("LinUCB requires a context vector.")
        x = np.asarray(context, dtype=float)
        scores = []
        for arm in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv @ self.b[arm]
            scores.append(float(x @ theta + self.alpha * np.sqrt(x @ A_inv @ x)))
        return int(np.argmax(scores))

    def update(self, arm: int, reward: float, context: np.ndarray | None = None) -> None:
        if context is None:
            raise ValueError("LinUCB requires a context vector.")
        x = np.asarray(context, dtype=float)
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x
        self.counts[arm] += 1

    def predict_expected_rewards(self, context: np.ndarray | None = None) -> np.ndarray:
        if context is None:
            raise ValueError("LinUCB requires a context vector.")
        x = np.asarray(context, dtype=float)
        out = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            out[arm] = x @ self._theta(arm)
        return out


class ContextualThompsonSampling(BaseBanditAgent):
    def __init__(self, arm_names: list[str], context_dim: int, v: float = 0.35, l2_reg: float = 1.0, random_state: int | None = None):
        self.context_dim = context_dim
        self.v = v
        self.l2_reg = l2_reg
        super().__init__(arm_names, random_state=random_state)

    def reset(self) -> None:
        d = self.context_dim
        self.B = [np.eye(d) * self.l2_reg for _ in range(self.n_arms)]
        self.f = [np.zeros(d) for _ in range(self.n_arms)]
        self.mu = [np.zeros(d) for _ in range(self.n_arms)]
        self.counts = np.zeros(self.n_arms, dtype=int)

    def select_arm(self, context: np.ndarray | None = None) -> int:
        if context is None:
            raise ValueError("Contextual Thompson Sampling requires a context vector.")
        x = np.asarray(context, dtype=float)
        samples = []
        for arm in range(self.n_arms):
            cov = self.v ** 2 * np.linalg.inv(self.B[arm])
            theta_sample = self.rng.multivariate_normal(self.mu[arm], cov)
            samples.append(float(x @ theta_sample))
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float, context: np.ndarray | None = None) -> None:
        if context is None:
            raise ValueError("Contextual Thompson Sampling requires a context vector.")
        x = np.asarray(context, dtype=float)
        self.B[arm] += np.outer(x, x)
        self.f[arm] += reward * x
        self.mu[arm] = np.linalg.solve(self.B[arm], self.f[arm])
        self.counts[arm] += 1

    def predict_expected_rewards(self, context: np.ndarray | None = None) -> np.ndarray:
        if context is None:
            raise ValueError("Contextual Thompson Sampling requires a context vector.")
        x = np.asarray(context, dtype=float)
        return np.array([x @ mu for mu in self.mu], dtype=float)
