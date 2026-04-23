from __future__ import annotations

import numpy as np

from src.bandit_base import BaseBanditAgent


class EpsilonGreedy(BaseBanditAgent):
    def __init__(self, arm_names: list[str], epsilon: float = 0.05, optimistic_init: float = 0.0, random_state: int | None = None):
        self.epsilon = epsilon
        self.optimistic_init = optimistic_init
        super().__init__(arm_names, random_state=random_state)

    def reset(self) -> None:
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.value_estimates = np.full(self.n_arms, self.optimistic_init, dtype=float)

    def select_arm(self, context=None) -> int:
        if self.rng.random() < self.epsilon or np.all(self.counts == 0):
            return int(self.rng.integers(0, self.n_arms))
        return int(np.argmax(self.value_estimates))

    def update(self, arm: int, reward: float, context=None) -> None:
        self.counts[arm] += 1
        n = self.counts[arm]
        self.value_estimates[arm] += (reward - self.value_estimates[arm]) / n

    def predict_expected_rewards(self, context=None) -> np.ndarray:
        return self.value_estimates.copy()


class UCB1(BaseBanditAgent):
    def __init__(self, arm_names: list[str], exploration_coef: float = 2.0, random_state: int | None = None):
        self.exploration_coef = exploration_coef
        super().__init__(arm_names, random_state=random_state)

    def reset(self) -> None:
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.value_estimates = np.zeros(self.n_arms, dtype=float)
        self.t = 0

    def select_arm(self, context=None) -> int:
        self.t += 1
        unpulled = np.where(self.counts == 0)[0]
        if len(unpulled) > 0:
            return int(unpulled[0])
        bonus = self.exploration_coef * np.sqrt(np.log(max(self.t, 2)) / self.counts)
        return int(np.argmax(self.value_estimates + bonus))

    def update(self, arm: int, reward: float, context=None) -> None:
        self.counts[arm] += 1
        n = self.counts[arm]
        self.value_estimates[arm] += (reward - self.value_estimates[arm]) / n

    def predict_expected_rewards(self, context=None) -> np.ndarray:
        return self.value_estimates.copy()


class GaussianThompsonSampling(BaseBanditAgent):
    def __init__(self, arm_names: list[str], prior_mean: float = 0.0, prior_var: float = 1.0, obs_var: float = 0.25, random_state: int | None = None):
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.obs_var = obs_var
        super().__init__(arm_names, random_state=random_state)

    def reset(self) -> None:
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.reward_sums = np.zeros(self.n_arms, dtype=float)
        self.posterior_means = np.full(self.n_arms, self.prior_mean, dtype=float)
        self.posterior_vars = np.full(self.n_arms, self.prior_var, dtype=float)

    def select_arm(self, context=None) -> int:
        samples = self.rng.normal(self.posterior_means, np.sqrt(self.posterior_vars))
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float, context=None) -> None:
        self.counts[arm] += 1
        self.reward_sums[arm] += reward
        precision_prior = 1.0 / self.prior_var
        precision_lik = self.counts[arm] / self.obs_var
        post_var = 1.0 / (precision_prior + precision_lik)
        post_mean = post_var * (precision_prior * self.prior_mean + self.reward_sums[arm] / self.obs_var)
        self.posterior_vars[arm] = post_var
        self.posterior_means[arm] = post_mean

    def predict_expected_rewards(self, context=None) -> np.ndarray:
        return self.posterior_means.copy()
