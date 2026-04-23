from __future__ import annotations

import numpy as np

from src.utils_math import softmax


def choose_logged_arm(expected_rewards: np.ndarray, arm_names: list[str], rng: np.random.Generator, config: dict) -> tuple[int, float, np.ndarray]:
    policy_cfg = config["logging_policy"]
    shares = np.array([
        policy_cfg["oracle_softmax_share"],
        policy_cfg["uniform_random_share"],
        policy_cfg["proxy_epsilon_greedy_share"],
        policy_cfg["heuristic_share"],
    ], dtype=float)
    shares = shares / shares.sum()
    mode = int(rng.choice(4, p=shares))
    n = len(arm_names)
    if mode == 0:
        probs = softmax(expected_rewards, temperature=policy_cfg["oracle_softmax_temp"])
    elif mode == 1:
        probs = np.full(n, 1.0 / n)
    elif mode == 2:
        epsilon = policy_cfg["proxy_epsilon"]
        probs = np.full(n, epsilon / n)
        probs[int(np.argmax(expected_rewards))] += 1.0 - epsilon
    else:
        heuristic_scores = expected_rewards + np.linspace(0.0, 0.02, n)
        probs = softmax(heuristic_scores, temperature=0.25)
    arm = int(rng.choice(n, p=probs))
    return arm, float(probs[arm]), probs
