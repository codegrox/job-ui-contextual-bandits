from __future__ import annotations

import numpy as np

from src.world_model import expected_reward_for_arm


def expected_rewards_all_arms(base_meta: dict, arm_profiles: dict, arm_names: list[str], arm_effect_multiplier: float = 1.0, fatigue_enabled: bool = True) -> tuple[np.ndarray, dict]:
    rewards = []
    per_arm_meta = {}
    for arm_name in arm_names:
        r, meta = expected_reward_for_arm(base_meta, arm_profiles[arm_name], arm_effect_multiplier=arm_effect_multiplier, fatigue_enabled=fatigue_enabled)
        rewards.append(r)
        per_arm_meta[arm_name] = meta
    return np.array(rewards, dtype=float), per_arm_meta


def oracle_arm_and_reward(base_meta: dict, arm_profiles: dict, arm_names: list[str], arm_effect_multiplier: float = 1.0, fatigue_enabled: bool = True) -> tuple[int, float, np.ndarray, dict]:
    rewards, per_arm_meta = expected_rewards_all_arms(base_meta, arm_profiles, arm_names, arm_effect_multiplier=arm_effect_multiplier, fatigue_enabled=fatigue_enabled)
    best_idx = int(np.argmax(rewards))
    return best_idx, float(rewards[best_idx]), rewards, per_arm_meta
