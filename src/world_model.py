from __future__ import annotations

import numpy as np

from config import FIT_WEIGHTS
from src.utils_math import clip01, sigmoid


def compute_fit_score(components: dict[str, float]) -> float:
    val = 0.0
    for k, w in FIT_WEIGHTS.items():
        val += w * float(components[k])
    return float(clip01(val))


def _goal_flags(meta: dict) -> dict[str, float]:
    goal = str(meta["session_goal"])
    return {
        "focused": 1.0 if goal == "focused_search" else 0.0,
        "broad": 1.0 if goal == "broad_exploration" else 0.0,
        "panic": 1.0 if goal == "panic_apply" else 0.0,
        "casual": 1.0 if goal == "casual_browse" else 0.0,
    }


def _mid_pref(x: float, center: float = 0.5, width: float = 0.35) -> float:
    return float(clip01(1.0 - abs(float(x) - center) / width))


def compute_ui_match(row_or_meta: dict, arm_profile: dict, arm_effect_multiplier: float = 1.0) -> float:
    flags = _goal_flags(row_or_meta)
    ui_need = float(row_or_meta["ui_need"])
    uncertainty = float(row_or_meta["applicant_uncertainty"])
    reading_patience = float(row_or_meta["reading_patience"])
    decision_speed = float(row_or_meta["decision_speed"])
    fatigue = float(row_or_meta["fatigue_before"])
    arm_name = arm_profile.get("arm_name", "")

    balanced_need = _mid_pref(ui_need, 0.45, 0.35)
    balanced_unc = _mid_pref(uncertainty, 0.45, 0.35)

    base = (
        0.08 * arm_profile["info_richness"] * ui_need
        + 0.04 * arm_profile["breadth"] * flags["broad"]
        + 0.03 * arm_profile["breadth"] * flags["casual"]
        + 0.06 * (arm_profile["self_filter_bonus"] + 0.5) * uncertainty
        + 0.04 * (1.0 - arm_profile["time_cost_base"]) * decision_speed
        + 0.04 * reading_patience * arm_profile["info_richness"]
        - 0.03 * fatigue * arm_profile["time_cost_base"]
    )

    bonus = 0.0
    if arm_name == "panel_split":
        bonus += 0.50 * flags["focused"] * (0.30 + ui_need)
        bonus += 0.18 * reading_patience * ui_need
        bonus += 0.14 * flags["focused"] * (1.0 - uncertainty)
        bonus -= 0.18 * flags["broad"]
        bonus -= 0.18 * flags["panic"]
        bonus -= 0.08 * flags["casual"]
    elif arm_name == "swipe_fast":
        bonus += 0.50 * flags["panic"] * (0.30 + decision_speed)
        bonus += 0.18 * flags["panic"] * (1.0 - ui_need)
        bonus += 0.18 * flags["broad"] * decision_speed
        bonus += 0.10 * flags["casual"] * decision_speed
        bonus -= 0.22 * flags["focused"]
        bonus -= 0.18 * ui_need
        bonus -= 0.12 * reading_patience
        bonus -= 0.06 * uncertainty
    elif arm_name == "card_grid":
        bonus += 0.34 * flags["broad"]
        bonus += 0.18 * flags["casual"]
        bonus += 0.14 * flags["broad"] * (1.0 - ui_need)
        bonus += 0.08 * flags["casual"] * (1.0 - ui_need)
        bonus -= 0.18 * flags["focused"]
        bonus -= 0.10 * flags["panic"]
        bonus -= 0.10 * ui_need
    elif arm_name == "guided_chat":
        bonus += 0.42 * flags["focused"] * uncertainty
        bonus += 0.26 * flags["focused"] * ui_need
        bonus += 0.12 * uncertainty * ui_need
        bonus += 0.06 * flags["casual"] * uncertainty
        bonus -= 0.16 * flags["broad"]
        bonus -= 0.20 * flags["panic"]
        bonus -= 0.06 * fatigue
    elif arm_name == "hybrid_ranked":
        bonus += 0.12 * flags["casual"] * balanced_need
        bonus += 0.05 * flags["broad"] * balanced_need
        bonus += 0.08 * balanced_need * balanced_unc
        bonus += 0.03 * (1.0 - fatigue)
        bonus -= 0.14 * flags["focused"] * ui_need
        bonus -= 0.14 * flags["panic"]
        bonus -= 0.10 * flags["broad"]

    return float(clip01(base + arm_effect_multiplier * bonus))


def simulate_behavior_intermediates(meta: dict, arm_profile: dict, rng: np.random.Generator | None = None, arm_effect_multiplier: float = 1.0, fatigue_enabled: bool = True) -> dict[str, float]:
    rng = np.random.default_rng() if rng is None else rng
    flags = _goal_flags(meta)
    fatigue = float(meta["fatigue_before"])
    ui_need = float(meta["ui_need"])
    uncertainty = float(meta["applicant_uncertainty"])
    reading_patience = float(meta["reading_patience"])
    balanced_need = _mid_pref(ui_need, 0.45, 0.35)
    balanced_unc = _mid_pref(uncertainty, 0.45, 0.35)
    arm_name = arm_profile.get("arm_name", "")

    info_coverage = (
        0.24 * arm_profile["info_richness"]
        + 0.14 * reading_patience
        + 0.06 * ui_need
        - 0.06 * fatigue
    )
    if arm_name == "panel_split":
        info_coverage += 0.24 * flags["focused"] + 0.14 * ui_need + 0.06 * reading_patience
        info_coverage -= 0.06 * uncertainty * flags["focused"]
    elif arm_name == "swipe_fast":
        info_coverage += 0.12 * flags["panic"] + 0.08 * flags["broad"] - 0.20 * ui_need - 0.06 * uncertainty
    elif arm_name == "card_grid":
        info_coverage += 0.18 * flags["broad"] + 0.08 * flags["casual"] - 0.10 * ui_need
    elif arm_name == "guided_chat":
        info_coverage += 0.20 * flags["focused"] + 0.16 * uncertainty + 0.10 * ui_need - 0.10 * flags["panic"]
    elif arm_name == "hybrid_ranked":
        info_coverage += 0.08 * balanced_need + 0.06 * balanced_unc + 0.05 * flags["casual"] - 0.05 * flags["panic"]
    info_coverage = clip01(info_coverage + rng.normal(0, 0.04))

    fit_comprehension = (
        0.34 * info_coverage
        + 0.22 * float(meta["self_filter_strength"])
        + 0.10 * float(meta["resume_quality"])
        + 0.10 * arm_profile["guidedness"]
    )
    if arm_name == "panel_split":
        fit_comprehension += 0.18 * ui_need * reading_patience + 0.06 * flags["focused"] - 0.08 * uncertainty * flags["focused"]
    elif arm_name == "guided_chat":
        fit_comprehension += 0.24 * uncertainty * flags["focused"] + 0.12 * ui_need - 0.10 * flags["panic"]
    elif arm_name == "swipe_fast":
        fit_comprehension += 0.12 * flags["panic"] - 0.22 * ui_need - 0.06 * flags["focused"]
    elif arm_name == "card_grid":
        fit_comprehension += 0.12 * flags["broad"] + 0.06 * flags["casual"] - 0.10 * ui_need
    elif arm_name == "hybrid_ranked":
        fit_comprehension += 0.08 * balanced_need + 0.06 * balanced_unc + 0.04 * flags["casual"]
    fit_comprehension = clip01(fit_comprehension + rng.normal(0, 0.04))

    dwell_seconds = (
        18.0
        + 52.0 * arm_profile["time_cost_base"]
        + 32.0 * float(meta["job_complexity"])
        + 18.0 * reading_patience
        - 20.0 * fatigue
    )
    if arm_name == "swipe_fast":
        dwell_seconds -= 22.0 * flags["panic"] + 14.0 * flags["broad"] + 10.0 * flags["casual"]
    elif arm_name == "card_grid":
        dwell_seconds -= 10.0 * flags["broad"] + 8.0 * flags["casual"]
    elif arm_name == "guided_chat":
        dwell_seconds += 12.0 * flags["focused"] + 8.0 * uncertainty
    elif arm_name == "panel_split":
        dwell_seconds += 10.0 * flags["focused"]
    elif arm_name == "hybrid_ranked":
        dwell_seconds += 2.0 * balanced_need - 4.0 * flags["casual"]
    dwell_seconds = max(3.0, dwell_seconds + rng.normal(0, 6.0))

    fatigue_increment = 0.0
    if fatigue_enabled:
        fatigue_increment = (
            0.10 * arm_profile["time_cost_base"]
            + 0.06 * float(meta["job_complexity"])
            + 0.08 * float(meta["fatigue_sensitivity"])
            - 0.05 * float(meta["ui_match_score"])
        )
        if arm_name == "guided_chat":
            fatigue_increment += 0.01 * flags["broad"] + 0.01 * flags["casual"]
        if arm_name == "swipe_fast":
            fatigue_increment -= 0.04 * flags["panic"]
        if arm_name == "card_grid":
            fatigue_increment -= 0.02 * flags["broad"]
        if arm_name == "hybrid_ranked":
            fatigue_increment -= 0.01 * balanced_need
        fatigue_increment = clip01(fatigue_increment + rng.normal(0, 0.025))

    self_filter_quality = (
        0.30 * fit_comprehension
        + 0.22 * float(meta["self_filter_strength"])
        + 0.10 * arm_profile["guidedness"]
        - 0.16 * arm_profile["impulsivity"]
    )
    if arm_name == "panel_split":
        self_filter_quality += 0.18 * ui_need * reading_patience + 0.08 * flags["focused"] - 0.08 * uncertainty * flags["focused"]
    elif arm_name == "guided_chat":
        self_filter_quality += 0.26 * uncertainty * flags["focused"] + 0.10 * ui_need - 0.10 * flags["broad"] - 0.08 * flags["casual"]
    elif arm_name == "swipe_fast":
        self_filter_quality += 0.10 * flags["panic"] - 0.22 * ui_need - 0.06 * flags["focused"]
    elif arm_name == "card_grid":
        self_filter_quality += 0.12 * flags["broad"] + 0.06 * flags["casual"] - 0.10 * ui_need
    elif arm_name == "hybrid_ranked":
        self_filter_quality += 0.08 * balanced_need + 0.06 * balanced_unc + 0.05 * flags["casual"]
    self_filter_quality = clip01(self_filter_quality + rng.normal(0, 0.04))

    return {
        "info_coverage": float(info_coverage),
        "fit_comprehension": float(fit_comprehension),
        "dwell_time_seconds": float(dwell_seconds),
        "fatigue_increment": float(fatigue_increment),
        "self_filter_quality": float(self_filter_quality),
    }


def compute_submit_probability(meta: dict) -> float:
    logit = (
        -1.20
        + 1.65 * float(meta["fit_score"])
        + 0.95 * float(meta["ui_match_score"])
        + 0.50 * float(meta["info_coverage"])
        - 0.65 * float(meta["fatigue_before"])
        - 0.38 * float(meta["job_complexity"])
        + 0.60 * float(meta["scatter_apply_tendency"])
        + 0.18 * float(meta["prestige_preference"]) * float(meta["company_prestige"])
    )
    return float(sigmoid(logit))


def compute_screen_score(meta: dict, rng: np.random.Generator | None = None, screen_noise_multiplier: float = 1.0) -> float:
    rng = np.random.default_rng() if rng is None else rng
    clarity_bonus = 0.50 * float(meta["fit_comprehension"]) + 0.50 * float(meta["self_filter_quality"])
    val = (
        float(meta["resume_match_weight"]) * float(meta["skill_match"])
        + float(meta["experience_weight"]) * float(meta["experience_match"])
        + float(meta["education_weight"]) * float(meta["education_match"])
        + float(meta["keyword_weight"]) * float(meta["keyword_match"])
        + float(meta["clarity_bonus_weight"]) * clarity_bonus
        + float(meta["location_weight"]) * float(meta["location_alignment"])
        + float(meta["salary_weight"]) * float(meta["salary_alignment"])
    )
    val += rng.normal(0.0, float(meta["screen_noise_std"]) * screen_noise_multiplier)
    return float(clip01(val))


def outcome_probabilities(meta: dict, screen_score: float) -> dict[str, float]:
    p_submit = float(meta["submit_probability"])
    p_respond = float(sigmoid(-0.40 + 2.40 * (screen_score - float(meta["ignore_threshold"])) + 0.40 * float(meta["response_rate_base"])))
    p_interview_given_respond = float(sigmoid(-1.00 + 3.20 * (screen_score - float(meta["interview_threshold"]))))
    p_interview = p_submit * p_respond * p_interview_given_respond
    p_rejected = p_submit * p_respond * (1.0 - p_interview_given_respond)
    p_ignored = max(0.0, 1.0 - p_interview - p_rejected)
    return {
        "p_submit": p_submit,
        "p_respond": p_respond,
        "p_interview_given_respond": p_interview_given_respond,
        "p_interview": p_interview,
        "p_rejected": p_rejected,
        "p_ignored": p_ignored,
    }


def compute_reward(meta: dict, outcome: str) -> float:
    outcome_reward = {
        "ignored": 0.0,
        "rejected": 0.08,
        "interview_1": 1.0,
    }[outcome]
    base = float(meta["company_value_weight"]) * outcome_reward
    friction = 0.06 * float(meta["dwell_time_seconds"]) / 120.0 + 0.05 * float(meta["fatigue_increment"])
    return float(max(0.0, base - friction))


def expected_reward_for_arm(base_meta: dict, arm_profile: dict, arm_effect_multiplier: float = 1.0, fatigue_enabled: bool = True) -> tuple[float, dict]:
    meta = dict(base_meta)
    meta["ui_match_score"] = compute_ui_match(meta, arm_profile, arm_effect_multiplier=arm_effect_multiplier)
    behavior = simulate_behavior_intermediates(meta, arm_profile, rng=np.random.default_rng(12345), arm_effect_multiplier=arm_effect_multiplier, fatigue_enabled=fatigue_enabled)
    meta.update(behavior)
    meta["submit_probability"] = compute_submit_probability(meta)
    screen_score = compute_screen_score(meta, rng=np.random.default_rng(54321), screen_noise_multiplier=1.0)
    probs = outcome_probabilities(meta, screen_score)
    expected_reward = (
        probs["p_ignored"] * compute_reward(meta, "ignored")
        + probs["p_rejected"] * compute_reward(meta, "rejected")
        + probs["p_interview"] * compute_reward(meta, "interview_1")
    )
    meta.update({"screen_score": screen_score})
    meta.update(probs)
    return float(expected_reward), meta


def sample_realized_outcome_and_reward(base_meta: dict, arm_profile: dict, rng: np.random.Generator | None = None, arm_effect_multiplier: float = 1.0, fatigue_enabled: bool = True, screen_noise_multiplier: float = 1.0) -> dict:
    rng = np.random.default_rng() if rng is None else rng
    meta = dict(base_meta)
    meta["ui_match_score"] = compute_ui_match(meta, arm_profile, arm_effect_multiplier=arm_effect_multiplier)
    behavior = simulate_behavior_intermediates(meta, arm_profile, rng=rng, arm_effect_multiplier=arm_effect_multiplier, fatigue_enabled=fatigue_enabled)
    meta.update(behavior)
    meta["submit_probability"] = compute_submit_probability(meta)
    submit = rng.random() < meta["submit_probability"]
    if not submit:
        meta["screen_score"] = 0.0
        probs = {
            "p_submit": meta["submit_probability"],
            "p_respond": 0.0,
            "p_interview_given_respond": 0.0,
            "p_interview": 0.0,
            "p_rejected": 0.0,
            "p_ignored": 1.0,
        }
        meta.update(probs)
        meta["submitted"] = 0
        meta["outcome"] = "ignored"
        meta["reward_realized"] = compute_reward(meta, "ignored")
        return meta

    meta["submitted"] = 1
    screen_score = compute_screen_score(meta, rng=rng, screen_noise_multiplier=screen_noise_multiplier)
    meta["screen_score"] = screen_score
    probs = outcome_probabilities(meta, screen_score)
    meta.update(probs)

    responded = rng.random() < probs["p_respond"]
    if not responded:
        outcome = "ignored"
    else:
        interviewed = rng.random() < probs["p_interview_given_respond"]
        outcome = "interview_1" if interviewed else "rejected"

    meta["outcome"] = outcome
    meta["reward_realized"] = compute_reward(meta, outcome)
    return meta


def sample_outcome_and_reward(base_meta: dict, arm_profile: dict, rng: np.random.Generator | None = None, arm_effect_multiplier: float = 1.0, fatigue_enabled: bool = True, screen_noise_multiplier: float = 1.0) -> tuple[str, float, dict]:
    meta = sample_realized_outcome_and_reward(
        base_meta=base_meta,
        arm_profile=arm_profile,
        rng=rng,
        arm_effect_multiplier=arm_effect_multiplier,
        fatigue_enabled=fatigue_enabled,
        screen_noise_multiplier=screen_noise_multiplier,
    )
    return str(meta["outcome"]), float(meta["reward_realized"]), meta
