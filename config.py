from __future__ import annotations

from pathlib import Path

PROJECT_NAME = "bt4014_job_ui_bandit"
GLOBAL_SEED = 4014
DEFAULT_PRESET = "debug"

DEBUG = {
    "n_applicants": 10_000,
    "n_companies": 500,
    "n_jobs": 8_000,
    "n_sessions": 15_000,
    "n_rounds": 50_000,
    "chunk_size": 10_000,
    "n_chunks": 5,
}

MEDIUM = {
    "n_applicants": 50_000,
    "n_companies": 2_000,
    "n_jobs": 35_000,
    "n_sessions": 80_000,
    "n_rounds": 300_000,
    "chunk_size": 50_000,
    "n_chunks": 6,
}

FULL = {
    "n_applicants": 200_000,
    "n_companies": 8_000,
    "n_jobs": 120_000,
    "n_sessions": 300_000,
    "n_rounds": 1_500_000,
    "chunk_size": 100_000,
    "n_chunks": 15,
}

PRESETS = {
    "debug": DEBUG,
    "medium": MEDIUM,
    "full": FULL,
}

N_SKILLS = 24
CONTEXT_DIM = 48

APPLICANT_COHORT_PROBS = {
    "new_grad": 0.18,
    "junior_generalist": 0.22,
    "mid_specialist": 0.22,
    "senior_specialist": 0.14,
    "career_switcher": 0.12,
    "burned_out_mass_applier": 0.12,
}

COMPANY_TIER_PROBS = {
    "small_local": 0.30,
    "startup": 0.28,
    "mid_market": 0.24,
    "large_public": 0.14,
    "elite_platform": 0.04,
}

SESSION_GOAL_PROBS = {
    "focused_search": 0.30,
    "broad_exploration": 0.28,
    "panic_apply": 0.22,
    "casual_browse": 0.20,
}

SKILL_COLUMNS = [
    "python", "sql", "ml", "statistics", "backend", "frontend",
    "devops", "cloud", "product_sense", "analytics", "communication",
    "design_tools", "ux_research", "sales_ops", "crm", "budgeting",
    "compliance", "operations_planning", "experimentation",
    "stakeholder_mgmt", "writing", "customer_support",
    "leadership", "domain_knowledge",
]

ARM_NAMES = [
    "panel_split",
    "swipe_fast",
    "card_grid",
    "guided_chat",
    "hybrid_ranked",
]

ARM_PROFILES = {
    "panel_split": {
        "arm_name": "panel_split",
        "info_richness": 0.92,
        "breadth": 0.18,
        "time_cost_base": 0.62,
        "self_filter_bonus": 0.32,
        "impulsivity": 0.06,
        "guidedness": 0.22,
    },
    "swipe_fast": {
        "arm_name": "swipe_fast",
        "info_richness": 0.10,
        "breadth": 1.00,
        "time_cost_base": 0.04,
        "self_filter_bonus": -0.32,
        "impulsivity": 1.00,
        "guidedness": 0.01,
    },
    "card_grid": {
        "arm_name": "card_grid",
        "info_richness": 0.42,
        "breadth": 0.94,
        "time_cost_base": 0.14,
        "self_filter_bonus": 0.00,
        "impulsivity": 0.36,
        "guidedness": 0.08,
    },
    "guided_chat": {
        "arm_name": "guided_chat",
        "info_richness": 0.90,
        "breadth": 0.12,
        "time_cost_base": 0.78,
        "self_filter_bonus": 0.42,
        "impulsivity": 0.03,
        "guidedness": 0.86,
    },
    "hybrid_ranked": {
        "arm_name": "hybrid_ranked",
        "info_richness": 0.54,
        "breadth": 0.40,
        "time_cost_base": 0.26,
        "self_filter_bonus": 0.08,
        "impulsivity": 0.12,
        "guidedness": 0.16,
    },
}

FIT_WEIGHTS = {
    "skill_match": 0.32,
    "experience_match": 0.18,
    "education_match": 0.08,
    "salary_alignment": 0.10,
    "location_alignment": 0.10,
    "role_family_alignment": 0.12,
    "prestige_alignment": 0.10,
}

UNCERTAINTY_WEIGHTS = {
    "one_minus_self_filter": 0.35,
    "scatter_apply_tendency": 0.20,
    "career_switcher_flag": 0.20,
    "job_complexity": 0.15,
    "fatigue": 0.10,
}

UI_NEED_WEIGHTS = {
    "job_complexity": 0.45,
    "description_length": 0.25,
    "uncertainty": 0.20,
    "one_minus_decision_speed": 0.10,
}

SUBMIT_LOGIT = {
    "bias": -1.10,
    "fit": 1.65,
    "ui_match": 0.80,
    "info_coverage": 0.55,
    "fatigue": -0.70,
    "job_complexity": -0.45,
    "scatter_apply_tendency": 0.60,
    "prestige_cross": 0.25,
}

RESPONSE_MODEL = {
    "bias": -0.40,
    "screen_multiplier": 2.40,
    "response_base_multiplier": 0.40,
}

INTERVIEW_MODEL = {
    "bias": -1.00,
    "screen_multiplier": 3.20,
}

OUTCOME_REWARD = {
    "ignored": 0.00,
    "rejected": 0.08,
    "interview_1": 1.00,
}

FRICTION_PENALTY = {
    "dwell_seconds_weight": 0.06 / 120.0,
    "fatigue_increment_weight": 0.05,
}

COMPANY_VALUE_RANGE = (0.80, 1.50)
NOISE_BEHAVIOR_STD = 0.05
SCREEN_NOISE_BY_TIER = {
    "small_local": 0.07,
    "startup": 0.08,
    "mid_market": 0.07,
    "large_public": 0.06,
    "elite_platform": 0.05,
}

EPSILON_GREEDY_DEFAULT = {
    "epsilon": 0.05,
    "optimistic_init": 0.0,
}
UCB1_DEFAULT = {
    "exploration_coef": 2.0,
}
GAUSSIAN_TS_DEFAULT = {
    "prior_mean": 0.0,
    "prior_var": 1.0,
    "obs_var": 0.25,
}
LINUCB_DEFAULT = {
    "alpha": 0.20,
    "l2_reg": 1.0,
}
CONTEXTUAL_TS_DEFAULT = {
    "v": 0.15,
    "l2_reg": 1.0,
}

HYPERPARAM_GRID_DEBUG = {
    "epsilon_greedy": [{"epsilon": 0.05}],
    "ucb1": [{"exploration_coef": 2.0}],
    "gaussian_ts": [{"prior_var": 1.0, "obs_var": 0.25}],
    "linucb": [{"alpha": 0.50}],
    "contextual_ts": [{"v": 0.35}],
}

HYPERPARAM_GRID_MEDIUM = {
    "epsilon_greedy": [{"epsilon": 0.03}, {"epsilon": 0.05}, {"epsilon": 0.10}],
    "ucb1": [{"exploration_coef": 1.5}, {"exploration_coef": 2.0}],
    "gaussian_ts": [
        {"prior_var": 1.0, "obs_var": 0.20},
        {"prior_var": 1.0, "obs_var": 0.25},
    ],
    "linucb": [{"alpha": 0.20}, {"alpha": 0.30}, {"alpha": 0.45}],
    "contextual_ts": [{"v": 0.15}, {"v": 0.20}, {"v": 0.30}],
}

SEED_LIST_DEBUG = [4014]
SEED_LIST_MEDIUM = [4014, 4015, 4016]
SEED_LIST_FULL = [4014, 4015, 4016]
SEED_OFFSETS = {
    "applicants": 11,
    "companies": 23,
    "jobs": 37,
    "sessions": 41,
    "rounds": 53,
    "algorithms": 67,
}

LOGGING_POLICY = {
    "mode": "mixed",
    "oracle_softmax_share": 0.40,
    "uniform_random_share": 0.25,
    "proxy_epsilon_greedy_share": 0.20,
    "heuristic_share": 0.15,
    "oracle_softmax_temp": 0.15,
    "proxy_epsilon": 0.10,
}

DATA_SPLIT = {
    "train_frac": 0.70,
    "valid_frac": 0.15,
    "test_frac": 0.15,
}

DIAGNOSTIC_THRESHOLDS = {
    "min_interview_rate": 0.03,
    "max_interview_rate": 0.18,
    "min_oracle_arm_share": 0.06,
    "max_oracle_arm_share": 0.40,
    "min_median_oracle_gap": 0.02,
    "max_median_oracle_gap": 0.12,
}

ABLATIONS = {
    "none": {
        "arm_effect_multiplier": 1.00,
        "fatigue_enabled": True,
        "screen_noise_multiplier": 1.00,
        "drop_context_blocks": [],
        "n_rounds_override": None,
    },
    "weak_ui": {
        "arm_effect_multiplier": 0.50,
        "fatigue_enabled": True,
        "screen_noise_multiplier": 1.00,
        "drop_context_blocks": [],
        "n_rounds_override": None,
    },
    "strong_ui": {
        "arm_effect_multiplier": 1.35,
        "fatigue_enabled": True,
        "screen_noise_multiplier": 1.00,
        "drop_context_blocks": [],
        "n_rounds_override": None,
    },
    "no_fatigue": {
        "arm_effect_multiplier": 1.00,
        "fatigue_enabled": False,
        "screen_noise_multiplier": 1.00,
        "drop_context_blocks": [],
        "n_rounds_override": None,
    },
    "high_noise": {
        "arm_effect_multiplier": 1.00,
        "fatigue_enabled": True,
        "screen_noise_multiplier": 1.50,
        "drop_context_blocks": [],
        "n_rounds_override": None,
    },
    "short_horizon": {
        "arm_effect_multiplier": 1.00,
        "fatigue_enabled": True,
        "screen_noise_multiplier": 1.00,
        "drop_context_blocks": [],
        "n_rounds_override": 20_000,
    },
    "no_session_goal_features": {
        "arm_effect_multiplier": 1.00,
        "fatigue_enabled": True,
        "screen_noise_multiplier": 1.00,
        "drop_context_blocks": ["session_goal"],
        "n_rounds_override": None,
    },
}

PLOT_DEFAULTS = {
    "dpi": 220,
    "save_format": "png",
    "rolling_window_debug": 200,
    "rolling_window_full": 2000,
}

TABLE_EXPORT = {
    "float_format": "%.4f",
    "latex_escape": False,
}


def get_paths(root: str | Path | None = None) -> dict[str, Path]:
    base = Path(root) if root is not None else Path(__file__).resolve().parent
    return {
        "root": base,
        "data": base / "data",
        "raw": base / "data" / "raw",
        "logged": base / "data" / "logged",
        "processed": base / "data" / "processed",
        "summaries": base / "data" / "summaries",
        "outputs": base / "outputs",
        "figures": base / "outputs" / "figures",
        "tables": base / "outputs" / "tables",
        "logs": base / "outputs" / "logs",
        "submission": base / "submission",
        "notebooks": base / "notebooks",
        "src": base / "src",
    }


def get_config(preset: str = DEFAULT_PRESET, root: str | Path | None = None) -> dict:
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}")
    cfg = {
        "project_name": PROJECT_NAME,
        "global_seed": GLOBAL_SEED,
        "preset": preset,
        "sizes": PRESETS[preset].copy(),
        "paths": get_paths(root),
        "arm_names": ARM_NAMES,
        "arm_profiles": ARM_PROFILES,
        "skill_columns": SKILL_COLUMNS,
        "fit_weights": FIT_WEIGHTS,
        "uncertainty_weights": UNCERTAINTY_WEIGHTS,
        "ui_need_weights": UI_NEED_WEIGHTS,
        "submit_logit": SUBMIT_LOGIT,
        "response_model": RESPONSE_MODEL,
        "interview_model": INTERVIEW_MODEL,
        "outcome_reward": OUTCOME_REWARD,
        "friction_penalty": FRICTION_PENALTY,
        "company_value_range": COMPANY_VALUE_RANGE,
        "noise_behavior_std": NOISE_BEHAVIOR_STD,
        "screen_noise_by_tier": SCREEN_NOISE_BY_TIER,
        "epsilon_greedy_default": EPSILON_GREEDY_DEFAULT,
        "ucb1_default": UCB1_DEFAULT,
        "gaussian_ts_default": GAUSSIAN_TS_DEFAULT,
        "linucb_default": LINUCB_DEFAULT,
        "contextual_ts_default": CONTEXTUAL_TS_DEFAULT,
        "hyperparam_grid_debug": HYPERPARAM_GRID_DEBUG,
        "hyperparam_grid_medium": HYPERPARAM_GRID_MEDIUM,
        "seed_offsets": SEED_OFFSETS,
        "logging_policy": LOGGING_POLICY,
        "data_split": DATA_SPLIT,
        "diagnostic_thresholds": DIAGNOSTIC_THRESHOLDS,
        "ablations": ABLATIONS,
        "plot_defaults": PLOT_DEFAULTS,
        "table_export": TABLE_EXPORT,
        "context_dim": CONTEXT_DIM,
        "n_skills": N_SKILLS,
        "applicant_cohort_probs": APPLICANT_COHORT_PROBS,
        "company_tier_probs": COMPANY_TIER_PROBS,
        "session_goal_probs": SESSION_GOAL_PROBS,
    }
    return cfg
