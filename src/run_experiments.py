from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import get_config
from src.bandits_contextual import ContextualThompsonSampling, LinUCB
from src.bandits_noncontext import EpsilonGreedy, GaussianThompsonSampling, UCB1
from src.utils_io import append_log, ensure_dirs, load_parquet, save_parquet
from src.utils_random import stage_seed
from src.world_model import sample_outcome_and_reward


def build_agents(cfg: dict, seed: int) -> dict:
    arm_names = cfg["arm_names"]
    d = cfg["context_dim"]
    return {
        "epsilon_greedy": EpsilonGreedy(arm_names, random_state=stage_seed(seed, cfg["seed_offsets"]["algorithms"], 1), **cfg["epsilon_greedy_default"]),
        "ucb1": UCB1(arm_names, random_state=stage_seed(seed, cfg["seed_offsets"]["algorithms"], 2), **cfg["ucb1_default"]),
        "gaussian_ts": GaussianThompsonSampling(arm_names, random_state=stage_seed(seed, cfg["seed_offsets"]["algorithms"], 3), **cfg["gaussian_ts_default"]),
        "linucb": LinUCB(arm_names, context_dim=d, random_state=stage_seed(seed, cfg["seed_offsets"]["algorithms"], 4), **cfg["linucb_default"]),
        "contextual_ts": ContextualThompsonSampling(arm_names, context_dim=d, random_state=stage_seed(seed, cfg["seed_offsets"]["algorithms"], 5), **cfg["contextual_ts_default"]),
    }


def _context_vector_from_row(row: pd.Series, cfg: dict) -> np.ndarray:
    return np.array([row[f"context_f_{j:02d}"] for j in range(cfg["context_dim"])], dtype=float)


def _base_meta_from_row(row: pd.Series) -> dict:
    return {
        "skill_match": float(row["skill_match"]),
        "keyword_match": float(row["keyword_match"]),
        "experience_match": float(row["experience_match"]),
        "education_match": float(row["education_match"]),
        "salary_alignment": float(row["salary_alignment"]),
        "location_alignment": float(row["location_alignment"]),
        "role_family_alignment": float(row["role_family_alignment"]),
        "prestige_alignment": float(row["prestige_alignment"]),
        "fit_score": float(row["fit_score"]),
        "session_goal": str(row["session_goal"]),
        "reading_patience": float(row["reading_patience"]),
        "decision_speed": float(row["decision_speed"]),
        "self_filter_strength": float(row["self_filter_strength"]),
        "resume_quality": float(row["resume_quality"]),
        "fatigue_before": float(row["fatigue_before"]),
        "fatigue_sensitivity": float(row["fatigue_sensitivity"]),
        "job_complexity": float(row["job_complexity"]),
        "scatter_apply_tendency": float(row["scatter_apply_tendency"]),
        "prestige_preference": float(row["prestige_preference"]),
        "company_prestige": float(row["company_prestige"]),
        "resume_match_weight": float(row["resume_match_weight"]),
        "experience_weight": float(row["experience_weight"]),
        "education_weight": float(row["education_weight"]),
        "keyword_weight": float(row["keyword_weight"]),
        "clarity_bonus_weight": float(row["clarity_bonus_weight"]),
        "location_weight": float(row["location_weight"]),
        "salary_weight": float(row["salary_weight"]),
        "response_rate_base": float(row["response_rate_base"]),
        "ignore_threshold": float(row["ignore_threshold"]),
        "interview_threshold": float(row["interview_threshold"]),
        "screen_noise_std": float(row["screen_noise_std"]),
        "company_value_weight": float(row["company_value_weight"]),
        "applicant_uncertainty": float(row["applicant_uncertainty"]),
        "ui_need": float(row["ui_need"]),
    }


def run_online_experiment(cfg: dict, df: pd.DataFrame, seeds: list[int], ablation_name: str = "none", max_rounds: int | None = None) -> pd.DataFrame:
    arm_profiles = cfg["arm_profiles"]
    arm_names = cfg["arm_names"]
    ablation = cfg["ablations"][ablation_name]
    rows = []
    if max_rounds is not None:
        df = df.head(max_rounds).copy()
    for seed in seeds:
        agents = build_agents(cfg, seed)
        for algo_name, agent in agents.items():
            cumulative_reward = 0.0
            cumulative_regret = 0.0
            rng = np.random.default_rng(seed + hash(algo_name) % 10_000)
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{algo_name}-{seed}"):
                context = _context_vector_from_row(row, cfg)
                arm_idx = agent.select_arm(context)
                arm_name = arm_names[arm_idx]
                base_meta = _base_meta_from_row(row)
                outcome, reward_realized, _ = sample_outcome_and_reward(
                    base_meta,
                    arm_profiles[arm_name],
                    rng,
                    arm_effect_multiplier=ablation["arm_effect_multiplier"],
                    fatigue_enabled=ablation["fatigue_enabled"],
                    screen_noise_multiplier=ablation["screen_noise_multiplier"],
                )
                agent.update(arm_idx, reward_realized, context)
                reward_expected = float(row[f"exp_reward_{arm_name}"])
                regret = float(row["oracle_expected_reward"] - reward_expected)
                cumulative_reward += reward_realized
                cumulative_regret += regret
                rows.append({
                    "seed": seed,
                    "algo_name": algo_name,
                    "mode": "online",
                    "round_id": int(row["round_id"]),
                    "arm_selected": arm_idx,
                    "arm_selected_name": arm_name,
                    "reward_realized": reward_realized,
                    "reward_expected": reward_expected,
                    "oracle_arm": int(row["oracle_arm"]),
                    "oracle_arm_name": str(row["oracle_arm_name"]),
                    "oracle_expected_reward": float(row["oracle_expected_reward"]),
                    "instant_expected_regret": regret,
                    "cumulative_reward": cumulative_reward,
                    "cumulative_expected_regret": cumulative_regret,
                    "matched_logged_action": None,
                    "updated": True,
                    "outcome": outcome,
                    "session_goal": str(row["session_goal"]),
                    "company_tier": str(row["company_tier"]),
                    "job_role_family": str(row["job_role_family"]),
                    "applicant_cohort": str(row["applicant_cohort"]),
                })
    return pd.DataFrame(rows)


def run_offline_replay_experiment(cfg: dict, df: pd.DataFrame, seeds: list[int], max_rounds: int | None = None) -> pd.DataFrame:
    rows = []
    if max_rounds is not None:
        df = df.head(max_rounds).copy()
    for seed in seeds:
        agents = build_agents(cfg, seed)
        for algo_name, agent in agents.items():
            cumulative_reward = 0.0
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"replay-{algo_name}-{seed}"):
                context = _context_vector_from_row(row, cfg)
                arm_idx = agent.select_arm(context)
                matched = arm_idx == int(row["arm_chosen"])
                reward = None
                if matched:
                    reward = float(row["reward_realized"])
                    agent.update(arm_idx, reward, context)
                    cumulative_reward += reward
                rows.append({
                    "seed": seed,
                    "algo_name": algo_name,
                    "mode": "offline_replay",
                    "round_id": int(row["round_id"]),
                    "arm_selected": arm_idx,
                    "arm_selected_name": cfg["arm_names"][arm_idx],
                    "reward_realized": reward,
                    "reward_expected": None,
                    "oracle_arm": int(row["oracle_arm"]),
                    "oracle_arm_name": str(row["oracle_arm_name"]),
                    "oracle_expected_reward": float(row["oracle_expected_reward"]),
                    "instant_expected_regret": None,
                    "cumulative_reward": cumulative_reward,
                    "cumulative_expected_regret": None,
                    "matched_logged_action": bool(matched),
                    "updated": bool(matched),
                    "outcome": row["outcome"] if matched else None,
                    "session_goal": str(row["session_goal"]),
                    "company_tier": str(row["company_tier"]),
                    "job_role_family": str(row["job_role_family"]),
                    "applicant_cohort": str(row["applicant_cohort"]),
                })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="debug")
    parser.add_argument("--mode", choices=["online", "offline_replay"], default="online")
    parser.add_argument("--seeds", nargs="+", type=int, default=[4014])
    parser.add_argument("--ablation", default="none")
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--root", type=str, default=None)
    args = parser.parse_args()

    t0 = time.time()
    cfg = get_config(args.preset, args.root)
    paths = cfg["paths"]
    ensure_dirs(paths["tables"], paths["logs"])
    df = load_parquet(paths["processed"] / "train_rounds.parquet")
    if args.mode == "online":
        results = run_online_experiment(cfg, df, args.seeds, ablation_name=args.ablation, max_rounds=args.max_rounds)
    else:
        results = run_offline_replay_experiment(cfg, df, args.seeds, max_rounds=args.max_rounds)
    out_path = paths["tables"] / f"experiment_results_{args.preset}_{args.mode}_{args.ablation}.parquet"
    save_parquet(results, out_path)
    append_log(paths["logs"] / "experiments.log", f"run_experiments,preset={args.preset},mode={args.mode},rows={len(results)},seconds={time.time()-t0:.2f}")
    print(results.head())
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
