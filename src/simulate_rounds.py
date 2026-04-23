from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import get_config
from src.feature_engineering import compute_fit_components, build_context_vector
from src.logging_policy import choose_logged_arm
from src.oracle import oracle_arm_and_reward
from src.utils_io import append_log, ensure_dirs, load_json, load_parquet, save_parquet
from src.utils_random import stage_rng
from src.world_model import compute_fit_score, sample_outcome_and_reward


def _make_job_index(jobs: pd.DataFrame) -> dict[str, np.ndarray]:
    out = {}
    for role, grp in jobs.groupby("role_family"):
        out[str(role)] = grp.index.to_numpy()
    return out


def _sample_job_index(applicant_row, jobs: pd.DataFrame, job_index: dict[str, np.ndarray], rng: np.random.Generator) -> int:
    primary_role = str(applicant_row["primary_role_family"])
    if primary_role in job_index and rng.random() < 0.70:
        idxs = job_index[primary_role]
        return int(idxs[int(rng.integers(0, len(idxs)))])
    return int(rng.integers(0, len(jobs)))


def simulate_chunk(cfg: dict, seed: int, chunk_id: int) -> pd.DataFrame:
    paths = cfg["paths"]
    applicants = load_parquet(paths["raw"] / "applicants.parquet")
    companies = load_parquet(paths["raw"] / "companies.parquet")
    jobs = load_parquet(paths["raw"] / "jobs.parquet")
    sessions = load_parquet(paths["raw"] / "sessions.parquet")
    arm_profiles = load_json(paths["raw"] / "arm_profiles.json")

    app_map = applicants.set_index("applicant_id")
    comp_map = companies.set_index("company_id")
    session_rows = sessions.to_dict("records")
    job_rows = jobs.to_dict("records")
    job_index = _make_job_index(jobs)

    ablation = cfg["ablations"]["none"]
    rng = stage_rng(seed, cfg["seed_offsets"]["rounds"], extra=chunk_id)
    chunk_size = cfg["sizes"]["chunk_size"]
    rows = []
    start_round = chunk_id * chunk_size

    for local_idx in tqdm(range(chunk_size), desc=f"chunk {chunk_id}"):
        session = session_rows[int(rng.integers(0, len(session_rows)))]
        applicant = app_map.loc[str(session["applicant_id"])].to_dict()
        applicant["applicant_id"] = str(session["applicant_id"])
        job_idx = _sample_job_index(applicant, jobs, job_index, rng)
        job = job_rows[job_idx]
        company = comp_map.loc[str(job["company_id"])].to_dict()
        company["company_id"] = str(job["company_id"])
        components = compute_fit_components(applicant, job, company)
        fit_score = compute_fit_score(components)
        base_meta = {
            **components,
            "fit_score": fit_score,
            "session_goal": str(session["session_goal"]),
            "reading_patience": float(applicant["reading_patience"]),
            "decision_speed": float(applicant["decision_speed"]),
            "self_filter_strength": float(applicant["self_filter_strength"]),
            "resume_quality": float(applicant["resume_quality"]),
            "fatigue_before": float(session["initial_fatigue"]),
            "fatigue_sensitivity": float(applicant["fatigue_sensitivity"]),
            "job_complexity": float(job["complexity_score"]),
            "scatter_apply_tendency": float(applicant["scatter_apply_tendency"]),
            "fatigue_sensitivity": float(applicant["fatigue_sensitivity"]),
            "prestige_preference": float(applicant["prestige_preference"]),
            "company_prestige": float(company["prestige_tier"]),
            "resume_match_weight": float(company["resume_match_weight"]),
            "experience_weight": float(company["experience_weight"]),
            "education_weight": float(company["education_weight"]),
            "keyword_weight": float(company["keyword_weight"]),
            "clarity_bonus_weight": float(company["clarity_bonus_weight"]),
            "location_weight": float(company["location_weight"]),
            "salary_weight": float(company["salary_weight"]),
            "response_rate_base": float(company["response_rate_base"]),
            "ignore_threshold": float(company["ignore_threshold"]),
            "interview_threshold": float(company["interview_threshold"]),
            "screen_noise_std": float(company["screen_noise_std"]),
            "company_value_weight": float(company["company_value_weight"]),
            "applicant_uncertainty": float(
                np.clip(
                    0.35 * (1.0 - float(applicant["self_filter_strength"]))
                    + 0.20 * float(applicant["scatter_apply_tendency"])
                    + 0.20 * float(str(applicant["cohort"]) == "career_switcher")
                    + 0.15 * float(job["complexity_score"])
                    + 0.10 * float(session["initial_fatigue"]),
                    0.0,
                    1.0,
                )
            ),
            "ui_need": float(
                np.clip(
                    0.45 * float(job["complexity_score"])
                    + 0.25 * float(job["description_length"])
                    + 0.20 * (1.0 - float(applicant["self_filter_strength"]))
                    + 0.10 * (1.0 - float(applicant["decision_speed"])),
                    0.0,
                    1.0,
                )
            ),
        }
        oracle_arm, oracle_reward, expected_rewards, per_arm_meta = oracle_arm_and_reward(
            base_meta,
            arm_profiles,
            cfg["arm_names"],
            arm_effect_multiplier=ablation["arm_effect_multiplier"],
            fatigue_enabled=ablation["fatigue_enabled"],
        )
        arm_idx, propensity, _ = choose_logged_arm(expected_rewards, cfg["arm_names"], rng, cfg)
        chosen_arm_name = cfg["arm_names"][arm_idx]
        outcome, reward_realized, realized_meta = sample_outcome_and_reward(
            base_meta,
            arm_profiles[chosen_arm_name],
            rng=rng,
            arm_effect_multiplier=ablation["arm_effect_multiplier"],
            fatigue_enabled=ablation["fatigue_enabled"],
            screen_noise_multiplier=ablation["screen_noise_multiplier"],
        )
        context = build_context_vector(applicant, job, company, session, components)
        row = {
            "round_id": start_round + local_idx,
            "chunk_id": chunk_id,
            "session_id": session["session_id"],
            "applicant_id": applicant["applicant_id"],
            "job_id": job["job_id"],
            "company_id": company["company_id"],
            "applicant_cohort": applicant["cohort"],
            "company_tier": company["company_tier"],
            "session_goal": session["session_goal"],
            "job_role_family": job["role_family"],
            "arm_chosen": arm_idx,
            "arm_chosen_name": chosen_arm_name,
            "arm_propensity": propensity,
            **{k: float(v) for k, v in components.items()},
            "fit_score": fit_score,
            "applicant_uncertainty": base_meta["applicant_uncertainty"],
            "ui_need": base_meta["ui_need"],
            "submit_probability": float(realized_meta["submit_probability"]),
            "submitted": int(realized_meta["p_submit"] > 0.5),
            "info_coverage": float(realized_meta["info_coverage"]),
            "fit_comprehension": float(realized_meta["fit_comprehension"]),
            "dwell_time_seconds": float(realized_meta["dwell_time_seconds"]),
            "fatigue_before": float(base_meta["fatigue_before"]),
            "fatigue_increment": float(realized_meta["fatigue_increment"]),
            "fatigue_after": float(np.clip(base_meta["fatigue_before"] + realized_meta["fatigue_increment"], 0.0, 1.0)),
            "self_filter_quality": float(realized_meta["self_filter_quality"]),
            "ui_match_score": float(realized_meta["ui_match_score"]),
            "screen_score": float(realized_meta["screen_score"]),
            "p_respond": float(realized_meta["p_respond"]),
            "p_interview_given_respond": float(realized_meta["p_interview_given_respond"]),
            "outcome": outcome,
            "reward_realized": reward_realized,
            "oracle_arm": oracle_arm,
            "oracle_arm_name": cfg["arm_names"][oracle_arm],
            "oracle_expected_reward": oracle_reward,
            "chosen_expected_reward": float(expected_rewards[arm_idx]),
            "instant_expected_regret": float(oracle_reward - expected_rewards[arm_idx]),
            "reading_patience": float(applicant["reading_patience"]),
            "decision_speed": float(applicant["decision_speed"]),
            "self_filter_strength": float(applicant["self_filter_strength"]),
            "resume_quality": float(applicant["resume_quality"]),
            "scatter_apply_tendency": float(applicant["scatter_apply_tendency"]),
            "fatigue_sensitivity": float(applicant["fatigue_sensitivity"]),
            "prestige_preference": float(applicant["prestige_preference"]),
            "company_prestige": float(company["prestige_tier"]),
            "response_rate_base": float(company["response_rate_base"]),
            "ignore_threshold": float(company["ignore_threshold"]),
            "interview_threshold": float(company["interview_threshold"]),
            "company_value_weight": float(company["company_value_weight"]),
            "resume_match_weight": float(company["resume_match_weight"]),
            "experience_weight": float(company["experience_weight"]),
            "education_weight": float(company["education_weight"]),
            "keyword_weight": float(company["keyword_weight"]),
            "clarity_bonus_weight": float(company["clarity_bonus_weight"]),
            "location_weight": float(company["location_weight"]),
            "salary_weight": float(company["salary_weight"]),
            "screen_noise_std": float(company["screen_noise_std"]),
            "job_complexity": float(job["complexity_score"]),
        }
        for j, val in enumerate(context):
            row[f"context_f_{j:02d}"] = float(val)
        for arm_name, val in zip(cfg["arm_names"], expected_rewards):
            row[f"exp_reward_{arm_name}"] = float(val)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="debug")
    parser.add_argument("--seed", type=int, default=4014)
    parser.add_argument("--chunk-id", type=int, required=True)
    parser.add_argument("--root", type=str, default=None)
    args = parser.parse_args()

    t0 = time.time()
    cfg = get_config(args.preset, args.root)
    paths = cfg["paths"]
    ensure_dirs(paths["logged"], paths["logs"], paths["summaries"])
    df = simulate_chunk(cfg, args.seed, args.chunk_id)
    out_path = paths["logged"] / f"rounds_chunk_{args.chunk_id:03d}.parquet"
    save_parquet(df, out_path)
    append_log(paths["logs"] / "generation.log", f"simulate_rounds,preset={args.preset},chunk={args.chunk_id},rows={len(df)},seconds={time.time()-t0:.2f}")
    if args.chunk_id == 0:
        df.describe(include="all").transpose().to_csv(paths["summaries"] / "summary_rounds.csv")
    print(df.head())
    print(f"Saved chunk to {out_path}")


if __name__ == "__main__":
    main()
