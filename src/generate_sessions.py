from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from config import get_config
from src.ontology import DEVICE_TYPES, SESSION_GOALS
from src.utils_io import append_log, ensure_dirs, load_parquet, save_parquet
from src.utils_random import stage_rng
from src.utils_math import clip01


def generate_sessions_df(cfg: dict, seed: int, applicants: pd.DataFrame) -> pd.DataFrame:
    rng = stage_rng(seed, cfg["seed_offsets"]["sessions"])
    n = cfg["sizes"]["n_sessions"]
    goal_names = list(cfg["session_goal_probs"].keys())
    goal_probs = list(cfg["session_goal_probs"].values())
    applicant_ids = applicants["applicant_id"].to_numpy()
    rows = []
    for i in range(n):
        applicant_id = str(applicant_ids[int(rng.integers(0, len(applicant_ids)))])
        rows.append({
            "session_id": f"session_{i:07d}",
            "applicant_id": applicant_id,
            "session_goal": rng.choice(goal_names, p=goal_probs),
            "device_type": rng.choice(DEVICE_TYPES, p=[0.55, 0.35, 0.10]),
            "start_hour_bucket": int(rng.integers(0, 24)),
            "time_budget_minutes": int(rng.integers(10, 90)),
            "initial_fatigue": float(clip01(rng.beta(2, 4))),
            "applications_last_7d": int(rng.poisson(6)),
            "rejections_last_30d": int(rng.poisson(10)),
            "ignores_last_30d": int(rng.poisson(15)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="debug")
    parser.add_argument("--seed", type=int, default=4014)
    parser.add_argument("--root", type=str, default=None)
    args = parser.parse_args()

    t0 = time.time()
    cfg = get_config(args.preset, args.root)
    paths = cfg["paths"]
    ensure_dirs(paths["raw"], paths["summaries"], paths["logs"])
    applicants = load_parquet(paths["raw"] / "applicants.parquet")
    df = generate_sessions_df(cfg, args.seed, applicants)
    save_parquet(df, paths["raw"] / "sessions.parquet")
    df.describe(include="all").transpose().to_csv(paths["summaries"] / "summary_sessions.csv")
    append_log(paths["logs"] / "generation.log", f"generate_sessions,preset={args.preset},rows={len(df)},seconds={time.time()-t0:.2f}")
    print(df.head())
    print(f"Saved {len(df)} sessions")


if __name__ == "__main__":
    main()
