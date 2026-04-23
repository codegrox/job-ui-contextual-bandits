from __future__ import annotations

import argparse
import time

import pandas as pd

from config import get_config
from src.utils_io import append_log, ensure_dirs, load_parquet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="debug")
    parser.add_argument("--seed", type=int, default=4014)
    parser.add_argument("--root", type=str, default=None)
    args = parser.parse_args()

    t0 = time.time()
    cfg = get_config(args.preset, args.root)
    paths = cfg["paths"]
    ensure_dirs(paths["summaries"], paths["logs"])
    df = load_parquet(paths["processed"] / "rounds_all_meta.parquet")

    fit_diag = df.assign(fit_decile=pd.qcut(df["fit_score"], q=10, duplicates="drop")) \
        .groupby("fit_decile", as_index=False) \
        .agg(interview_rate=("outcome", lambda s: (s == "interview_1").mean()),
             mean_reward=("reward_realized", "mean"))

    arm_context = df.groupby(["session_goal", "oracle_arm_name"], as_index=False).size().rename(columns={"size": "count"})
    arm_context["share_within_goal"] = arm_context.groupby("session_goal")["count"].transform(lambda x: x / x.sum())

    outcomes = df.groupby(["arm_chosen_name", "outcome"], as_index=False).size().rename(columns={"size": "count"})
    outcomes["share_within_arm"] = outcomes.groupby("arm_chosen_name")["count"].transform(lambda x: x / x.sum())

    oracle_gap = df[["round_id", "instant_expected_regret", "oracle_expected_reward", "chosen_expected_reward"]].copy()

    fit_diag.to_csv(paths["summaries"] / "diagnostic_fit_vs_interview.csv", index=False)
    arm_context.to_csv(paths["summaries"] / "diagnostic_arm_by_context.csv", index=False)
    outcomes.to_csv(paths["summaries"] / "diagnostic_outcomes.csv", index=False)
    oracle_gap.to_csv(paths["summaries"] / "diagnostic_oracle_gap.csv", index=False)
    append_log(paths["logs"] / "diagnostics.log", f"diagnostics,preset={args.preset},rows={len(df)},seconds={time.time()-t0:.2f}")
    print(fit_diag.head())
    print(arm_context.head())


if __name__ == "__main__":
    main()
