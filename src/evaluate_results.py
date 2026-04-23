from __future__ import annotations

import argparse

import pandas as pd

from config import get_config
from src.utils_io import ensure_dirs, load_parquet


def compute_summary_table(results_df: pd.DataFrame) -> pd.DataFrame:
    agg = results_df.groupby(["algo_name", "mode"], as_index=False).agg(
        final_cumulative_reward=("cumulative_reward", "max"),
        final_cumulative_expected_regret=("cumulative_expected_regret", "max"),
        mean_reward=("reward_realized", "mean"),
        interview_rate=("outcome", lambda s: (s == "interview_1").mean()),
        ignore_rate=("outcome", lambda s: (s == "ignored").mean()),
    )
    return agg.sort_values(["mode", "final_cumulative_expected_regret", "final_cumulative_reward"], ascending=[True, True, False])


def compute_arm_selection_frequencies(results_df: pd.DataFrame) -> pd.DataFrame:
    out = results_df.groupby(["algo_name", "arm_selected_name"], as_index=False).size().rename(columns={"size": "count"})
    out["share"] = out.groupby("algo_name")["count"].transform(lambda x: x / x.sum())
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="debug")
    parser.add_argument("--mode", default="online")
    parser.add_argument("--ablation", default="none")
    parser.add_argument("--root", type=str, default=None)
    args = parser.parse_args()

    cfg = get_config(args.preset, args.root)
    paths = cfg["paths"]
    ensure_dirs(paths["tables"])
    results = load_parquet(paths["tables"] / f"experiment_results_{args.preset}_{args.mode}_{args.ablation}.parquet")
    summary = compute_summary_table(results)
    arm_freq = compute_arm_selection_frequencies(results)
    summary.to_csv(paths["tables"] / "algo_summary.csv", index=False)
    arm_freq.to_csv(paths["tables"] / "arm_selection_frequency.csv", index=False)
    print(summary)


if __name__ == "__main__":
    main()
