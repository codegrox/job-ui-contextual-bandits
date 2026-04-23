from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from config import get_config
from src.utils_io import append_log, ensure_dirs, load_parquet, save_parquet
from src.utils_random import stage_rng


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="debug")
    parser.add_argument("--seed", type=int, default=4014)
    parser.add_argument("--root", type=str, default=None)
    args = parser.parse_args()

    t0 = time.time()
    cfg = get_config(args.preset, args.root)
    paths = cfg["paths"]
    ensure_dirs(paths["processed"], paths["logs"])
    frames = []
    for chunk_id in range(cfg["sizes"]["n_chunks"]):
        path = paths["logged"] / f"rounds_chunk_{chunk_id:03d}.parquet"
        if path.exists():
            frames.append(load_parquet(path))
    if not frames:
        raise FileNotFoundError("No chunk files found.")
    df = pd.concat(frames, ignore_index=True)
    save_parquet(df, paths["processed"] / "rounds_all_meta.parquet")

    rng = stage_rng(args.seed, cfg["seed_offsets"]["rounds"], extra=999)
    perm = rng.permutation(len(df))
    train_end = int(cfg["data_split"]["train_frac"] * len(df))
    valid_end = int((cfg["data_split"]["train_frac"] + cfg["data_split"]["valid_frac"]) * len(df))
    train_idx = perm[:train_end]
    valid_idx = perm[train_end:valid_end]
    test_idx = perm[valid_end:]
    save_parquet(df.iloc[train_idx].reset_index(drop=True), paths["processed"] / "train_rounds.parquet")
    save_parquet(df.iloc[valid_idx].reset_index(drop=True), paths["processed"] / "valid_rounds.parquet")
    save_parquet(df.iloc[test_idx].reset_index(drop=True), paths["processed"] / "test_rounds.parquet")

    oracle_summary = df.groupby("oracle_arm_name", as_index=False).agg(
        oracle_best_share=("round_id", "count"),
        mean_oracle_reward=("oracle_expected_reward", "mean"),
    )
    oracle_summary["oracle_best_share"] = oracle_summary["oracle_best_share"] / len(df)
    save_parquet(oracle_summary, paths["processed"] / "oracle_summary.parquet")
    append_log(paths["logs"] / "generation.log", f"merge_and_split,preset={args.preset},rows={len(df)},seconds={time.time()-t0:.2f}")
    print(oracle_summary)


if __name__ == "__main__":
    main()
