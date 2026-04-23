from __future__ import annotations

import argparse

import pandas as pd

from config import get_config
from src.utils_io import ensure_dirs, load_parquet


def to_latex_if_possible(df: pd.DataFrame, path):
    try:
        df.to_latex(path, index=False, float_format="%.4f")
    except Exception:
        path.write_text(df.to_string(index=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="debug")
    parser.add_argument("--root", type=str, default=None)
    args = parser.parse_args()

    cfg = get_config(args.preset, args.root)
    paths = cfg["paths"]
    ensure_dirs(paths["tables"])
    algo_summary = pd.read_csv(paths["tables"] / "algo_summary.csv")
    world_summary = pd.read_csv(paths["summaries"] / "summary_rounds.csv") if (paths["summaries"] / "summary_rounds.csv").exists() else pd.DataFrame()

    algo_summary.to_csv(paths["tables"] / "algo_summary.csv", index=False)
    if not world_summary.empty:
        world_summary.to_csv(paths["tables"] / "world_summary.csv")

    to_latex_if_possible(algo_summary, paths["tables"] / "algo_summary.tex")
    if not world_summary.empty:
        to_latex_if_possible(world_summary.reset_index(), paths["tables"] / "world_summary.tex")
    print("Exported report tables.")


if __name__ == "__main__":
    main()
