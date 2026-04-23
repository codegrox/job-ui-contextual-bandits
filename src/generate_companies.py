from __future__ import annotations

import argparse
import time

import pandas as pd

from config import get_config
from src.ontology import INDUSTRIES, COMPANY_TIERS, COMPANY_TIER_TO_IDX
from src.utils_io import append_log, ensure_dirs, save_parquet
from src.utils_random import stage_rng
from src.utils_math import clip01, normalized_index


def generate_companies_df(cfg: dict, seed: int) -> pd.DataFrame:
    rng = stage_rng(seed, cfg["seed_offsets"]["companies"])
    n = cfg["sizes"]["n_companies"]
    tiers = list(cfg["company_tier_probs"].keys())
    tier_probs = list(cfg["company_tier_probs"].values())
    industries = INDUSTRIES

    rows = []
    for i in range(n):
        tier = rng.choice(tiers, p=tier_probs)
        industry = rng.choice(industries)
        tier_idx = COMPANY_TIER_TO_IDX[tier]
        prestige_tier = normalized_index(tier_idx, len(COMPANY_TIERS))
        difficulty_tier = clip01(0.15 + 0.18 * tier_idx + rng.normal(0, 0.05))
        response_rate_base = clip01(0.85 - 0.12 * tier_idx + rng.normal(0, 0.04))
        company_value_weight = float(0.80 + 0.16 * tier_idx + rng.normal(0, 0.04))
        company_value_weight = min(max(company_value_weight, 0.80), 1.50)
        ignore_threshold = clip01(0.25 + 0.05 * tier_idx + rng.normal(0, 0.03))
        interview_threshold = clip01(0.58 + 0.07 * tier_idx + rng.normal(0, 0.04))
        base = rng.dirichlet([3.0, 2.0, 1.2, 1.5, 1.2, 1.0, 1.0])
        rows.append({
            "company_id": f"company_{i:06d}",
            "industry": industry,
            "company_tier": tier,
            "company_size": int(rng.integers(20, 5000)),
            "prestige_tier": prestige_tier,
            "difficulty_tier": difficulty_tier,
            "response_rate_base": response_rate_base,
            "company_value_weight": company_value_weight,
            "ignore_threshold": ignore_threshold,
            "interview_threshold": max(interview_threshold, ignore_threshold + 0.05),
            "resume_match_weight": float(base[0]),
            "experience_weight": float(base[1]),
            "education_weight": float(base[2]),
            "keyword_weight": float(base[3]),
            "clarity_bonus_weight": float(base[4]),
            "location_weight": float(base[5]),
            "salary_weight": float(base[6]),
            "salary_flexibility": float(clip01(rng.beta(2, 3))),
            "remote_friendliness": float(clip01(rng.beta(2 + tier_idx * 0.3, 2))),
            "screen_noise_std": cfg["screen_noise_by_tier"][tier],
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
    df = generate_companies_df(cfg, args.seed)
    save_parquet(df, paths["raw"] / "companies.parquet")
    df.describe(include="all").transpose().to_csv(paths["summaries"] / "summary_companies.csv")
    append_log(paths["logs"] / "generation.log", f"generate_companies,preset={args.preset},rows={len(df)},seconds={time.time()-t0:.2f}")
    print(df.head())
    print(f"Saved {len(df)} companies")


if __name__ == "__main__":
    main()
