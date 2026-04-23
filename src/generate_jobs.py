from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from config import get_config
from src.generate_applicants import ROLE_SKILL_PROTOTYPES
from src.ontology import ROLE_FAMILIES, REMOTE_MODES, SENIORITY_LEVELS, SENIORITY_TO_IDX, EDUCATION_LEVELS, EDUCATION_TO_IDX
from src.utils_io import append_log, ensure_dirs, load_parquet, save_parquet
from src.utils_random import stage_rng
from src.utils_math import clip01, normalized_index


ROLE_BY_INDUSTRY = {
    "consumer_tech": ["software_engineering", "product_management", "design", "marketing", "data_science"],
    "enterprise_software": ["software_engineering", "product_management", "sales", "customer_success", "data_science"],
    "finance": ["finance", "operations", "data_science", "software_engineering"],
    "healthcare": ["operations", "data_science", "customer_success", "software_engineering"],
    "education": ["software_engineering", "product_management", "customer_success", "marketing"],
    "retail": ["operations", "marketing", "sales", "finance", "design"],
    "logistics": ["operations", "software_engineering", "finance", "customer_success"],
    "media": ["design", "marketing", "product_management", "sales", "customer_success"],
}


def generate_jobs_df(cfg: dict, seed: int, companies: pd.DataFrame) -> pd.DataFrame:
    rng = stage_rng(seed, cfg["seed_offsets"]["jobs"])
    n_jobs = cfg["sizes"]["n_jobs"]
    skill_cols = cfg["skill_columns"]
    company_rows = companies.to_dict("records")
    rows = []
    for i in range(n_jobs):
        company = company_rows[int(rng.integers(0, len(company_rows)))]
        role_family = rng.choice(ROLE_BY_INDUSTRY.get(company["industry"], ROLE_FAMILIES))
        seniority = rng.choice(SENIORITY_LEVELS, p=[0.05, 0.25, 0.35, 0.25, 0.10])
        seniority_norm = normalized_index(SENIORITY_TO_IDX[seniority], len(SENIORITY_LEVELS))
        required_exp = float(np.clip(rng.normal(1 + 10 * seniority_norm, 1.5), 0.0, 15.0))
        required_edu = rng.choice(EDUCATION_LEVELS, p=[0.1, 0.6, 0.25, 0.05])
        required_edu_norm = normalized_index(EDUCATION_TO_IDX[required_edu], len(EDUCATION_LEVELS))
        proto = np.array(ROLE_SKILL_PROTOTYPES[role_family], dtype=float)
        required = np.clip(proto + 0.15 * seniority_norm + rng.normal(0, 0.08, size=len(skill_cols)), 0.0, 1.0)
        keywords = np.clip(required + rng.normal(0, 0.05, size=len(skill_cols)), 0.0, 1.0)
        rows.append({
            "job_id": f"job_{i:08d}",
            "company_id": company["company_id"],
            "industry": company["industry"],
            "role_family": role_family,
            "seniority_level": seniority,
            "required_experience": required_exp,
            "required_education": required_edu,
            "salary_band_norm": float(clip01(0.25 + 0.60 * seniority_norm + rng.normal(0, 0.08))),
            "remote_mode": rng.choice(REMOTE_MODES, p=[0.25, 0.45, 0.30]),
            "description_length": float(clip01(0.35 + 0.35 * company["difficulty_tier"] + rng.normal(0, 0.08))),
            "complexity_score": float(clip01(0.25 + 0.55 * seniority_norm + 0.15 * company["difficulty_tier"] + rng.normal(0, 0.06))),
            **{f"skillreq_{k}": float(v) for k, v in zip(skill_cols, required)},
            **{f"keyword_strength_{k}": float(v) for k, v in zip(skill_cols, keywords)},
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
    companies = load_parquet(paths["raw"] / "companies.parquet")
    df = generate_jobs_df(cfg, args.seed, companies)
    save_parquet(df, paths["raw"] / "jobs.parquet")
    df.describe(include="all").transpose().to_csv(paths["summaries"] / "summary_jobs.csv")
    append_log(paths["logs"] / "generation.log", f"generate_jobs,preset={args.preset},rows={len(df)},seconds={time.time()-t0:.2f}")
    print(df.head())
    print(f"Saved {len(df)} jobs")


if __name__ == "__main__":
    main()
