from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from config import get_config
from src.ontology import ROLE_FAMILIES, EDUCATION_LEVELS, EDUCATION_TO_IDX, ROLE_TO_IDX
from src.utils_io import append_log, ensure_dirs, save_parquet
from src.utils_random import stage_rng
from src.utils_math import clip01, normalized_index


ROLE_SKILL_PROTOTYPES = {
    "software_engineering": [0.9, 0.55, 0.2, 0.2, 0.9, 0.6, 0.6, 0.6, 0.2, 0.4, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.3, 0.5, 0.3, 0.2, 0.4, 0.3],
    "data_science": [0.9, 0.8, 0.9, 0.9, 0.4, 0.1, 0.2, 0.4, 0.3, 0.8, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 0.2, 0.8, 0.5, 0.4, 0.1, 0.4, 0.5],
    "product_management": [0.3, 0.4, 0.2, 0.2, 0.2, 0.1, 0.1, 0.2, 0.9, 0.7, 0.8, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.7, 0.9, 0.8, 0.2, 0.6, 0.5],
    "design": [0.1, 0.1, 0.1, 0.1, 0.2, 0.4, 0.1, 0.1, 0.4, 0.3, 0.6, 0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4, 0.6, 0.8, 0.1, 0.3, 0.4],
    "marketing": [0.1, 0.3, 0.1, 0.2, 0.1, 0.2, 0.1, 0.1, 0.5, 0.8, 0.8, 0.3, 0.3, 0.4, 0.3, 0.2, 0.1, 0.2, 0.7, 0.7, 0.9, 0.2, 0.4, 0.5],
    "sales": [0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4, 0.4, 0.9, 0.1, 0.1, 0.8, 0.9, 0.1, 0.1, 0.2, 0.2, 0.8, 0.7, 0.4, 0.5, 0.4],
    "operations": [0.2, 0.4, 0.1, 0.2, 0.1, 0.1, 0.2, 0.2, 0.4, 0.6, 0.6, 0.1, 0.1, 0.2, 0.2, 0.5, 0.4, 0.8, 0.5, 0.6, 0.4, 0.5, 0.5, 0.5],
    "finance": [0.2, 0.8, 0.2, 0.7, 0.1, 0.1, 0.1, 0.2, 0.3, 0.8, 0.5, 0.1, 0.1, 0.1, 0.2, 0.9, 0.7, 0.3, 0.5, 0.5, 0.4, 0.1, 0.5, 0.6],
    "hr": [0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 0.4, 0.9, 0.1, 0.3, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.7, 0.8, 0.5, 0.4, 0.4],
    "customer_success": [0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4, 0.5, 0.9, 0.1, 0.2, 0.4, 0.4, 0.1, 0.1, 0.3, 0.4, 0.7, 0.8, 0.9, 0.4, 0.4],
}

COHORT_EXPERIENCE = {
    "new_grad": (0.5, 0.4),
    "junior_generalist": (2.0, 0.8),
    "mid_specialist": (4.5, 1.0),
    "senior_specialist": (8.5, 1.5),
    "career_switcher": (5.0, 2.0),
    "burned_out_mass_applier": (3.5, 1.8),
}

EDU_ROLE_BOOST = {
    "data_science": [0.05, 0.35, 0.45, 0.15],
    "finance": [0.05, 0.45, 0.35, 0.15],
    "software_engineering": [0.10, 0.60, 0.25, 0.05],
}


def choose_education(role: str, rng: np.random.Generator) -> str:
    probs = EDU_ROLE_BOOST.get(role, [0.10, 0.65, 0.20, 0.05])
    return rng.choice(EDUCATION_LEVELS, p=probs)


def generate_applicants_df(cfg: dict, seed: int) -> pd.DataFrame:
    rng = stage_rng(seed, cfg["seed_offsets"]["applicants"])
    n = cfg["sizes"]["n_applicants"]
    cohorts = list(cfg["applicant_cohort_probs"].keys())
    cohort_probs = list(cfg["applicant_cohort_probs"].values())
    skill_cols = cfg["skill_columns"]
    rows = []
    for i in range(n):
        cohort = rng.choice(cohorts, p=cohort_probs)
        primary_role = rng.choice(ROLE_FAMILIES)
        secondary_candidates = [r for r in ROLE_FAMILIES if r != primary_role]
        secondary_role = rng.choice(secondary_candidates)
        mean_exp, std_exp = COHORT_EXPERIENCE[cohort]
        years_exp = float(np.clip(rng.normal(mean_exp, std_exp), 0.0, 20.0))
        relevant_exp = float(np.clip(years_exp * rng.uniform(0.5, 1.0), 0.0, 20.0))
        education = choose_education(primary_role, rng)
        education_norm = normalized_index(EDUCATION_TO_IDX[education], len(EDUCATION_LEVELS))
        proto = np.array(ROLE_SKILL_PROTOTYPES[primary_role], dtype=float)
        specialization = 0.03 * years_exp + 0.10 * education_norm
        skills = np.clip(proto + specialization + rng.normal(0, 0.10, size=len(skill_cols)), 0.0, 1.0)
        scatter = clip01(rng.beta(2, 4) + (0.20 if cohort == "burned_out_mass_applier" else 0.0))
        self_filter = clip01(rng.beta(3, 2) - (0.25 if cohort == "burned_out_mass_applier" else 0.0) - (0.10 if cohort == "career_switcher" else 0.0))
        rows.append({
            "applicant_id": f"person_{i:07d}",
            "cohort": cohort,
            "years_experience": years_exp,
            "relevant_experience": relevant_exp,
            "education_level": education,
            "primary_role_family": primary_role,
            "secondary_role_family": secondary_role,
            "resume_quality": float(clip01(rng.beta(3, 2))),
            "keyword_resume_strength": float(clip01(rng.beta(3, 2))),
            "reading_patience": float(clip01(rng.beta(2, 2))),
            "decision_speed": float(clip01(rng.beta(2, 2))),
            "fatigue_sensitivity": float(clip01(rng.beta(2, 3))),
            "scatter_apply_tendency": float(scatter),
            "self_filter_strength": float(self_filter),
            "prestige_preference": float(clip01(rng.beta(2, 2))),
            "salary_sensitivity": float(clip01(rng.beta(2, 2))),
            "remote_preference": float(clip01(rng.beta(2, 2))),
            **{f"skill_{k}": float(v) for k, v in zip(skill_cols, skills)},
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
    df = generate_applicants_df(cfg, args.seed)
    save_parquet(df, paths["raw"] / "applicants.parquet")
    df.describe(include="all").transpose().to_csv(paths["summaries"] / "summary_applicants.csv")
    append_log(paths["logs"] / "generation.log", f"generate_applicants,preset={args.preset},rows={len(df)},seconds={time.time()-t0:.2f}")
    print(df.head())
    print(f"Saved {len(df)} applicants")


if __name__ == "__main__":
    main()
