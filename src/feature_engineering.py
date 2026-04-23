from __future__ import annotations

import numpy as np

from config import SKILL_COLUMNS, CONTEXT_DIM
from src.ontology import ROLE_TO_IDX, EDUCATION_TO_IDX, COMPANY_TIER_TO_IDX, DEVICE_TO_IDX, SESSION_GOAL_TO_IDX, REMOTE_TO_IDX, ROLE_FAMILIES
from src.utils_math import clip01, cosine_similarity, normalized_index


def _skill_vec_from_applicant(row) -> np.ndarray:
    return np.array([row[f"skill_{k}"] for k in SKILL_COLUMNS], dtype=float)


def _skill_req_vec_from_job(row) -> np.ndarray:
    return np.array([row[f"skillreq_{k}"] for k in SKILL_COLUMNS], dtype=float)


def _keyword_vec_from_job(row) -> np.ndarray:
    return np.array([row[f"keyword_strength_{k}"] for k in SKILL_COLUMNS], dtype=float)


def compute_skill_match(applicant_row, job_row) -> float:
    return clip01(cosine_similarity(_skill_vec_from_applicant(applicant_row), _skill_req_vec_from_job(job_row)))


def compute_keyword_match(applicant_row, job_row) -> float:
    app = _skill_vec_from_applicant(applicant_row) * float(applicant_row["keyword_resume_strength"])
    job = _keyword_vec_from_job(job_row)
    return clip01(cosine_similarity(app, job))


def compute_experience_match(applicant_row, job_row) -> float:
    req = float(job_row["required_experience"])
    have = float(applicant_row["relevant_experience"])
    if req <= 1e-8:
        return 1.0
    return clip01(1.0 - max(req - have, 0.0) / max(req, 1.0))


def compute_education_match(applicant_row, job_row) -> float:
    have = EDUCATION_TO_IDX[str(applicant_row["education_level"])]
    req = EDUCATION_TO_IDX[str(job_row["required_education"])]
    if have >= req:
        return 1.0
    return clip01(1.0 - 0.35 * (req - have))


def compute_salary_alignment(applicant_row, job_row) -> float:
    pref = 0.20 + 0.70 * (1.0 - float(applicant_row["salary_sensitivity"]))
    return clip01(1.0 - abs(float(job_row["salary_band_norm"]) - pref))


def compute_location_alignment(applicant_row, job_row, company_row) -> float:
    remote_pref = float(applicant_row["remote_preference"])
    remote_mode = str(job_row["remote_mode"])
    remote_score = {"onsite": 0.2, "hybrid": 0.6, "remote": 1.0}[remote_mode]
    friendliness = float(company_row["remote_friendliness"])
    return clip01(1.0 - abs(remote_pref - (0.7 * remote_score + 0.3 * friendliness)))


def compute_role_family_alignment(applicant_row, job_row) -> float:
    role = str(job_row["role_family"])
    if role == str(applicant_row["primary_role_family"]):
        return 1.0
    if role == str(applicant_row["secondary_role_family"]):
        return 0.65
    return 0.20


def compute_prestige_alignment(applicant_row, company_row) -> float:
    return clip01(1.0 - abs(float(applicant_row["prestige_preference"]) - float(company_row["prestige_tier"])))


def compute_fit_components(applicant_row, job_row, company_row) -> dict[str, float]:
    return {
        "skill_match": compute_skill_match(applicant_row, job_row),
        "keyword_match": compute_keyword_match(applicant_row, job_row),
        "experience_match": compute_experience_match(applicant_row, job_row),
        "education_match": compute_education_match(applicant_row, job_row),
        "salary_alignment": compute_salary_alignment(applicant_row, job_row),
        "location_alignment": compute_location_alignment(applicant_row, job_row, company_row),
        "role_family_alignment": compute_role_family_alignment(applicant_row, job_row),
        "prestige_alignment": compute_prestige_alignment(applicant_row, company_row),
    }


def build_context_vector(applicant_row, job_row, company_row, session_row, components: dict[str, float]) -> np.ndarray:
    # applicant block: 12
    applicant = np.array([
        float(applicant_row["years_experience"]) / 20.0,
        normalized_index(EDUCATION_TO_IDX[str(applicant_row["education_level"])], len(EDUCATION_TO_IDX)),
        float(applicant_row["resume_quality"]),
        float(applicant_row["keyword_resume_strength"]),
        float(applicant_row["reading_patience"]),
        float(applicant_row["decision_speed"]),
        float(applicant_row["fatigue_sensitivity"]),
        float(applicant_row["scatter_apply_tendency"]),
        float(applicant_row["self_filter_strength"]),
        float(applicant_row["prestige_preference"]),
        float(applicant_row["salary_sensitivity"]),
        float(applicant_row["remote_preference"]),
    ], dtype=float)

    # job block: 10
    job = np.array([
        normalized_index(ROLE_TO_IDX[str(job_row["role_family"])], len(ROLE_FAMILIES)),
        float(job_row["complexity_score"]),
        float(job_row["description_length"]),
        float(job_row["required_experience"]) / 15.0,
        normalized_index(EDUCATION_TO_IDX[str(job_row["required_education"])], len(EDUCATION_TO_IDX)),
        float(job_row["salary_band_norm"]),
        normalized_index(REMOTE_TO_IDX[str(job_row["remote_mode"])], len(REMOTE_TO_IDX)),
        float([0.0, 0.25, 0.5, 0.75, 1.0][["intern", "junior", "mid", "senior", "staff_plus"].index(str(job_row["seniority_level"]))]),
        float(components["keyword_match"]),
        float(components["skill_match"]),
    ], dtype=float)

    # company block: 8
    company = np.array([
        normalized_index(COMPANY_TIER_TO_IDX[str(company_row["company_tier"])], len(COMPANY_TIER_TO_IDX)),
        float(company_row["prestige_tier"]),
        float(company_row["difficulty_tier"]),
        float(company_row["response_rate_base"]),
        float(company_row["company_value_weight"] - 0.80) / 0.70,
        float(company_row["interview_threshold"]),
        float(company_row["keyword_weight"]),
        float(company_row["clarity_bonus_weight"]),
    ], dtype=float)

    # session block: 8
    session = np.array([
        float(session_row["time_budget_minutes"]) / 90.0,
        float(session_row["initial_fatigue"]),
        min(float(session_row["applications_last_7d"]) / 20.0, 1.0),
        min(float(session_row["rejections_last_30d"]) / 30.0, 1.0),
        min(float(session_row["ignores_last_30d"]) / 50.0, 1.0),
        normalized_index(SESSION_GOAL_TO_IDX[str(session_row["session_goal"])], len(SESSION_GOAL_TO_IDX)),
        normalized_index(DEVICE_TO_IDX[str(session_row["device_type"])], len(DEVICE_TO_IDX)),
        float(session_row["start_hour_bucket"]) / 23.0,
    ], dtype=float)

    # cross features: 10
    ui_need = clip01(0.45 * float(job_row["complexity_score"]) + 0.25 * float(job_row["description_length"]) + 0.20 * (1.0 - float(applicant_row["self_filter_strength"])) + 0.10 * (1.0 - float(applicant_row["decision_speed"])))
    applicant_uncertainty = clip01(0.35 * (1.0 - float(applicant_row["self_filter_strength"])) + 0.20 * float(applicant_row["scatter_apply_tendency"]) + 0.20 * float(str(applicant_row["cohort"]) == "career_switcher") + 0.15 * float(job_row["complexity_score"]) + 0.10 * float(session_row["initial_fatigue"]))
    cross = np.array([
        float(components["skill_match"]),
        float(components["keyword_match"]),
        float(components["experience_match"]),
        float(components["education_match"]),
        float(components["salary_alignment"]),
        float(components["location_alignment"]),
        float(components["role_family_alignment"]),
        float(components["prestige_alignment"]),
        ui_need,
        applicant_uncertainty,
    ], dtype=float)

    context = np.concatenate([applicant, job, company, session, cross])
    if context.shape[0] != CONTEXT_DIM:
        raise ValueError(f"Context dimension mismatch: {context.shape[0]} != {CONTEXT_DIM}")
    return context
