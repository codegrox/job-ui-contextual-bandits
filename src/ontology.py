ROLE_FAMILIES = [
    "software_engineering",
    "data_science",
    "product_management",
    "design",
    "marketing",
    "sales",
    "operations",
    "finance",
    "hr",
    "customer_success",
]

INDUSTRIES = [
    "consumer_tech",
    "enterprise_software",
    "finance",
    "healthcare",
    "education",
    "retail",
    "logistics",
    "media",
]

COMPANY_TIERS = [
    "small_local",
    "startup",
    "mid_market",
    "large_public",
    "elite_platform",
]

SESSION_GOALS = [
    "focused_search",
    "broad_exploration",
    "panic_apply",
    "casual_browse",
]

EDUCATION_LEVELS = ["diploma", "bachelors", "masters", "phd"]
SENIORITY_LEVELS = ["intern", "junior", "mid", "senior", "staff_plus"]
DEVICE_TYPES = ["desktop", "mobile", "tablet"]
REMOTE_MODES = ["onsite", "hybrid", "remote"]

ROLE_TO_IDX = {v: i for i, v in enumerate(ROLE_FAMILIES)}
INDUSTRY_TO_IDX = {v: i for i, v in enumerate(INDUSTRIES)}
COMPANY_TIER_TO_IDX = {v: i for i, v in enumerate(COMPANY_TIERS)}
SESSION_GOAL_TO_IDX = {v: i for i, v in enumerate(SESSION_GOALS)}
EDUCATION_TO_IDX = {v: i for i, v in enumerate(EDUCATION_LEVELS)}
SENIORITY_TO_IDX = {v: i for i, v in enumerate(SENIORITY_LEVELS)}
DEVICE_TO_IDX = {v: i for i, v in enumerate(DEVICE_TYPES)}
REMOTE_TO_IDX = {v: i for i, v in enumerate(REMOTE_MODES)}
