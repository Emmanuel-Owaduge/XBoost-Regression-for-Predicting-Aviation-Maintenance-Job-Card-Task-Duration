"""Central configuration for the aviation-maintenance regression pipeline.

All tunable constants live here so that a change (e.g. re-running the
signal-strength sanity check with a different noise level) touches one file,
per spec/architecture.md section 4.
"""

from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "generated"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
METRICS_DIR = OUTPUTS_DIR / "metrics"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# --- Reproducibility ---
SEED = 42

# --- Dataset size / shape ---
N_RECORDS = 800
N_TECHNICIANS = 40
JOB_DATE_WINDOW_DAYS = 365

# --- Target generation (spec: two signal components + log-normal noise, clipped) ---
SIGMA_LOG = 0.225  # log-scale noise sigma, starting point per spec Risks section
OVERRUN_FLOOR = 0.4
OVERRUN_CAP = 3.5

# Modest additive main effect (not an interaction) from staffing_ratio on the
# target's log-mean: higher staffing relative to the 0.6-1.2 sampling range's
# midpoint modestly lowers overrun; lower staffing modestly raises it. Linear
# in log-space, so it's a real signal a linear model (fit on log-target, see
# models/linear.py) can recover, not a tree-only structural advantage.
STAFFING_RATIO_LOG_COEF = -0.15
STAFFING_RATIO_REFERENCE = 0.9  # midpoint of the staffing_ratio sampling range

# --- Split (spec: chronological 70/15/15 by job_date) ---
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train, calibration, test

# --- Model tuning ---
CV_FOLDS = 5
RF_PARAM_GRID = {
    "model__n_estimators": [100, 200, 400],
    "model__max_depth": [1, 2, 3, 5, 10],
    "model__min_samples_leaf": [1, 3, 5, 15, 40],
}
XGB_PARAM_GRID = {
    "model__n_estimators": [50, 100, 200, 400],
    "model__max_depth": [1, 2, 3, 5, 7],
    "model__learning_rate": [0.01, 0.03, 0.1, 0.3],
}

# --- Conformal prediction ---
CONFORMAL_ALPHA = 0.10  # 90% nominal coverage

# --- Sanity-check thresholds (spec/architecture.md section 6) ---
SANITY_CHECK_MIN_RATIO = 0.15
SANITY_CHECK_MAX_RATIO = 0.95

# --- Modeling feature columns ---
# record_id, job_date, and technician_id are deliberately excluded — see
# spec/architecture.md section 5 ("Job date and technician_id excluded from
# modeling features").
CATEGORICAL_FEATURES = ["task_type", "task_category", "shift_type", "day_of_week"]
NUMERIC_FEATURES = [
    "estimated_duration_hours",
    "rolling_efficiency_30d",
    "task_type_experience_count",
    "learning_curve_index",
    "staffing_ratio",
    "attendance_rate",
]
TARGET_COL = "overrun_factor"
DATE_COL = "job_date"
ID_COL = "record_id"
