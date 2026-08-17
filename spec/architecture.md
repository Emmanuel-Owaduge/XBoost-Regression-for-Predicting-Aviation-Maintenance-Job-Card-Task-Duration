# Architecture: XGBoost Aviation Maintenance Regression Pipeline

Status: DRAFT — pending student approval before the `developer` skill begins
implementation. Based on `spec/project-specification.md` (approved, including
the chronological-split revision; the target-interaction revision was
proposed and then reversed — no *interaction* term was added). The target
formula was later amended to add a third factor beyond the spec's original
two signal components: a modest additive staffing_ratio main effect on the
log-mean (a plain main effect, not an interaction — see §4's
`data/generate.py` section for the exact expression). This amendment is
reflected in both documents.

## 1. Repo Layout

```
.
├── spec/                        (existing — specification + this document)
├── journal/                     (existing — reflection skill)
├── report/                      (new — prose deliverables, not code)
│   ├── literature_review.md
│   └── report.md                 methodology/results/discussion, assembled last
├── src/
│   ├── config.py                  single source of truth for all tunable constants
│   ├── data/
│   │   ├── generate.py            synthetic dataset generator (incl. job_date,
│   │   │                          causal rolling features)
│   │   └── split.py               chronological 70/15/15 split
│   ├── features/
│   │   └── preprocessing.py       shared ColumnTransformer builder
│   ├── models/
│   │   ├── common.py              ModelResult dataclass, shared fit-and-time helper
│   │   ├── linear.py
│   │   ├── random_forest.py
│   │   └── xgboost_model.py
│   ├── conformal/
│   │   └── conformal.py           split conformal calibration + interval prediction
│   ├── diagnostics/
│   │   └── sanity_check.py        signal-strength gate (spec task 4)
│   ├── evaluation/
│   │   ├── metrics.py             RMSE / MAE / MAPE / coverage
│   │   ├── compare.py             comparison table builder
│   │   └── shap_analysis.py       SHAP global + individual explanations
│   └── pipeline.py                single entry point, runs stages in spec order
├── data/generated/                train.csv / calibration.csv / test.csv (reproducible artifacts)
├── outputs/
│   ├── models/                    joblib-serialized fitted pipelines
│   ├── metrics/                   comparison table, coverage report (csv/json)
│   └── figures/                   SHAP plots
├── tests/                         for the `test` skill
└── requirements.txt
```

This mirrors the spec's task order directly: `data/` → `features/` → `models/`
→ `conformal/` → `evaluation/`, with `pipeline.py` as the only place that
knows the full sequence.

## 2. Components and Responsibilities

| Component | Responsibility | Spec task(s) |
|---|---|---|
| `config.py` | All constants: seed, record count, sigma, floor/cap, split ratios, CV folds, conformal alpha, param grids, sanity-check thresholds, job-date window | supports 2–11 |
| `data/generate.py` | Synthesize the raw DataFrame, assigning job dates and computing rolling/experience features causally | 2 |
| `data/split.py` | Chronological 70/15/15 split by job date | 3 |
| `features/preprocessing.py` | Build the shared encoder used by all three models | supports 5–7 |
| `diagnostics/sanity_check.py` | Gate: is the injected noise well-calibrated? | 4 |
| `models/linear.py`, `random_forest.py`, `xgboost_model.py` | Build + tune each model behind one common interface | 5, 6, 7 |
| `conformal/conformal.py` | Calibrate and apply 90% prediction intervals on XGBoost | 8 |
| `evaluation/metrics.py` | Point-estimate error metrics + interval coverage | 9 |
| `evaluation/shap_analysis.py` | SHAP importance + individual explanation | 10 |
| `evaluation/compare.py` | Accuracy / complexity / efficiency comparison table | 11 |
| `pipeline.py` | Orchestrates the above in spec order, writes `outputs/` | 2–11 |
| `report/` | Literature review + narrative report (prose, not code) | 1, 12 |

## 3. Data Flow

```
config.py
   │
   ▼
generate.py
   1. build technician roster: persistent skill effect + roster start date
      (staggered, so not every technician has a full history from day one)
   2. generate job records per technician in date order, each job_date on
      or after that technician's start date
   3. compute rolling_efficiency_30d / task_type_experience_count causally,
      per technician, using only that technician's own prior job_dates
   4. compute overrun_factor = clip(
        skill_effect(technician)
        × learning_curve(experience)
        × exp(staffing_log_effect)
        × lognormal_noise(sigma)
      , FLOOR, CAP)
   ▼
raw_df (n=500–1000 rows: modeling columns + record_id + job_date)
   │
   ▼
split.py ──► sort by job_date ──► train_df (earliest 70%) /
                                  cal_df (next 15%) /
                                  test_df (final 15%)
   │              │              │
   │              │              └──────────────────────────┐
   │              ▼                                          │
   │        sanity_check.py (uses linear.py's fit_tuned      │
   │        as the probe against a mean-predictor baseline)  │
   │              │                                          │
   │         pass │ fail → halt, adjust config.SIGMA_LOG,    │
   │              │        rerun from generate.py             │
   │              ▼                                          │
   │   ┌──────────┼──────────┐                               │
   │   ▼          ▼          ▼                                │
   │ linear.py  random_forest.py  xgboost_model.py            │
   │   │          │          │                                │
   │   └────┬─────┴─────┬────┘                                │
   │        ▼           ▼                                     │
   │   ModelResult   ModelResult(xgb) ──► conformal.py ◄───────┘
   │   (per model)        │                    │
   │        │             │              q_hat (calibrated)
   │        ▼             │
   │   joblib.dump()      │
   │   outputs/models/*.joblib
   │                      │
   │                      ▼                    │
   │                 test_df ───────► predict_with_interval()
   │                      │                    │
   │                      ▼                    ▼
   │                 metrics.py          metrics.py (coverage)
   │                      │                    │
   │                      └─────────┬──────────┘
   │                                ▼
   │                          compare.py ──► outputs/metrics/comparison.csv
   │                                │
   │                    xgboost ModelResult
   │                                ▼
   │                      shap_analysis.py ──► outputs/figures/
   │
   └────────────────────────────────────────────► report/report.md (manual synthesis)
```

Everything left of `report/` is code; `report/` is written by the student
referencing `outputs/metrics/` and `outputs/figures/` as source data — the
architecture does not automate prose generation.

## 4. Interfaces (precise contracts for the developer)

### `config.py`
Flat module-level constants: `SEED`, `N_RECORDS`, `SIGMA_LOG` (start 0.225),
`OVERRUN_FLOOR` (0.4), `OVERRUN_CAP` (3.5), `SPLIT_RATIOS` (0.70, 0.15, 0.15),
`CV_FOLDS` (5), `CONFORMAL_ALPHA` (0.10), `RF_PARAM_GRID`, `XGB_PARAM_GRID`,
`SANITY_CHECK_MIN_RATIO` (0.15), `SANITY_CHECK_MAX_RATIO` (0.95),
`JOB_DATE_WINDOW_DAYS` (e.g. 365 — span over which job dates are drawn),
`N_TECHNICIANS`.

### `data/generate.py`
```python
def generate_dataset(n_records: int = N_RECORDS, seed: int = SEED,
                      return_debug: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]
```
`return_debug=True` additionally returns a second DataFrame exposing the
internal signal components (skill_effect, learning_curve_multiplier,
staffing_log_effect, noise, raw_overrun) per record — generation/validation
scaffolding, not part of the modeling feature set. Used by the
technician-coverage diagnostic (§2) and by tests that need to verify causal
feature computation independently.

Internal process (deterministic given `seed`):
1. Build a technician roster, each with a persistent skill effect and a
   staggered roster start date within the job-date window — not every
   technician has a full history from day one, which is what makes the
   later causal rolling-feature computation meaningful rather than trivial.
2. Generate job records per technician, in date order, with `job_date` on
   or after that technician's start date.
3. Compute `rolling_efficiency_30d` and `task_type_experience_count`
   causally — for each job, using only that same technician's own prior
   `job_date`s. This is why generation must proceed in per-technician date
   order internally, even though the returned DataFrame's row order doesn't
   otherwise matter.
4. Compute `overrun_factor = clip(skill_effect × learning_curve(experience)
   × exp(staffing_log_effect) × lognormal_noise(sigma=SIGMA_LOG), FLOOR,
   CAP)`, where `staffing_log_effect = STAFFING_RATIO_LOG_COEF ×
   (staffing_ratio - STAFFING_RATIO_REFERENCE)` — the modest additive
   staffing main effect on the log-mean (config.py; see models/linear.py
   below for why it's additive in log-space specifically). The `clip(...,
   FLOOR, CAP)` wrapper is still active after this term was added — confirmed
   directly against the current code (`src/data/generate.py`, the
   `overrun_factor = float(np.clip(raw_overrun, cfg.OVERRUN_FLOOR,
   cfg.OVERRUN_CAP))` line).

Returns one row per job-card task with columns:
- Job-level: `task_type`, `estimated_duration_hours`, `task_category`
- Technician-level: `rolling_efficiency_30d`, `task_type_experience_count`, `learning_curve_index`
- Contextual: `shift_type`, `day_of_week`, `staffing_ratio`, `attendance_rate`
- `record_id` (unique), `job_date` (needed for split ordering)
- `overrun_factor` (target, clipped to `[OVERRUN_FLOOR, OVERRUN_CAP]`)

`technician_id` and raw `job_date` are **not** included in the modeling
feature set passed to any model (see §5) — both are generation/splitting
scaffolding, not features to fit against.

### `data/split.py`
```python
def make_splits(df: pd.DataFrame, date_col: str = "job_date",
                 ratios: tuple = SPLIT_RATIOS) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
```
Sorts `df` by `date_col` ascending, then slices by row position: earliest
`ratios[0]` fraction → `train_df`, next `ratios[1]` fraction →  `cal_df`,
remaining → `test_df`. No shuffling, no stratification — order is the whole
point. Returns `(train_df, cal_df, test_df)` and also writes them to
`data/generated/{train,calibration,test}.csv`. `job_date` and
`technician_id`-derived columns are dropped from the modeling feature set at
the `preprocessing.py` stage, not here — the split files retain `job_date`
for traceability/inspection.

### `features/preprocessing.py`
```python
def build_preprocessor(scale_numeric: bool) -> sklearn.compose.ColumnTransformer
```
Operates on the fixed modeling feature list (job-level + technician-level +
contextual columns from §4's `generate.py` contract, explicitly excluding
`record_id` and `job_date`). One shared categorical encoding (one-hot) for
all three models — chosen over per-model encodings (e.g. XGBoost's native
categorical support) so comparisons across models aren't confounded by
different feature representations. `scale_numeric=True` for Linear
Regression (adds `StandardScaler` on numeric columns); `False` for Random
Forest and XGBoost (tree models don't need it).

### `models/common.py`
```python
@dataclass
class ModelResult:
    name: str
    pipeline: Any                          # fitted, .predict()-ready (Pipeline for
                                            # RF/XGBoost; SmearedLogLinearModel
                                            # wrapping a Pipeline for linear — see below)
    best_params: dict
    single_fit_time_sec: float             # time of one representative fit
    total_tuning_time_sec: float           # full grid-search wall time (== single_fit_time for linear)
    n_fits: int                            # 1 for linear; n_candidates * cv_folds otherwise
    complexity: float                      # see §6
```
This is the one type every downstream stage (`sanity_check`, `conformal`,
`evaluation`) depends on — none of them need to know whether the underlying
estimator is `LinearRegression`, `RandomForestRegressor`, or `XGBRegressor`.

### `models/linear.py`, `random_forest.py`, `xgboost_model.py`
Each exposes:
```python
def fit_tuned(X_train: pd.DataFrame, y_train: pd.Series, cv: int = CV_FOLDS) -> ModelResult
```
`linear.py` does a plain `.fit()` (no grid search — no hyperparameters to
tune per spec), but fits on `log(overrun_factor)` rather than the raw
target: the target is generated multiplicatively (skill_effect ×
learning_curve × staffing main effect × noise), so log-space is where OLS's
linearity assumption actually holds. Naively exponentiating log-space
predictions back to the raw scale is not unbiased (Jensen's inequality —
confirmed empirically via a positive mean signed residual on train), so
predictions are corrected with Duan's smearing estimator: `smearing_factor
= mean(exp(log-scale training residuals))`, raw-scale prediction =
`exp(log_pred) * smearing_factor`. `SmearedLogLinearModel` (a small wrapper,
not a sklearn `BaseEstimator`) holds the fitted log-scale pipeline and the
smearing factor, and exposes `.predict()` returning raw-scale predictions —
every downstream caller (sanity_check.py, pipeline.py) still just calls
`.predict()` without needing to know about the transform or the correction.
Random Forest and XGBoost are deliberately left on the raw scale — tree
splits don't carry OLS's distributional assumption, so they don't need the
same fix.
`random_forest.py` and `xgboost_model.py` wrap
`GridSearchCV(pipeline, param_grid, cv=cv, scoring="neg_root_mean_squared_error")`
— note `cv` here is a **plain k-fold** over the training partition (already
chronologically the earliest 70%), not a further time-respecting split;
this is consistent with the spec's constraint (grid search CV within the
training partition) but is worth naming explicitly, since k-fold shuffles
within that partition rather than preserving order at the fold level (see
§5 trade-offs).

### `diagnostics/sanity_check.py`
```python
@dataclass
class SanityCheckResult:
    mean_baseline_rmse: float
    probe_rmse: float          # from linear.py's fit_tuned, reused as the probe
    ratio: float                # probe_rmse / mean_baseline_rmse
    verdict: Literal["ok", "noise_too_low", "noise_too_high"]

def run_sanity_check(train_df: pd.DataFrame) -> SanityCheckResult
```
Reuses the Linear Regression pipeline as the diagnostic probe rather than
building a separate throwaway model — spec tasks 4 and 5 share one
implementation, so this function is called once and its `pipeline` is reused
directly as the task-5 baseline if the check passes. `verdict = "noise_too_low"`
if `ratio < SANITY_CHECK_MIN_RATIO`; `"noise_too_high"` if
`ratio > SANITY_CHECK_MAX_RATIO`; otherwise `"ok"`. `pipeline.py` halts with
an explicit message pointing at `config.SIGMA_LOG` if the verdict isn't
`"ok"`.

### `conformal/conformal.py`
```python
def calibrate(xgb_pipeline, X_cal: pd.DataFrame, y_cal: pd.Series,
              alpha: float = CONFORMAL_ALPHA) -> float
```
Split conformal on absolute residuals: `q_hat` is the finite-sample-corrected
`ceil((n+1)(1-alpha))/n` empirical quantile of `|y_cal - xgb_pipeline.predict(X_cal)|`,
computed on the chronologically-later calibration partition (never on
training data).
```python
def predict_with_interval(xgb_pipeline, X: pd.DataFrame,
                           q_hat: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]
```
Returns `(y_hat, lower, upper)` with `lower = y_hat - q_hat`,
`upper = y_hat + q_hat`. Plain (symmetric, constant-width) split conformal —
see §5 for the trade-off, now compounded by the chronological calibration
set (§5 also covers that interaction).

### `evaluation/metrics.py`
```python
def rmse(y_true, y_pred) -> float
def mae(y_true, y_pred) -> float
def mape(y_true, y_pred) -> float
def empirical_coverage(y_true, lower, upper) -> float   # fraction inside [lower, upper]
```

### `evaluation/compare.py`
```python
def build_comparison_table(results: list[ModelResult],
                            test_metrics: dict[str, dict]) -> pd.DataFrame
```
One row per model; columns = RMSE, MAE, MAPE, `complexity`,
`single_fit_time_sec`, `total_tuning_time_sec`.

### `evaluation/shap_analysis.py`
```python
def compute_shap_values(xgb_pipeline, X_test: pd.DataFrame) -> shap.Explanation
def plot_global_importance(shap_values) -> pathlib.Path   # saved to outputs/figures/
def plot_individual_explanation(shap_values, index: int) -> pathlib.Path
```

### `pipeline.py`
Single `main()` that calls the above in spec order (2 → 3 → 4 → [5,6,7] → 8
→ 9 → 10 → 11), writing every artifact to `outputs/`. This is the only file
that encodes the full task ordering — every other module is independently
testable in isolation given the right inputs.

## 5. Architectural Risks / Trade-offs (surfaced, not hidden)

- **Job date and technician_id excluded from modeling features** — both are
  generation/splitting scaffolding. A raw `job_date` would let a model
  extrapolate on a value that is, by construction, always higher in
  test/calibration than in training (defeats the purpose of testing
  generalization); `technician_id` is high-cardinality and would let a model
  memorize individuals instead of generalizing from the derived features.
  `day_of_week` remains as the appropriate cyclical/contextual proxy from
  the original feature list.
- **k-fold CV inside a chronological training partition still shuffles
  within that partition** — the spec fixes grid-search tuning as k-fold CV
  within training (not nested time-series CV), and the training partition
  is itself already the earliest 70% of records. This is consistent with
  the spec, but worth naming: individual CV folds do not preserve
  within-training chronological order, so hyperparameter selection is not
  fully "causal" at the fold level even though the outer train/cal/test
  split is. Accepted as the simpler design the spec calls for; a
  time-series-aware CV (e.g. `TimeSeriesSplit`) would be the alternative if
  stricter temporal discipline were required at every level.
- **Calibration set is chronologically after training, not exchangeable
  with training in the usual conformal-prediction sense** — standard split
  conformal assumes calibration and test data are exchangeable with each
  other (which chronological ordering preserves, since both are "future"
  relative to training) but not necessarily with training itself. This is
  fine for the coverage guarantee (calibration and test are still adjacent,
  both "future" partitions), but if the data-generating process drifts over
  time (e.g. technician skill distribution shifts), interval width
  calibrated on the 15% calibration slice may not perfectly transfer to the
  15% test slice. Worth a line in the report's limitations section.
- **Shared one-hot encoding across all models** — chosen for a fair,
  apples-to-apples comparison; trade-off is that XGBoost's native
  categorical handling is deliberately not used. Acceptable given low
  cardinality (task_type, shift_type, day_of_week).
- **Plain split conformal, not adaptive/CQR** — produces constant-width
  intervals regardless of input. Simpler to implement and reason about, but
  doesn't shrink/widen per-prediction based on local uncertainty — a real
  limitation worth naming in the report's discussion, not just an
  implementation detail.
- **Complexity metric is a simplification** — `complexity` in `ModelResult`
  is proposed as: LR → number of coefficients; RF/XGBoost → `n_estimators`
  as a simple, comparable proxy. Not a rigorous model-complexity measure
  (ignores tree depth/leaf count); a deliberate simplification in favor of
  one comparable number across very different model families.
- **Sanity-check probe reuses the Linear Regression pipeline** — spec listed
  task 4 (sanity check) and task 5 (LR baseline) as separate steps; this
  design collapses them into one execution (fit once, use twice) since
  building a second throwaway diagnostic model would duplicate logic for no
  benefit. If the check fails, only `config.SIGMA_LOG` (and possibly the
  data itself, since regeneration is needed) changes — no code changes.
- **Split CSVs persisted to `data/generated/`** — chosen for reproducibility
  and manual inspection, at the cost of an extra I/O step versus keeping
  everything in memory within a single script run. Reasonable given this is
  a report-driven academic project.

## 6. Concrete Defaults (tunable, developer implements against these)

**Sanity-check thresholds** (`ratio = probe_rmse / mean_baseline_rmse`):
- `ratio < 0.15` → `noise_too_low`
- `ratio > 0.95` → `noise_too_high`
- otherwise → `ok`

Delivery (implementation) does not begin until this design is approved.
