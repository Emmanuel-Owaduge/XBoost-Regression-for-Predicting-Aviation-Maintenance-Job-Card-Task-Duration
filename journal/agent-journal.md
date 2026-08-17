# Agent Journal

This file is maintained by the `reflection` skill (see `.claude/skills/reflection/SKILL.md`).
Each entry corresponds to a significant task completed by an agent and records:
what happened, where the agent was uncertain, what assumptions were made,
and what was learned. This journal is the evidence base for Appendix A.4 of
the AI Coding Governance declaration.

---

## 2026-08-14 — Pipeline implementation (spec tasks 2–11)

Implemented the full modeling pipeline per `spec/project-specification.md`
and `spec/architecture.md`: `src/config.py`, `src/data/{generate,split}.py`,
`src/features/preprocessing.py`, `src/models/{common,linear,random_forest,
xgboost_model}.py`, `src/diagnostics/sanity_check.py`,
`src/conformal/conformal.py`, `src/evaluation/{metrics,compare,
shap_analysis}.py`, `src/pipeline.py`. Ran end-to-end successfully via
`python -m src.pipeline`. Literature review and `report/` prose (spec tasks
1, 12) were out of scope for this pass.

**Environment setup (not a spec decision, flagged for the record):** the
system Python was 3.14 with no matplotlib/xgboost/shap installed; xgboost
additionally failed to load without the OpenMP runtime. Created a
project-local `.venv` and ran `brew install libomp` (a one-time macOS
system dependency, not a repo change) to get xgboost/shap working. Added
`matplotlib` to `requirements.txt` — not named in the spec's "Python,
scikit-learn + xgboost" constraint, but required to persist SHAP plots to
`outputs/figures/` per the architecture's interface contract for
`shap_analysis.py`.

**Assumption made, not specified in the spec:** the target formula
(`skill_effect × learning_curve × log-normal noise`, clipped) uses only the
two named signal components. Job-level and contextual features (task_type,
estimated_duration_hours, task_category, shift_type, day_of_week,
staffing_ratio, attendance_rate) are generated independently of the target
— they carry no causal relationship to overrun_factor by construction. This
follows the spec literally ("two grounded signal components... plus
noise", nothing more), but means SHAP will likely show these features as
near-zero importance. That's a legitimate finding for the discussion
section, not a bug, but it's a modeling choice the student should be aware
of before writing up "feature selection based on hypothesised contribution"
in the report — some hypotheses will come back unconfirmed by design.

**Pipeline run results (seed=42, n=800):**
- Sanity check: ratio=0.904, verdict=`ok` (threshold: fails above 0.95) —
  passes, but close to the "noise too weak" boundary. Worth knowing before
  treating this seed's results as final; not adjusted unilaterally per the
  developer skill's instruction to flag rather than silently tune.
- Test-set RMSE: linear=0.361, random_forest=0.356, xgboost=0.335 — XGBoost
  modestly outperforms the linear baseline despite no explicit interaction
  term in the target formula (that revision was reversed). Plausible
  explanation: skill_effect and learning_curve combine multiplicatively in
  the generating process, which a raw (non-log-transformed) linear model
  can't represent perfectly even without a named "interaction" feature —
  worth a line in the discussion.
**Follow-up (2026-08-15): fixed the job_date clipping bug found in code review.**
Independent review (`review` skill) found that `np.clip(np.cumsum(gaps), 0,
available_days)` in `generate.py` pinned any overshooting job to the exact
window-boundary date, producing 52/800 (6.5%) records sharing one
fabricated `job_date` — concentrated entirely in the test partition (43% of
its 120 rows) since the boundary is the dataset's global max date. Since
day_of_week derives from job_date, this also collapsed day_of_week
diversity in test (55% "Tuesday" vs a natural ~14%). Fixed per the
student's explicit instructions: replaced clipping with per-job redraw
(`_generate_job_offsets`, MAX_GAP_RETRIES=100) — an overshooting gap is
redrawn, not clamped; if a technician's quota genuinely can't fit even
after 100 redraws, generation stops early for that technician (explicit
quota reduction, surfaced via `warnings.warn`) rather than fabricating a
date or silently dropping jobs. Verified: 0 duplicate job_date values
post-fix (down from 52). At seed=42, n_records=800 requested, 790 actually
generated — 4 technicians (0, 9, 14, 21) short by 1-4 jobs each, all
warned. All 16 tests still pass unmodified (none hardcoded n=800).
Re-ran the full pipeline: RMSE dropped across all three models (e.g.
XGBoost 0.335→0.293) and conformal coverage moved from under-covering
(0.842) to over-covering (0.966) vs. the 0.90 nominal target — expected,
since the prior numbers were partly measuring the artifact, not model
quality. The over-coverage itself is a new, legitimate number worth
watching on future runs, not yet investigated further.

Prior (now superseded) note, kept for the record:
- Conformal coverage: empirical 0.858 vs nominal 0.90 — under-coverage.
  Plausibly explained by the small calibration partition (n=120), matching
  the small-sample risk already documented in the spec. Not corrected;
  reported as-is for the student to address in the report's limitations
  section, consistent with the spec's "check MAPE/coverage empirically"
  instruction.

**Follow-up (same day): log-target fit for Linear Regression.** Student
asked what scale the sanity-check probe was fit on; answer surfaced that
Linear Regression was fitting raw `overrun_factor` directly, even though
the target is generated multiplicatively (skill_effect × learning_curve ×
noise) — a mismatch with OLS's linearity assumption. Changed
`models/linear.py` to fit on `log(overrun_factor)` via
`sklearn.compose.TransformedTargetRegressor`, which auto-exponentiates
`.predict()` output back to raw scale, so no downstream caller needed to
change. Random Forest and XGBoost intentionally left on the raw scale per
the student's instruction (tree splits don't carry that assumption).
`ModelResult.pipeline`'s type hint widened from `sklearn.pipeline.Pipeline`
to `sklearn.base.BaseEstimator` in `models/common.py` to reflect that it's
no longer always a plain Pipeline; `spec/architecture.md` updated to match.
Re-ran the pipeline: LR test RMSE improved 0.361 → 0.357 (now close to
Random Forest's 0.356); sanity-check ratio shifted 0.904 → 0.915 (still
`ok`, still close to the 0.95 boundary).
