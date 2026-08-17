# Specification: XGBoost Regression for Aviation Maintenance Job Card Task Duration

Status: DRAFT — design decisions resolved (see "Resolved Design Decisions" below);
pending final sign-off before any implementation/writing begins.

## Goal

Build and evaluate a supervised regression pipeline (Linear Regression baseline,
Random Forest, XGBoost with a conformal prediction layer) on a synthetic
aviation-maintenance job-card dataset to predict task-duration overrun, backed by
a literature review that justifies feature-aware regression over ARIMA-style
time series for this problem, and demonstrating explainability (SHAP) and
calibrated uncertainty quantification (conformal prediction). The deliverable
is an academic report: literature review, methodology, results, and discussion.

## WHAT

1. **Literature review** covering: regression modelling foundations; gradient
   boosting with focus on XGBoost; feature engineering for structured tabular
   data; uncertainty quantification via conformal prediction; ARIMA as a
   contrasting statistical approach; and an explicit research-gap statement
   (job card systems rely on static estimates, lacking feature-aware
   data-driven regression models).
2. **Synthetic dataset**: 500–1,000 records, each assigned a job date (needed
   for chronological ordering — see split, below — and for causal computation
   of the rolling/experience technician features). Target = ratio of actual to
   estimated task duration ("overrun factor"), generated from two grounded
   per-technician signal components (persistent skill effect, experience-based
   learning curve), a modest additive main effect from staffing_ratio on the
   log-mean (added after initial implementation, at the student's request —
   not an interaction, a plain third factor; see spec/architecture.md §4 for
   the exact expression), plus multiplicative log-normal noise (sigma ≈
   0.2–0.25 on the log scale, tuned empirically — see Resolved Design
   Decisions), then clipped to a floor/cap range (≈0.4–3.5, tuned
   empirically). Features across three categories — job-level (task type,
   estimated duration, task category), technician-level (rolling 30-day
   efficiency average, task-type experience count, learning curve index),
   contextual (shift type, day of week, staffing ratio, attendance metrics).
3. **Data split**: chronological 70% train / 15% calibration / 15% test —
   records sorted by job date, earliest 70% as train, next 15% as
   calibration, final 15% as test, applied consistently across all models.
   No separate validation split — hyperparameter tuning uses k-fold
   cross-validation inside the training partition instead. The calibration
   partition is reserved exclusively for XGBoost's conformal prediction step
   and is never used for tuning.
4. **Three models**: Linear Regression (interpretable baseline, no
   hyperparameter search needed), Random Forest (ensemble comparison),
   XGBoost (primary model). Hyperparameter tuning via k-fold cross-validated
   grid search within the training partition for RF and XGBoost.
5. **Conformal prediction** layer on XGBoost producing calibrated 90%
   prediction intervals alongside point estimates.
6. **Evaluation**: RMSE, MAE, MAPE on held-out test set for all three models;
   empirical coverage of the 90% intervals against the calibration set; SHAP
   feature importance rankings and individual prediction explanations for
   XGBoost; a three-way model comparison on predictive accuracy, model
   complexity, and computational efficiency (reported as single-fit time —
   a minor, likely-uninformative note — versus total time-to-tuned-model,
   which is the real efficiency comparison since XGBoost's grid search fits
   far more model instances than a single LR or RF fit).
7. **Discussion**: interpretation of results, limitations of synthetic data,
   and a qualitative discussion of transferability to healthcare appointment
   scheduling, logistics delivery estimation, and workforce operations.

## WHY

Job card management systems currently rely on static task-duration estimates.
No feature-aware, data-driven regression models are in common use for this
operational prediction problem despite multi-dimensional job, technician, and
contextual data being available. This project demonstrates that gap and a
candidate solution, using aviation maintenance as the application domain while
keeping the regression methodology as the actual research focus.

## CONSTRAINTS

- Python, scikit-learn + xgboost.
- Synthetic data only, 500–1,000 records — no real/proprietary data.
- Hyperparameter search restricted to grid search with cross-validation (not
  random/Bayesian search).
- Prediction intervals fixed at 90% nominal coverage.
- Chronological 70/15/15 train/calibration/test split (sorted by job date)
  applied consistently across all three models; no dedicated validation
  split — tuning uses k-fold CV inside the training partition.
- Deliverable is an academic report (implies literature citations, formal
  structure) rather than a production system.

## RISKS / TRADE-OFFS (surfaced, not hidden in a chosen option)

- **Small-sample effects**: at n=500–1,000, a 70/15/15 split leaves the
  calibration and test partitions at roughly 75–150 rows each, and k-fold CV
  inside training further thins per-fold samples. This affects the stability
  of both hyperparameter selection and empirical coverage estimates — worth
  flagging as a limitation rather than solving away.
- **Noise calibration is empirical, not fixed a priori**: sigma ≈ 0.2–0.25
  (log scale) is a starting point, not a guarantee. It must be checked after
  the pipeline runs — if the Linear Regression baseline gets suspiciously
  close to zero error, noise is too low (target too deterministic from
  technician/context alone); if nothing beats predicting the mean, noise is
  too high. Adjust sigma and re-run before treating results as final.
- **Floor/cap on overrun factor is also empirical**: 0.4–3.5 is a starting
  range chosen to keep MAPE's denominator away from zero while preserving a
  plausible severe-but-real tail. Must be checked once MAPE is computed on
  real pipeline output and tightened further if still unstable.
- **Transferability discussion is qualitative only**: healthcare, logistics,
  and workforce transferability will be argued narratively, not empirically
  tested. This should be scoped explicitly as discussion, not implied as a
  validated result.
- **Chronological split can create technician cohort imbalance**: technicians
  who only begin appearing late in the timeline are disproportionately pushed
  into calibration/test, so evaluation performance may partly reflect which
  technicians happen to fall in later periods rather than pure model quality.
  Accepted as a deliberate, documented trade-off in exchange for a
  leakage-free, deployment-realistic split — not something to engineer
  around (doing so, e.g. via technician-stratified sampling, would
  reintroduce the leakage the chronological split is meant to prevent).
- **Rolling/experience features must be computed causally**: `rolling_efficiency_30d`
  and `task_type_experience_count` must be computed using only each
  technician's history strictly prior to a given job's date. This was
  implicit in the original feature list but is now load-bearing, since the
  split itself depends on chronological ordering — any leakage in how these
  rolling features are computed would undermine the entire time-based split.
- **Computational efficiency has two sub-metrics of different value**:
  single-fit time is likely uninformative at this scale (sub-second for all
  three models) and should be reported as a minor note only. Total
  time-to-tuned-model (accounting for grid-search fit counts) is the
  meaningful comparison and should carry the actual "efficiency" claim in
  the discussion.

## SUCCESS / ACCEPTANCE CRITERIA

- Literature review draft addresses all five listed areas and states the
  research gap explicitly.
- Synthetic data generation is reproducible (fixed seed) and documented,
  including the two-component signal (technician skill effect, learning
  curve), the log-normal noise term and its sigma, and the floor/cap applied
  to the overrun factor.
- Rolling/experience technician features are computed causally (only from
  each technician's history prior to a given job's date), and the
  train/calibration/test split is strictly chronological with no future
  record leaking into training or calibration.
- Sanity check performed and documented: baseline error is neither
  near-zero (signal too strong/deterministic) nor no-better-than-mean-prediction
  (signal too weak); sigma adjusted and re-run if either failure mode appears.
- All three models trained and tuned (RF/XGBoost via k-fold CV grid search on
  the training partition only), evaluated on the same held-out test set, with
  RMSE/MAE/MAPE reported per model.
- XGBoost conformal prediction intervals produced using the dedicated 15%
  calibration split, with empirical coverage reported against the 90%
  nominal target.
- SHAP global feature importance and at least one individual prediction
  explanation produced for XGBoost.
- Comparison table across accuracy / complexity / compute-time (single-fit
  time noted as minor; total tuning time reported as the primary efficiency
  comparison) for all three models.
- Discussion section covers interpretation, synthetic-data limitations, and
  transferability to the three named domains.

## TASK DECOMPOSITION (ordered, dependencies noted)

1. **Literature review draft** — independent, no code dependency. Can proceed
   in parallel with everything below.
2. **Synthetic data generation design** — defines the two-component signal
   (technician skill effect + learning curve), per-record job dates for
   chronological ordering and causal rolling-feature computation, log-normal
   noise (sigma ≈ 0.2–0.25 to start), the 0.4–3.5 floor/cap, feature
   distributions, and category encodings. No dependencies; must be resolved
   before 3.
3. **Data split implementation** — chronological 70/15/15
   train/calibration/test, sorted by job date. Depends on 2.
4. **Signal-strength sanity check** — fit a trivial baseline (e.g. predict
   the mean, or single-feature regression) against the training split to
   confirm sigma is in a reasonable range before investing in the full model
   suite. Depends on 3. Loop back to 2 and re-run if the check fails.
5. **Linear Regression baseline** — depends on 4.
6. **Random Forest + k-fold CV grid search (within training partition)** —
   depends on 4. Independent of 5 and 7.
7. **XGBoost + k-fold CV grid search (within training partition)** —
   depends on 4. Independent of 5 and 6.
8. **Conformal prediction layer on XGBoost** — depends on 7 and the
   calibration split from 3.
9. **Evaluation suite** (RMSE/MAE/MAPE for all models on test split;
   empirical coverage for XGBoost against calibration-derived intervals) —
   depends on 5, 6, 7, 8.
10. **SHAP analysis on XGBoost** — depends on 7.
11. **Model comparison table** (accuracy / complexity / single-fit time as
    minor note / total tuning time as primary efficiency measure) —
    depends on 9.
12. **Discussion & report integration** — depends on 1, 9, 10, 11.

Tasks 5, 6, 7 can run in parallel once 4 passes. Task 1 can run in parallel
with 2–11 throughout.

## RESOLVED DESIGN DECISIONS

1. **Calibration split** — chronological 70% train / 15% calibration / 15%
   test: records sorted by job date, earliest 70% as train, next 15% as
   calibration, final 15% as test. No separate validation split and no
   fourth split; hyperparameter tuning uses k-fold CV inside the training
   partition instead of a dedicated validation set. Rationale: sorting by
   time rules out leakage directly (no test record can chronologically
   precede a training record) and matches how the system would actually be
   deployed — predicting future jobs from past history, not
   randomly-interspersed ones. Trade-off accepted deliberately, not by
   default: technicians who only appear late in the timeline are
   disproportionately pushed into calibration/test, which can affect
   evaluation numbers for reasons unrelated to model quality. This is
   documented as a limitation rather than corrected for, since correcting
   for it (e.g. technician-stratified sampling) would reintroduce the
   leakage this split is designed to avoid. Supersedes the earlier
   stratified-random-split decision.
2. **Target generation formula** — two grounded signal components (persistent
   per-technician skill effect; experience-based learning curve) plus
   multiplicative log-normal noise (chosen because duration ratios are
   strictly positive and real-world variability here is right-skewed and
   multiplicative rather than symmetric). Starting sigma ≈ 0.2–0.25 on the
   log scale — strong enough that a model exploiting real features should
   meaningfully beat a naive baseline, not so weak that the outcome is
   basically deterministic. To be validated empirically (task 4 above) and
   adjusted if the baseline is suspiciously accurate or no better than
   predicting the mean. **Amended after initial implementation**: a modest
   additive main effect from staffing_ratio on the log-mean was added as a
   third factor, at the student's explicit request — a plain main effect,
   not an interaction (an earlier, different interaction-term proposal —
   e.g. learning-curve strength moderated by shift type — was considered and
   reversed; this staffing effect is unrelated to that reversed proposal).
   See spec/architecture.md §4 for the exact expression.
3. **Overrun-factor range** — floor ≈0.4–0.5, cap ≈3–3.5, applied because
   MAPE's denominator is the true value and percentage error blows up as
   that approaches zero (a known instability in the metric itself). To be
   checked empirically once MAPE is computed on real pipeline output and
   tightened further if still unstable.
4. **Computational efficiency axis** — reported as two sub-metrics rather
   than one: single-fit time (minor, likely-uninformative note, since all
   three models fit in a fraction of a second at this scale) and total
   time-to-tuned-model (the real comparison, since XGBoost's grid search
   fits many more model instances than a single LR or RF fit).

Delivery does not begin until this specification is approved.
