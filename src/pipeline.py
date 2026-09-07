"""Single entry point. Runs every stage in spec task order (2 -> 11) and
writes all artifacts to outputs/. This is the only file that encodes the
full pipeline ordering — every other module is independently testable given
the right inputs. Run with: python -m src.pipeline
"""

import json
import sys

import joblib

from src import config as cfg
from src.conformal import conformal
from src.data.generate import generate_dataset
from src.data.split import make_splits
from src.diagnostics.sanity_check import run_sanity_check
from src.evaluation import metrics as metrics_mod
from src.evaluation.compare import build_comparison_table
from src.evaluation.shap_analysis import (
    compute_shap_values,
    plot_global_importance,
    plot_individual_explanation,
)
from src.models import random_forest as rf_model
from src.models import xgboost_model as xgb_model

FEATURE_COLS = cfg.CATEGORICAL_FEATURES + cfg.NUMERIC_FEATURES


def main():
    cfg.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/10] Generating synthetic dataset (n={cfg.N_RECORDS}, seed={cfg.SEED})...")
    df = generate_dataset()
    print(
        f"      Generated {len(df)} records spanning "
        f"{df[cfg.DATE_COL].min().date()} to {df[cfg.DATE_COL].max().date()}"
    )

    print("[2/10] Splitting chronologically (70/15/15)...")
    train_df, cal_df, test_df = make_splits(df)
    print(f"      train={len(train_df)} calibration={len(cal_df)} test={len(test_df)}")

    print("[3/10] Running signal-strength sanity check...")
    sanity = run_sanity_check(train_df)
    print(
        f"      mean-baseline RMSE={sanity.mean_baseline_rmse:.4f} "
        f"probe RMSE={sanity.probe_rmse:.4f} ratio={sanity.ratio:.3f} "
        f"verdict={sanity.verdict}"
    )
    print(
        f"      probe mean signed residual (raw scale, train, y_true - y_pred)="
        f"{sanity.probe_mean_signed_residual:+.4f}"
    )
    if sanity.verdict != "ok":
        print(
            f"HALT: sanity check verdict is '{sanity.verdict}'. "
            f"Adjust config.SIGMA_LOG and rerun from data generation "
            f"before proceeding to the full model suite."
        )
        sys.exit(1)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[cfg.TARGET_COL]
    X_cal = cal_df[FEATURE_COLS]
    y_cal = cal_df[cfg.TARGET_COL]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[cfg.TARGET_COL]

    print("[4/10] Reusing sanity-check probe as the Linear Regression baseline...")
    lr_result = sanity.probe_result
    print(f"      Duan smearing_factor={lr_result.best_params['smearing_factor']:.4f}")

    print("[5/10] Fitting Random Forest (TimeSeriesSplit CV grid search)...")
    rf_result = rf_model.fit_tuned(X_train, y_train)
    print(f"      best_params={rf_result.best_params}")

    print("[6/10] Fitting XGBoost (TimeSeriesSplit CV grid search)...")
    xgb_result = xgb_model.fit_tuned(X_train, y_train)
    print(f"      best_params={xgb_result.best_params}")

    print("[7/10] Persisting fitted model pipelines...")
    for result in (lr_result, rf_result, xgb_result):
        model_path = cfg.MODELS_DIR / f"{result.name}.joblib"
        joblib.dump(result.pipeline, model_path)
        print(f"      wrote {model_path}")

    print("[8/10] Calibrating conformal prediction on XGBoost...")
    q_hat = conformal.calibrate(xgb_result.pipeline, X_cal, y_cal)
    _, lower, upper = conformal.predict_with_interval(xgb_result.pipeline, X_test, q_hat)
    coverage = metrics_mod.empirical_coverage(y_test, lower, upper)
    print(f"      q_hat={q_hat:.4f} empirical coverage={coverage:.3f} (nominal 0.90)")

    # test_df is already chronologically sorted (inherited from split.py's
    # global sort), so a positional midpoint split is a chronological split.
    mid = len(test_df) // 2
    coverage_first_half = metrics_mod.empirical_coverage(
        y_test.iloc[:mid], lower[:mid], upper[:mid]
    )
    coverage_second_half = metrics_mod.empirical_coverage(
        y_test.iloc[mid:], lower[mid:], upper[mid:]
    )
    print(
        f"      coverage by chronological test half: "
        f"first={coverage_first_half:.3f} (n={mid}) "
        f"second={coverage_second_half:.3f} (n={len(test_df) - mid})"
    )

    print("[9/10] Evaluating all models on the test set...")
    results = [lr_result, rf_result, xgb_result]
    test_metrics = {}
    for result in results:
        preds = result.pipeline.predict(X_test)
        test_metrics[result.name] = {
            "rmse": metrics_mod.rmse(y_test, preds),
            "mae": metrics_mod.mae(y_test, preds),
            "mape": metrics_mod.mape(y_test, preds),
        }
        print(f"      {result.name}: {test_metrics[result.name]}")

    comparison = build_comparison_table(results, test_metrics)
    comparison.to_csv(cfg.METRICS_DIR / "comparison.csv", index=False)

    coverage_report = {
        "nominal_coverage": 1 - cfg.CONFORMAL_ALPHA,
        "empirical_coverage": coverage,
        "empirical_coverage_first_half": coverage_first_half,
        "empirical_coverage_second_half": coverage_second_half,
        "q_hat": q_hat,
    }
    with open(cfg.METRICS_DIR / "conformal_coverage.json", "w") as f:
        json.dump(coverage_report, f, indent=2)

    print("[10/10] Computing SHAP values for XGBoost...")
    shap_values = compute_shap_values(xgb_result.pipeline, X_test)
    global_path = plot_global_importance(shap_values)
    individual_path = plot_individual_explanation(shap_values, index=0)
    print(f"      SHAP figures written to {global_path} and {individual_path}")

    print("\nPipeline complete.\n")
    print(comparison.to_string(index=False))

    return {
        "sanity": sanity,
        "results": results,
        "test_metrics": test_metrics,
        "coverage": coverage_report,
        "comparison": comparison,
    }


if __name__ == "__main__":
    main()
