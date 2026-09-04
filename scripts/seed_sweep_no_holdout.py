"""Ablation: does the calibration holdout cost point-prediction accuracy?

Same 10 seeds as scripts/seed_sweep.py. For each seed, train+calibration
are combined into a single fit set (no calibration holdout withheld) and
Linear Regression, Random Forest, and XGBoost are refit on it, then
evaluated on the same chronological test split used by the with-holdout
run. Results are compared, paired by seed, against the with-holdout numbers
already recorded in outputs/metrics/seed_sweep.json.

Random Forest and XGBoost each get their own independent GridSearchCV grid
search on the larger combined set -- fit_tuned() always fits a fresh
GridSearchCV, so calling it on train+calibration naturally re-tunes rather
than reusing the with-holdout best_params_. Reusing those would confound
"does more data help" with "were those hyperparameters right for this data
size", which defeats the point of the comparison. Linear Regression has no
hyperparameters, so this is a direct refit.

Same seed-binding fix as seed_sweep.py: pass seed explicitly to
generate_dataset, set cfg.SEED before fitting so RF/XGB random_state (read
live in fit_tuned) follows. Splits are computed in-memory only (make_splits
still writes CSVs, but redirected to a scratch dir so the canonical
seed-42 dataset in data/generated/ is untouched).
"""

import json
import statistics
import sys
import tempfile
from pathlib import Path

import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as cfg
from src.data.generate import generate_dataset
from src.data.split import make_splits
from src.evaluation import metrics as metrics_mod
from src.models import linear as linear_model
from src.models import random_forest as rf_model
from src.models import xgboost_model as xgb_model

FEATURE_COLS = cfg.CATEGORICAL_FEATURES + cfg.NUMERIC_FEATURES
SEEDS = [42, 7, 123, 2024, 31415, 8675309, 271828, 90210, 555, 1001]
MODEL_NAMES = ("linear_regression", "random_forest", "xgboost")
WITH_HOLDOUT_PATH = (
    Path(__file__).resolve().parent.parent / "outputs" / "metrics" / "seed_sweep.json"
)


def run_one_seed(seed: int, scratch_dir: Path) -> dict:
    cfg.SEED = seed  # so RF/XGB random_state (read live in fit_tuned) follows

    df = generate_dataset(seed=seed)  # explicit arg bypasses the frozen default
    train_df, cal_df, test_df = make_splits(df, output_dir=scratch_dir / str(seed))
    train_full_df = pd.concat([train_df, cal_df], ignore_index=True)

    X_train_full = train_full_df[FEATURE_COLS]
    y_train_full = train_full_df[cfg.TARGET_COL]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[cfg.TARGET_COL]

    lr_result = linear_model.fit_tuned(X_train_full, y_train_full)
    rf_result = rf_model.fit_tuned(X_train_full, y_train_full)
    xgb_result = xgb_model.fit_tuned(X_train_full, y_train_full)

    test_metrics = {}
    best_params = {}
    for result in (lr_result, rf_result, xgb_result):
        preds = result.pipeline.predict(X_test)
        test_metrics[result.name] = {
            "rmse": metrics_mod.rmse(y_test, preds),
            "mae": metrics_mod.mae(y_test, preds),
            "mape": metrics_mod.mape(y_test, preds),
        }
        best_params[result.name] = result.best_params

    return {
        "seed": seed,
        "n_train_full": len(train_full_df),
        "n_test": len(test_df),
        "test_metrics": test_metrics,
        "best_params": best_params,
    }


def summarize(values: list) -> str:
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.4f} ± {std:.4f}  (min {min(values):.4f}, max {max(values):.4f})"


def paired_wilcoxon(no_holdout: list, with_holdout: list) -> dict:
    """One-sided paired test: is `no_holdout` error significantly lower than
    `with_holdout` error, across the same seeds? H1 = the calibration
    holdout costs point-prediction accuracy (more train data -> lower
    error). Wilcoxon signed-rank is primary (n=10, no normality assumption);
    paired t-test reported alongside for reference."""
    w_stat, w_p = stats.wilcoxon(no_holdout, with_holdout, alternative="less")
    t_stat, t_p = stats.ttest_rel(no_holdout, with_holdout, alternative="less")
    return {
        "no_holdout_mean": statistics.mean(no_holdout),
        "with_holdout_mean": statistics.mean(with_holdout),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p": float(w_p),
        "ttest_stat": float(t_stat),
        "ttest_p": float(t_p),
    }


def main():
    if not WITH_HOLDOUT_PATH.exists():
        print(f"ERROR: {WITH_HOLDOUT_PATH} not found -- run scripts/seed_sweep.py first.")
        sys.exit(1)
    with open(WITH_HOLDOUT_PATH) as f:
        with_holdout_results = json.load(f)
    with_holdout_by_seed = {r["seed"]: r for r in with_holdout_results}
    missing = [s for s in SEEDS if s not in with_holdout_by_seed]
    if missing:
        print(f"ERROR: seeds {missing} missing from {WITH_HOLDOUT_PATH}. Re-run scripts/seed_sweep.py.")
        sys.exit(1)

    results = []
    with tempfile.TemporaryDirectory(prefix="seed_sweep_no_holdout_") as tmp:
        scratch_dir = Path(tmp)
        for seed in SEEDS:
            print(f"=== seed {seed} ===")
            r = run_one_seed(seed, scratch_dir)
            print(f"  n_train_full={r['n_train_full']} n_test={r['n_test']}")
            for name, m in r["test_metrics"].items():
                print(f"  {name}: rmse={m['rmse']:.4f} mae={m['mae']:.4f} mape={m['mape']:.4f}")
            results.append(r)

    cfg.SEED = 42  # restore module-level default for anything imported after this

    out_path = (
        Path(__file__).resolve().parent.parent / "outputs" / "metrics" / "seed_sweep_no_holdout.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw per-seed results written to {out_path}")

    print(f"\n=== No-holdout ablation: train+cal combined, across {len(SEEDS)} seeds (mean ± std, min-max) ===")
    print(f"seeds used: {SEEDS}\n")
    for name in MODEL_NAMES:
        print(f"[{name}]")
        for metric in ("rmse", "mae", "mape"):
            vals = [r["test_metrics"][name][metric] for r in results]
            print(f"  {metric}: {summarize(vals)}")
        print()

    print(f"=== With-holdout baseline (outputs/metrics/seed_sweep.json), same {len(SEEDS)} seeds ===\n")
    for name in MODEL_NAMES:
        print(f"[{name}]")
        for metric in ("rmse", "mae", "mape"):
            vals = [with_holdout_by_seed[s]["test_metrics"][name][metric] for s in SEEDS]
            print(f"  {metric}: {summarize(vals)}")
        print()

    print(
        "\n=== Does dropping the calibration holdout (train+cal combined, no "
        f"calibration step) improve point-prediction accuracy? ({len(SEEDS)} seeds) ==="
    )
    print(
        "H1: no_holdout metric < with_holdout metric, paired by seed. "
        "alpha=0.05 (one-sided: tests whether the holdout costs accuracy)\n"
    )
    alpha = 0.05
    comparison = {}
    for name in MODEL_NAMES:
        print(f"[{name}]")
        comparison[name] = {}
        for metric in ("rmse", "mae", "mape"):
            no_h = [r["test_metrics"][name][metric] for r in results]
            with_h = [with_holdout_by_seed[s]["test_metrics"][name][metric] for s in SEEDS]
            t = paired_wilcoxon(no_h, with_h)
            verdict = "HOLDOUT COSTS ACCURACY (significant)" if t["wilcoxon_p"] < alpha else "no significant cost"
            comparison[name][metric] = {**t, "alpha": alpha, "verdict": verdict}
            print(
                f"  {metric}: no_holdout={t['no_holdout_mean']:.4f} "
                f"with_holdout={t['with_holdout_mean']:.4f} "
                f"| Wilcoxon p={t['wilcoxon_p']:.4f}  paired-t p={t['ttest_p']:.4f}  -> {verdict}"
            )
        print()

    comparison_path = (
        Path(__file__).resolve().parent.parent
        / "outputs" / "metrics" / "no_holdout_ablation_comparison.json"
    )
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Comparison stats written to {comparison_path}")


if __name__ == "__main__":
    main()
