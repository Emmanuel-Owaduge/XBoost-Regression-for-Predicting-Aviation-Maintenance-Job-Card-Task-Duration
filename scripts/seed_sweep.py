"""Multi-seed variance sweep.

Reimplements pipeline.main()'s steps 1-9 in-memory for a set of seeds, so
we get real seed-to-seed variance instead of a single-seed point estimate.

Deliberately does NOT call pipeline.main() or data.split.make_splits'
default output_dir: those write to data/generated/ and outputs/, which
would overwrite the canonical seed-42 run's artifacts. Splits here are
computed in-memory only (make_splits still writes CSVs, but redirected to
a scratch dir so the canonical dataset is untouched).

Confirmed root cause of the seed-binding trap this works around: in
data/generate.py, `generate_dataset(seed: int = cfg.SEED, ...)` binds its
default at module-def time (first import), so `generate_dataset()` with no
args ignores any later change to cfg.SEED -- the data would be frozen at
seed 42 every run. Model random_state (random_forest.py, xgboost_model.py)
is read live inside fit_tuned()'s body, so it DOES follow cfg.SEED. Fix:
pass seed explicitly to generate_dataset, and set cfg.SEED before fitting
so RF/XGB move with it.
"""

import json
import statistics
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as cfg
from src.conformal import conformal
from src.data.generate import generate_dataset
from src.data.split import make_splits
from src.diagnostics.sanity_check import run_sanity_check
from src.evaluation import metrics as metrics_mod
from src.models import random_forest as rf_model
from src.models import xgboost_model as xgb_model

FEATURE_COLS = cfg.CATEGORICAL_FEATURES + cfg.NUMERIC_FEATURES
SEEDS = [42, 7, 123, 2024, 31415, 8675309, 271828, 90210, 555, 1001]


def run_one_seed(seed: int, scratch_dir: Path) -> dict:
    cfg.SEED = seed  # so RF/XGB random_state (read live in fit_tuned) follows

    df = generate_dataset(seed=seed)  # explicit arg bypasses the frozen default
    train_df, cal_df, test_df = make_splits(df, output_dir=scratch_dir / str(seed))

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[cfg.TARGET_COL]
    X_cal = cal_df[FEATURE_COLS]
    y_cal = cal_df[cfg.TARGET_COL]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[cfg.TARGET_COL]

    sanity = run_sanity_check(train_df)
    lr_result = sanity.probe_result
    rf_result = rf_model.fit_tuned(X_train, y_train)
    xgb_result = xgb_model.fit_tuned(X_train, y_train)

    q_hat = conformal.calibrate(xgb_result.pipeline, X_cal, y_cal)
    _, lower, upper = conformal.predict_with_interval(xgb_result.pipeline, X_test, q_hat)
    coverage = metrics_mod.empirical_coverage(y_test, lower, upper)

    mid = len(test_df) // 2
    coverage_first_half = metrics_mod.empirical_coverage(
        y_test.iloc[:mid], lower[:mid], upper[:mid]
    )
    coverage_second_half = metrics_mod.empirical_coverage(
        y_test.iloc[mid:], lower[mid:], upper[mid:]
    )

    def half_split_metrics(y_true, preds) -> dict:
        return {
            "rmse": metrics_mod.rmse(y_true, preds),
            "mae": metrics_mod.mae(y_true, preds),
            "mape": metrics_mod.mape(y_true, preds),
        }

    test_metrics = {}
    test_metrics_first_half = {}
    test_metrics_second_half = {}
    naive_preds = np.ones(len(y_test))  # constant overrun_factor = 1.0, no fitting
    named_preds = [("naive_baseline", naive_preds)]
    named_preds += [(result.name, result.pipeline.predict(X_test)) for result in (lr_result, rf_result, xgb_result)]
    for name, preds in named_preds:
        test_metrics[name] = half_split_metrics(y_test, preds)
        test_metrics_first_half[name] = half_split_metrics(y_test.iloc[:mid], preds[:mid])
        test_metrics_second_half[name] = half_split_metrics(y_test.iloc[mid:], preds[mid:])

    return {
        "seed": seed,
        "sanity_ratio": sanity.ratio,
        "sanity_verdict": sanity.verdict,
        "coverage": coverage,
        "coverage_first_half": coverage_first_half,
        "coverage_second_half": coverage_second_half,
        "test_metrics": test_metrics,
        "test_metrics_first_half": test_metrics_first_half,
        "test_metrics_second_half": test_metrics_second_half,
    }


def summarize(values: list) -> str:
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.4f} ± {std:.4f}  (min {min(values):.4f}, max {max(values):.4f})"


def paired_beats_baseline(candidate: list, baseline: list) -> dict:
    """One-sided paired test: is `candidate` significantly lower (better)
    than `baseline` across the same seeds? Wilcoxon signed-rank is the
    primary test (n=10, no normality assumption); paired t-test reported
    alongside for reference."""
    w_stat, w_p = stats.wilcoxon(candidate, baseline, alternative="less")
    t_stat, t_p = stats.ttest_rel(candidate, baseline, alternative="less")
    return {
        "candidate_mean": statistics.mean(candidate),
        "baseline_mean": statistics.mean(baseline),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p": float(w_p),
        "ttest_stat": float(t_stat),
        "ttest_p": float(t_p),
    }


def paired_second_worse_than_first(second_half: list, first_half: list) -> dict:
    """One-sided paired test: is `second_half` error significantly higher
    (worse) than `first_half`, across the same seeds? Tests for concept
    drift / accuracy degradation over the chronological test period."""
    w_stat, w_p = stats.wilcoxon(second_half, first_half, alternative="greater")
    t_stat, t_p = stats.ttest_rel(second_half, first_half, alternative="greater")
    return {
        "first_half_mean": statistics.mean(first_half),
        "second_half_mean": statistics.mean(second_half),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p": float(w_p),
        "ttest_stat": float(t_stat),
        "ttest_p": float(t_p),
    }


def main():
    results = []
    with tempfile.TemporaryDirectory(prefix="seed_sweep_") as tmp:
        scratch_dir = Path(tmp)
        for seed in SEEDS:
            print(f"=== seed {seed} ===")
            r = run_one_seed(seed, scratch_dir)
            print(
                f"  sanity ratio={r['sanity_ratio']:.4f} ({r['sanity_verdict']}) "
                f"coverage={r['coverage']:.4f} "
                f"(first={r['coverage_first_half']:.4f} second={r['coverage_second_half']:.4f})"
            )
            for name, m in r["test_metrics"].items():
                print(
                    f"  {name}: rmse={m['rmse']:.4f} mae={m['mae']:.4f} mape={m['mape']:.4f}"
                )
            results.append(r)

    cfg.SEED = 42  # restore module-level default for anything imported after this

    out_path = Path(__file__).resolve().parent.parent / "outputs" / "metrics" / "seed_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw per-seed results written to {out_path}")

    print(f"\n=== Summary across {len(SEEDS)} seeds (mean ± std, min-max) ===")
    print(f"seeds used: {SEEDS}\n")

    model_names = list(results[0]["test_metrics"].keys())
    for name in model_names:
        print(f"[{name}]")
        for metric in ("rmse", "mae", "mape"):
            vals = [r["test_metrics"][name][metric] for r in results]
            print(f"  {metric}: {summarize(vals)}")
        print()

    print("[sanity check ratio]")
    print(f"  {summarize([r['sanity_ratio'] for r in results])}")
    verdicts = {r["sanity_verdict"] for r in results}
    print(f"  verdicts seen: {verdicts}\n")

    print("[conformal coverage]")
    print(f"  overall:     {summarize([r['coverage'] for r in results])}")
    print(f"  first half:  {summarize([r['coverage_first_half'] for r in results])}")
    print(f"  second half: {summarize([r['coverage_second_half'] for r in results])}")

    print(
        "\n=== Accuracy by chronological test half, across "
        f"{len(SEEDS)} seeds (concept drift check) ==="
    )
    print(
        "H1: second-half error > first-half error, paired by seed. "
        "alpha=0.05 (one-sided: tests for degradation only)\n"
    )
    alpha = 0.05
    for name in model_names:
        print(f"[{name}]")
        for metric in ("rmse", "mae", "mape"):
            first_vals = [r["test_metrics_first_half"][name][metric] for r in results]
            second_vals = [r["test_metrics_second_half"][name][metric] for r in results]
            print(f"  {metric}: first  {summarize(first_vals)}")
            print(f"  {metric}: second {summarize(second_vals)}")
            t = paired_second_worse_than_first(second_vals, first_vals)
            verdict = "DEGRADES (significant)" if t["wilcoxon_p"] < alpha else "no significant degradation"
            print(
                f"           Wilcoxon p={t['wilcoxon_p']:.4f}  paired-t p={t['ttest_p']:.4f}"
                f"  -> {verdict}"
            )
        print()

    print(
        "\n=== Does Linear Regression beat the naive baseline "
        "(overrun_factor = 1.0, no fitting)? ==="
    )
    print("H1: linear_regression metric < naive_baseline metric, paired by seed. alpha=0.05\n")
    alpha = 0.05
    all_significant = True
    for metric in ("rmse", "mae", "mape"):
        lr_vals = [r["test_metrics"]["linear_regression"][metric] for r in results]
        naive_vals = [r["test_metrics"]["naive_baseline"][metric] for r in results]
        t = paired_beats_baseline(lr_vals, naive_vals)
        verdict = "SIGNIFICANT" if t["wilcoxon_p"] < alpha else "not significant"
        all_significant &= t["wilcoxon_p"] < alpha
        print(
            f"  {metric}: LR={t['candidate_mean']:.4f} vs naive={t['baseline_mean']:.4f} "
            f"| Wilcoxon p={t['wilcoxon_p']:.4f}  paired-t p={t['ttest_p']:.4f}  -> {verdict}"
        )
    print(
        f"\n  Linear Regression {'DOES' if all_significant else 'does NOT'} significantly "
        f"outperform the naive baseline on all three metrics (alpha={alpha})."
    )


if __name__ == "__main__":
    main()
