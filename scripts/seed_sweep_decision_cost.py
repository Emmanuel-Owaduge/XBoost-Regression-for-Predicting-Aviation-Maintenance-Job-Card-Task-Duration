"""Decision-cost proxy: point-estimate vs. interval-aware staffing rules.

*** MODELLED PROXY, NOT REAL OPERATIONAL VALIDATION ***
This compares two hypothetical staffing-buffer rules using XGBoost's
already-fitted point predictions and conformal-calibrated intervals against
the synthetic test set's actual overrun_factor. It is a decision-cost
simulation built entirely on the synthetic data-generating process and an
assumed linear cost function -- it does NOT reflect a real maintenance
operation's staffing behaviour, technician availability constraints, or
actual cost structure. Every number below should be read as "what this
proxy implies", not "what would happen if deployed".

Same 10 seeds and same in-memory refit pattern as scripts/seed_sweep.py
(see that file's docstring for the seed-binding trap this works around).
For each seed: fit XGBoost on train, calibrate conformal intervals on the
calibration split (90% nominal, cfg.CONFORMAL_ALPHA), predict on test.

Two staffing rules, per test record (estimated_duration_hours = dur):
  point rule:    allocated_time = y_hat_overrun  * dur   (buffer = that - dur)
  interval rule: allocated_time = upper_overrun  * dur   (buffer = that - dur)
Since upper = y_hat + q_hat with q_hat >= 0, the interval rule always
allocates a buffer >= the point rule's by construction -- it can only trade
shortfall cost for unused-buffer cost, never both improve independently.

Cost per record, given allocated_time and actual_time = actual_overrun * dur:
  understaffed (actual > allocated): cost = (actual - allocated) * RATIO
  overstaffed  (actual <= allocated): cost = (allocated - actual) * 1
RATIO is swept over {2, 3, 5}; 3 is the primary assumption (understaffing
cascades operationally, overstaffing only wastes capacity), 2 and 5 are a
sensitivity check so the finding doesn't rest on one arbitrary number.
Shortfall and unused-buffer are accumulated once per rule per seed (both
ratio-independent); total cost per ratio is then a cheap linear combination,
not a re-simulation.
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
from src.models import xgboost_model as xgb_model

FEATURE_COLS = cfg.CATEGORICAL_FEATURES + cfg.NUMERIC_FEATURES
SEEDS = [42, 7, 123, 2024, 31415, 8675309, 271828, 90210, 555, 1001]
RATIOS = (2, 3, 5)
PRIMARY_RATIO = 3


def rule_shortfall_unused(actual_time: np.ndarray, allocated_time: np.ndarray) -> tuple[float, float]:
    """Sum of shortfall (understaffed) and sum of unused buffer (overstaffed)
    across records, for one rule. Cost at a given ratio = ratio*shortfall + unused."""
    diff = actual_time - allocated_time
    shortfall = np.clip(diff, a_min=0, a_max=None)
    unused = np.clip(-diff, a_min=0, a_max=None)
    return float(shortfall.sum()), float(unused.sum())


def run_one_seed(seed: int, scratch_dir: Path) -> dict:
    cfg.SEED = seed  # so XGB random_state (read live in fit_tuned) follows

    df = generate_dataset(seed=seed)  # explicit arg bypasses the frozen default
    train_df, cal_df, test_df = make_splits(df, output_dir=scratch_dir / str(seed))

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[cfg.TARGET_COL]
    X_cal = cal_df[FEATURE_COLS]
    y_cal = cal_df[cfg.TARGET_COL]
    X_test = test_df[FEATURE_COLS]

    xgb_result = xgb_model.fit_tuned(X_train, y_train)
    q_hat = conformal.calibrate(xgb_result.pipeline, X_cal, y_cal)
    y_hat, _, upper = conformal.predict_with_interval(xgb_result.pipeline, X_test, q_hat)

    dur = test_df["estimated_duration_hours"].to_numpy()
    actual_overrun = test_df[cfg.TARGET_COL].to_numpy()
    actual_time = actual_overrun * dur

    allocated_point = y_hat * dur
    allocated_interval = upper * dur

    point_shortfall, point_unused = rule_shortfall_unused(actual_time, allocated_point)
    interval_shortfall, interval_unused = rule_shortfall_unused(actual_time, allocated_interval)

    costs_by_ratio = {}
    for ratio in RATIOS:
        point_cost = ratio * point_shortfall + point_unused
        interval_cost = ratio * interval_shortfall + interval_unused
        winner = "interval_aware" if interval_cost < point_cost else (
            "point_estimate" if point_cost < interval_cost else "tie"
        )
        costs_by_ratio[str(ratio)] = {
            "point_estimate_total_cost": point_cost,
            "interval_aware_total_cost": interval_cost,
            "winner": winner,
        }

    return {
        "seed": seed,
        "n_test": len(test_df),
        "q_hat": q_hat,
        "point_estimate_shortfall_sum": point_shortfall,
        "point_estimate_unused_buffer_sum": point_unused,
        "interval_aware_shortfall_sum": interval_shortfall,
        "interval_aware_unused_buffer_sum": interval_unused,
        "costs_by_ratio": costs_by_ratio,
    }


def summarize(values: list) -> str:
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.4f} ± {std:.4f}  (min {min(values):.4f}, max {max(values):.4f})"


def paired_interval_beats_point(interval_costs: list, point_costs: list) -> dict:
    """One-sided paired test: is interval-aware total cost significantly
    lower than point-estimate total cost across the same seeds? Wilcoxon
    signed-rank is primary (n=10, no normality assumption); paired t-test
    reported alongside for reference."""
    w_stat, w_p = stats.wilcoxon(interval_costs, point_costs, alternative="less")
    t_stat, t_p = stats.ttest_rel(interval_costs, point_costs, alternative="less")
    return {
        "interval_aware_mean": statistics.mean(interval_costs),
        "point_estimate_mean": statistics.mean(point_costs),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p": float(w_p),
        "ttest_stat": float(t_stat),
        "ttest_p": float(t_p),
    }


def main():
    print("*** MODELLED DECISION-COST PROXY -- NOT REAL OPERATIONAL VALIDATION ***")
    print("Synthetic test set, XGBoost point predictions + conformal intervals, ")
    print("assumed linear understaffed/overstaffed cost function. See module ")
    print("docstring for scope and caveats.\n")

    results = []
    with tempfile.TemporaryDirectory(prefix="seed_sweep_decision_cost_") as tmp:
        scratch_dir = Path(tmp)
        for seed in SEEDS:
            print(f"=== seed {seed} ===")
            r = run_one_seed(seed, scratch_dir)
            print(f"  n_test={r['n_test']} q_hat={r['q_hat']:.4f}")
            for ratio in RATIOS:
                c = r["costs_by_ratio"][str(ratio)]
                print(
                    f"  ratio={ratio}:1  point={c['point_estimate_total_cost']:.2f}  "
                    f"interval={c['interval_aware_total_cost']:.2f}  -> {c['winner']}"
                )
            results.append(r)

    cfg.SEED = 42  # restore module-level default for anything imported after this

    out_path = Path(__file__).resolve().parent.parent / "outputs" / "metrics" / "decision_cost_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "disclaimer": (
                    "Modelled decision-cost proxy on the synthetic test set. "
                    "Not real operational validation."
                ),
                "ratios": list(RATIOS),
                "primary_ratio": PRIMARY_RATIO,
                "seeds": SEEDS,
                "per_seed": results,
            },
            f,
            indent=2,
        )
    print(f"\nRaw per-seed results written to {out_path}")

    print(
        f"\n=== Decision-cost proxy summary across {len(SEEDS)} seeds "
        "(mean ± std, min-max) ===  [MODELLED PROXY]"
    )
    print(f"seeds used: {SEEDS}\n")

    alpha = 0.05
    summary = {}
    for ratio in RATIOS:
        tag = " (PRIMARY)" if ratio == PRIMARY_RATIO else ""
        print(f"[penalty ratio {ratio}:1{tag}]")
        point_vals = [r["costs_by_ratio"][str(ratio)]["point_estimate_total_cost"] for r in results]
        interval_vals = [r["costs_by_ratio"][str(ratio)]["interval_aware_total_cost"] for r in results]
        print(f"  point-estimate rule total cost:  {summarize(point_vals)}")
        print(f"  interval-aware rule total cost:  {summarize(interval_vals)}")

        wins = [r["costs_by_ratio"][str(ratio)]["winner"] for r in results]
        point_wins = wins.count("point_estimate")
        interval_wins = wins.count("interval_aware")
        ties = wins.count("tie")
        print(f"  per-seed wins: interval_aware={interval_wins}  point_estimate={point_wins}  tie={ties}")

        t = paired_interval_beats_point(interval_vals, point_vals)
        verdict = "INTERVAL-AWARE SIGNIFICANTLY LOWER COST" if t["wilcoxon_p"] < alpha else "no significant difference"
        print(
            f"  H1: interval_aware cost < point_estimate cost, paired by seed "
            f"(alpha={alpha}) -> Wilcoxon p={t['wilcoxon_p']:.4f}  paired-t p={t['ttest_p']:.4f}  -> {verdict}"
        )
        print()

        summary[str(ratio)] = {
            "point_estimate_mean": statistics.mean(point_vals),
            "point_estimate_std": statistics.stdev(point_vals) if len(point_vals) > 1 else 0.0,
            "interval_aware_mean": statistics.mean(interval_vals),
            "interval_aware_std": statistics.stdev(interval_vals) if len(interval_vals) > 1 else 0.0,
            "interval_aware_wins": interval_wins,
            "point_estimate_wins": point_wins,
            "ties": ties,
            "paired_test": {**t, "alpha": alpha, "verdict": verdict},
        }

    summary_path = Path(__file__).resolve().parent.parent / "outputs" / "metrics" / "decision_cost_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "disclaimer": (
                    "Modelled decision-cost proxy on the synthetic test set. "
                    "Not real operational validation."
                ),
                "primary_ratio": PRIMARY_RATIO,
                "by_ratio": summary,
            },
            f,
            indent=2,
        )
    print(f"Summary stats written to {summary_path}")


if __name__ == "__main__":
    main()
