"""Hyperparameter-selection and timing sweep, same 10 seeds as
scripts/seed_sweep.py. That script discarded rf_result/xgb_result after
pulling test_metrics, so best_params_/timing were never persisted -- this
captures them instead. Same seed-binding fix as seed_sweep.py: pass seed
explicitly to generate_dataset, set cfg.SEED before fitting so RF/XGB
random_state (read live in fit_tuned) follows.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as cfg
from src.data.generate import generate_dataset
from src.data.split import make_splits
from src.models import random_forest as rf_model
from src.models import xgboost_model as xgb_model

FEATURE_COLS = cfg.CATEGORICAL_FEATURES + cfg.NUMERIC_FEATURES
SEEDS = [42, 7, 123, 2024, 31415, 8675309, 271828, 90210, 555, 1001]


def run_one_seed(seed: int, scratch_dir: Path) -> dict:
    cfg.SEED = seed

    df = generate_dataset(seed=seed)
    train_df, _cal_df, _test_df = make_splits(df, output_dir=scratch_dir / str(seed))

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[cfg.TARGET_COL]

    rf_result = rf_model.fit_tuned(X_train, y_train)
    xgb_result = xgb_model.fit_tuned(X_train, y_train)

    out = {"seed": seed}
    for result in (rf_result, xgb_result):
        out[result.name] = {
            "best_params": result.best_params,
            "single_fit_time_sec": result.single_fit_time_sec,
            "total_tuning_time_sec": result.total_tuning_time_sec,
            "n_fits": result.n_fits,
        }
    return out


def main():
    results = []
    with tempfile.TemporaryDirectory(prefix="seed_sweep_hparams_") as tmp:
        scratch_dir = Path(tmp)
        for seed in SEEDS:
            print(f"=== seed {seed} ===")
            r = run_one_seed(seed, scratch_dir)
            for name in ("random_forest", "xgboost"):
                print(f"  {name}: {r[name]['best_params']} "
                      f"single_fit={r[name]['single_fit_time_sec']:.3f}s "
                      f"total_tuning={r[name]['total_tuning_time_sec']:.3f}s")
            results.append(r)

    cfg.SEED = 42

    out_path = (
        Path(__file__).resolve().parent.parent
        / "outputs" / "metrics" / "seed_sweep_hparams.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw per-seed hyperparameter/timing results written to {out_path}")


if __name__ == "__main__":
    main()
