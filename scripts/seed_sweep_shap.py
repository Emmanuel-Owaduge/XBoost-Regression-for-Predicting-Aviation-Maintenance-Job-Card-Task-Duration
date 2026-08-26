"""SHAP global-importance stability sweep, same 10 seeds as
scripts/seed_sweep.py.

compute_shap_values() returns per-column SHAP values in the one-hot-encoded
feature space (e.g. `categorical__task_type_airframe`, `numeric__staffing_ratio`
-- see tests/test_shap_analysis.py). The dissertation's global-importance
discussion is at the level of the 10 raw modeling features (4 categorical +
6 numeric), not individual one-hot columns, so this script aggregates each
categorical raw feature's dummy columns back together before ranking.

Aggregation method: for each row, sum the raw (signed) SHAP values across a
raw feature's one-hot columns to get that row's single contribution from the
raw feature (valid because exactly one dummy is "active" per row and SHAP's
local-accuracy property means contributions from a column group sum
linearly); then take mean(|per-row contribution|) across the test set. This
is the standard sum-then-abs-then-mean approach (summing abs per column
instead would overcount cases where dummy contributions partially offset).
Numeric features need no grouping -- one column each.

Same seed-binding fix as seed_sweep.py: pass seed explicitly to
generate_dataset, set cfg.SEED before fitting so XGB random_state follows.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as cfg
from src.data.generate import generate_dataset
from src.data.split import make_splits
from src.evaluation.shap_analysis import compute_shap_values
from src.models import xgboost_model as xgb_model

FEATURE_COLS = cfg.CATEGORICAL_FEATURES + cfg.NUMERIC_FEATURES
SEEDS = [42, 7, 123, 2024, 31415, 8675309, 271828, 90210, 555, 1001]


def _raw_feature_for_column(col_name: str) -> str:
    if col_name.startswith("numeric__"):
        return col_name[len("numeric__"):]
    assert col_name.startswith("categorical__")
    rest = col_name[len("categorical__"):]
    for feat in cfg.CATEGORICAL_FEATURES:
        if rest.startswith(feat + "_"):
            return feat
    raise ValueError(f"Could not map column {col_name!r} to a raw categorical feature")


def aggregate_to_raw_features(shap_values) -> dict:
    """Return {raw_feature_name: mean(|per-row summed contribution|)}."""
    groups: dict[str, list[int]] = {}
    for idx, col_name in enumerate(shap_values.feature_names):
        raw = _raw_feature_for_column(col_name)
        groups.setdefault(raw, []).append(idx)

    values = shap_values.values  # shape (n_rows, n_encoded_features)
    importance = {}
    for raw, col_idxs in groups.items():
        per_row_contribution = values[:, col_idxs].sum(axis=1)
        importance[raw] = float(np.mean(np.abs(per_row_contribution)))
    return importance


def run_one_seed(seed: int, scratch_dir: Path) -> dict:
    cfg.SEED = seed

    df = generate_dataset(seed=seed)
    train_df, _cal_df, test_df = make_splits(df, output_dir=scratch_dir / str(seed))

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[cfg.TARGET_COL]
    X_test = test_df[FEATURE_COLS]

    xgb_result = xgb_model.fit_tuned(X_train, y_train)
    shap_values = compute_shap_values(xgb_result.pipeline, X_test)
    importance = aggregate_to_raw_features(shap_values)

    ranking = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "seed": seed,
        "importance": importance,
        "ranking": [name for name, _ in ranking],
    }


def main():
    results = []
    with tempfile.TemporaryDirectory(prefix="seed_sweep_shap_") as tmp:
        scratch_dir = Path(tmp)
        for seed in SEEDS:
            print(f"=== seed {seed} ===")
            r = run_one_seed(seed, scratch_dir)
            for i, name in enumerate(r["ranking"], start=1):
                print(f"  {i}. {name}: {r['importance'][name]:.5f}")
            results.append(r)

    cfg.SEED = 42

    out_path = (
        Path(__file__).resolve().parent.parent
        / "outputs" / "metrics" / "seed_sweep_shap.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw per-seed SHAP ranking results written to {out_path}")


if __name__ == "__main__":
    main()
