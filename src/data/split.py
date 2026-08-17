"""Chronological train/calibration/test split.

Implements spec/architecture.md section 4 (`data/split.py`). Records are
sorted by job_date and sliced by position — earliest ratios[0] fraction as
train, next ratios[1] fraction as calibration, remainder as test. No
shuffling, no stratification: chronological order is the point (rules out
any test record chronologically preceding a training record).
"""

from pathlib import Path

import pandas as pd

from src import config as cfg


def make_splits(
    df: pd.DataFrame,
    date_col: str = cfg.DATE_COL,
    ratios: tuple = cfg.SPLIT_RATIOS,
    output_dir: Path = cfg.DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert abs(sum(ratios) - 1.0) < 1e-9, "split ratios must sum to 1"

    df_sorted = df.sort_values(date_col, kind="mergesort").reset_index(drop=True)
    n = len(df_sorted)
    n_train = int(round(n * ratios[0]))
    n_cal = int(round(n * ratios[1]))

    train_df = df_sorted.iloc[:n_train].copy()
    cal_df = df_sorted.iloc[n_train : n_train + n_cal].copy()
    test_df = df_sorted.iloc[n_train + n_cal :].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    cal_df.to_csv(output_dir / "calibration.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    return train_df, cal_df, test_df
