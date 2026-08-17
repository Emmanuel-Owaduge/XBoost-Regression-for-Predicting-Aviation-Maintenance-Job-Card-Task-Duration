"""Tests for src/data/split.py: chronological ordering across the three splits."""

from src.data.generate import generate_dataset
from src.data.split import make_splits


def test_split_chronological_ordering(tmp_path):
    df = generate_dataset(n_records=300, seed=7)
    train_df, cal_df, test_df = make_splits(df, output_dir=tmp_path)

    assert train_df["job_date"].max() <= cal_df["job_date"].min()
    assert cal_df["job_date"].min() <= test_df["job_date"].min()
    # sanity: no records dropped or duplicated across the three splits
    assert len(train_df) + len(cal_df) + len(test_df) == len(df)
