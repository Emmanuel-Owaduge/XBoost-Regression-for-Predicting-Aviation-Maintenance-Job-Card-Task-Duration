"""Technician coverage diagnostic for the training partition.

Quantifies the "chronological split can create technician cohort imbalance"
risk documented in spec/project-specification.md: technicians who only
begin appearing late in the timeline are disproportionately pushed into
calibration/test and may end up with little or no training history.

technician_id is never a modeling feature (see data/generate.py) — this
module only uses it as a diagnostic join key, via the debug DataFrame
returned by `generate_dataset(..., return_debug=True)`.
"""

import pandas as pd

from src import config as cfg


def technician_train_counts(train_df: pd.DataFrame, debug_df: pd.DataFrame) -> pd.Series:
    """Per-technician record counts within the training partition."""
    merged = train_df[[cfg.ID_COL]].merge(
        debug_df[[cfg.ID_COL, "technician_id"]], on=cfg.ID_COL, how="left"
    )
    return merged.groupby("technician_id").size()


def summarize_low_coverage(
    train_df: pd.DataFrame,
    debug_df: pd.DataFrame,
    n_technicians: int = cfg.N_TECHNICIANS,
    threshold: int = 5,
) -> dict:
    """Counts technicians with 0 training records, and separately those with
    more than 0 but fewer than `threshold`."""
    counts = technician_train_counts(train_df, debug_df)
    present_ids = set(counts.index)
    all_ids = set(range(n_technicians))
    zero_record_ids = all_ids - present_ids
    below_threshold_nonzero = counts[(counts > 0) & (counts < threshold)]

    return {
        "threshold": threshold,
        "n_technicians": n_technicians,
        "zero_records": len(zero_record_ids),
        "below_threshold_nonzero": int(len(below_threshold_nonzero)),
    }
