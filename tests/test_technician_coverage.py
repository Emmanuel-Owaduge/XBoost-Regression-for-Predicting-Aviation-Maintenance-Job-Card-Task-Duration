"""Hand-verified test for src/diagnostics/technician_coverage.py."""

import pandas as pd

from src.diagnostics.technician_coverage import summarize_low_coverage


def test_summarize_low_coverage_hand_verified():
    # technician 0: 6 records in train
    # technician 1: 2 records in train (below threshold=5, but nonzero)
    # technician 2: 0 records in train (its only record, id=100, is not in
    # train_df at all -- e.g. it landed in calibration/test)
    train_df = pd.DataFrame({"record_id": [0, 1, 2, 3, 4, 5, 6, 7]})
    debug_df = pd.DataFrame(
        {
            "record_id": [0, 1, 2, 3, 4, 5, 6, 7, 100],
            "technician_id": [0, 0, 0, 0, 0, 0, 1, 1, 2],
        }
    )

    summary = summarize_low_coverage(train_df, debug_df, n_technicians=3, threshold=5)

    assert summary == {
        "threshold": 5,
        "n_technicians": 3,
        "zero_records": 1,  # technician 2
        "below_threshold_nonzero": 1,  # technician 1 (count=2)
    }
