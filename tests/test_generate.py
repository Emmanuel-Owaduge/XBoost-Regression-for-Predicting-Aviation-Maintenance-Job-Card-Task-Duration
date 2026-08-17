"""Tests for src/data/generate.py: causal feature computation and determinism.

generate.py's causal-window logic (which prior jobs feed rolling_efficiency_30d
and task_type_experience_count) is inline in the per-technician generation
loop, not factored into a separately callable function. To verify the
"strictly earlier job_date only" property, this test independently
reconstructs the same causal aggregates from the returned DataFrame (joined
against the debug DataFrame for technician_id, which is generation/diagnostic
scaffolding only — never a modeling feature) and asserts they match the
production output exactly. A mismatch here would mean the production code
either used a future job or missed a valid prior one.
"""

import numpy as np
import pandas as pd

from src.data.generate import generate_dataset


def test_causal_features_use_only_strictly_prior_history():
    df, debug_df = generate_dataset(n_records=200, seed=123, return_debug=True)
    merged = df.merge(debug_df[["record_id", "technician_id"]], on="record_id")

    checked = 0
    for _tech_id, group in merged.groupby("technician_id"):
        group_sorted = group.sort_values("job_date")
        seen = []  # (job_date, task_type, overrun_factor) for this technician only

        for _, row in group_sorted.iterrows():
            # Independent reconstruction using only jobs strictly before this one.
            prior = [h for h in seen if h[0] < row["job_date"]]
            recent = [h for h in prior if h[0] >= row["job_date"] - pd.Timedelta(days=30)]

            expected_experience_count = sum(1 for h in prior if h[1] == row["task_type"])
            expected_rolling = (
                round(float(np.mean([h[2] for h in recent])), 3) if recent else 1.0
            )

            assert row["task_type_experience_count"] == expected_experience_count, (
                f"record_id={row['record_id']}: task_type_experience_count "
                f"{row['task_type_experience_count']} != causally-recomputed "
                f"{expected_experience_count} (future-job leakage or missed prior job)"
            )
            assert abs(row["rolling_efficiency_30d"] - expected_rolling) < 1e-9, (
                f"record_id={row['record_id']}: rolling_efficiency_30d "
                f"{row['rolling_efficiency_30d']} != causally-recomputed {expected_rolling}"
            )

            seen.append((row["job_date"], row["task_type"], row["overrun_factor"]))
            checked += 1

    assert checked == len(df)  # every record was actually verified, not skipped


def test_generate_dataset_deterministic():
    df1 = generate_dataset(seed=42)
    df2 = generate_dataset(seed=42)
    pd.testing.assert_frame_equal(df1, df2)
