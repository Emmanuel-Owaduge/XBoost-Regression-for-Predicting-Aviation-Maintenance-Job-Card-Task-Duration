"""Accuracy / complexity / efficiency comparison table across all models."""

import pandas as pd

from src.models.common import ModelResult


def build_comparison_table(
    results: list[ModelResult], test_metrics: dict[str, dict]
) -> pd.DataFrame:
    rows = []
    for result in results:
        m = test_metrics[result.name]
        rows.append(
            {
                "model": result.name,
                "rmse": m["rmse"],
                "mae": m["mae"],
                "mape": m["mape"],
                "complexity": result.complexity,
                "single_fit_time_sec": result.single_fit_time_sec,
                "total_tuning_time_sec": result.total_tuning_time_sec,
            }
        )
    return pd.DataFrame(rows)
