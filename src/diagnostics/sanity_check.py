"""Signal-strength sanity check (spec task 4).

Implements spec/architecture.md section 4 (`diagnostics/sanity_check.py`).
Reuses the Linear Regression pipeline as the diagnostic probe rather than
building a separate throwaway model — its result is reused directly as the
task-5 Linear Regression baseline when the check passes (see
spec/architecture.md section 5).

Both RMSEs below are computed in-sample on the training partition: this is
a quick diagnostic of how much predictive content the features carry
relative to the target, not a generalization check.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src import config as cfg
from src.models import linear as linear_model
from src.models.common import ModelResult

FEATURE_COLS = cfg.CATEGORICAL_FEATURES + cfg.NUMERIC_FEATURES

Verdict = Literal["ok", "noise_too_low", "noise_too_high"]


@dataclass
class SanityCheckResult:
    mean_baseline_rmse: float
    probe_rmse: float
    ratio: float  # probe_rmse / mean_baseline_rmse
    verdict: Verdict
    probe_result: ModelResult
    # mean(y_true - y_pred) on the raw overrun_factor scale, in-sample on
    # train. The probe (Linear Regression) fits log(target) and exponentiates
    # predictions back (see models/linear.py) — by Jensen's inequality that
    # round-trip is not unbiased in the raw scale, so this checks whether a
    # systematic over/under-prediction bias shows up in practice.
    probe_mean_signed_residual: float


def run_sanity_check(train_df: pd.DataFrame) -> SanityCheckResult:
    X_train = train_df[FEATURE_COLS]
    y_train = train_df[cfg.TARGET_COL]

    mean_pred = np.full(len(y_train), y_train.mean())
    mean_baseline_rmse = float(np.sqrt(np.mean((y_train.to_numpy() - mean_pred) ** 2)))

    probe_result = linear_model.fit_tuned(X_train, y_train)
    probe_pred = probe_result.pipeline.predict(X_train)
    residuals = y_train.to_numpy() - probe_pred
    probe_rmse = float(np.sqrt(np.mean(residuals**2)))
    probe_mean_signed_residual = float(np.mean(residuals))

    ratio = probe_rmse / mean_baseline_rmse

    if ratio < cfg.SANITY_CHECK_MIN_RATIO:
        verdict: Verdict = "noise_too_low"
    elif ratio > cfg.SANITY_CHECK_MAX_RATIO:
        verdict = "noise_too_high"
    else:
        verdict = "ok"

    return SanityCheckResult(
        mean_baseline_rmse=mean_baseline_rmse,
        probe_rmse=probe_rmse,
        ratio=ratio,
        verdict=verdict,
        probe_result=probe_result,
        probe_mean_signed_residual=probe_mean_signed_residual,
    )
