"""Shared result type for all three models.

Implements spec/architecture.md section 4 (`models/common.py`). Every
downstream stage (sanity check, conformal, evaluation) depends only on this
type — none of them need to know whether the underlying estimator is
LinearRegression, RandomForestRegressor, or XGBRegressor, or whether it
wraps a target transform (see models/linear.py).
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class ModelResult:
    name: str
    # Fitted, .predict()-ready object. A plain Pipeline for RF/XGBoost; a
    # SmearedLogLinearModel wrapping a Pipeline for Linear Regression
    # (log-target fit, predictions exponentiated + Duan-smearing-corrected
    # back to raw scale — see models/linear.py). Not always a sklearn
    # BaseEstimator, so typed loosely; every caller only relies on .predict().
    pipeline: Any
    best_params: dict
    single_fit_time_sec: float  # time of one representative fit
    total_tuning_time_sec: float  # full grid-search wall time (== single_fit_time for linear)
    n_fits: int  # 1 for linear; n_candidates * cv_folds otherwise
    complexity: float  # LR: #coefficients; RF/XGBoost: n_estimators (proxy, see spec)
