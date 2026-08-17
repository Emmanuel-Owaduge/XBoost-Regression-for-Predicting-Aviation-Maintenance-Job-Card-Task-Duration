"""Split conformal prediction on top of the fitted XGBoost pipeline.

Implements spec/architecture.md section 4 (`conformal/conformal.py`). Plain
(symmetric, constant-width) split conformal on absolute residuals — not
adaptive/CQR; see spec/architecture.md section 5 for that trade-off.
"""

import numpy as np
import pandas as pd

from src import config as cfg


def calibrate(
    xgb_pipeline, X_cal: pd.DataFrame, y_cal: pd.Series, alpha: float = cfg.CONFORMAL_ALPHA
) -> float:
    """Finite-sample-corrected empirical quantile of |y_cal - y_hat_cal|."""
    residuals = np.abs(y_cal.to_numpy() - xgb_pipeline.predict(X_cal))
    n = len(residuals)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(residuals, q_level, method="higher"))


def predict_with_interval(
    xgb_pipeline, X: pd.DataFrame, q_hat: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (y_hat, lower, upper) with symmetric width q_hat."""
    y_hat = xgb_pipeline.predict(X)
    lower = y_hat - q_hat
    upper = y_hat + q_hat
    return y_hat, lower, upper
