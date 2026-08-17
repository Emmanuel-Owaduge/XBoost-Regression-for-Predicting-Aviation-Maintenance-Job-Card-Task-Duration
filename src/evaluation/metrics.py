"""Point-estimate error metrics + prediction-interval coverage."""

import numpy as np


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred) -> float:
    """Percent MAPE. Note: unstable as y_true approaches zero — mitigated
    here by the OVERRUN_FLOOR clip applied during data generation."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)


def empirical_coverage(y_true, lower, upper) -> float:
    y_true = np.asarray(y_true)
    lower, upper = np.asarray(lower), np.asarray(upper)
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))
