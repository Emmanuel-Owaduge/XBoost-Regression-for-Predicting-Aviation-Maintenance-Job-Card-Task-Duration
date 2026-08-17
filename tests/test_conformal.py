"""Tests for the finite-sample-corrected quantile index in
src/conformal/conformal.py: q_level = min(ceil((n+1)*(1-alpha))/n, 1.0).

A fake "pipeline" that always predicts 0 makes calibrate()'s residuals
(|y_cal - predict(X_cal)|) exactly equal to the y_cal values, so the chosen
residual list controls the quantile computation directly and precisely.
"""

import numpy as np
import pandas as pd

from src.conformal import conformal


class _ZeroPredictor:
    def predict(self, X):
        return np.zeros(len(X))


def test_quantile_index_hits_capped_max_for_small_n_at_90_percent():
    # n=5, alpha=0.10: q_level_raw = ceil(6*0.9)/5 = ceil(5.4)/5 = 6/5 = 1.2,
    # which is > 1.0 (and not a valid quantile level on its own) -- the
    # min(..., 1.0) cap in calibrate() is what keeps this from being an
    # invalid quantile call. Capped q_level=1.0 -> quantile is simply max().
    residuals = [3, 1, 4, 1, 5]
    y_cal = pd.Series(residuals, dtype=float)
    X_cal = pd.DataFrame({"dummy": range(len(residuals))})

    q_hat = conformal.calibrate(_ZeroPredictor(), X_cal, y_cal, alpha=0.10)

    assert q_hat == max(residuals)


def test_quantile_index_interior_case_hand_verified():
    # n=5, alpha=0.50: q_level = ceil(6*0.5)/5 = ceil(3.0)/5 = 3/5 = 0.6.
    # numpy's method="higher" virtual index = q_level*(n-1) = 0.6*4 = 2.4,
    # rounded up to index 3 (0-based) of the sorted array [10,20,30,40,50]
    # -> value 40. Row order is shuffled to confirm sorting happens correctly.
    residuals = [30, 10, 50, 20, 40]
    y_cal = pd.Series(residuals, dtype=float)
    X_cal = pd.DataFrame({"dummy": range(len(residuals))})

    q_hat = conformal.calibrate(_ZeroPredictor(), X_cal, y_cal, alpha=0.50)

    assert q_hat == 40.0
