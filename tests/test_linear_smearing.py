"""Tests for the Duan smearing factor in src/models/linear.py.

The smearing computation (`smearing_factor = mean(exp(log-scale training
residuals))`) is inline inside fit_tuned, not a separately callable
function. Two complementary tests, since neither alone gives full coverage
without modifying production code:

1. A formula-replica test: hand-picked log-residuals, checked against the
   exact expression used in linear.py (line: `float(np.mean(np.exp(
   y_train_log - log_pred_train)))`). This verifies the arithmetic is
   correct, but does not exercise fit_tuned's real code path — it is a
   golden-value check on a duplicated expression, not a black-box test of
   the shipped function.

2. An integration test through the real fit_tuned(): OLS with more encoded
   parameters (one-hot categoricals + numeric) than training rows is
   underdetermined and generically achieves exact-zero training residual
   (verified empirically, not assumed) — so the hand-computable expected
   smearing factor is exactly 1.0 (mean(exp(0)) = 1). This exercises the
   real function but only for the residuals-are-all-zero case, not
   arbitrary hand-picked nonzero residuals — see the note in the test-skill
   report about why arbitrary nonzero residuals can't be forced through a
   real OLS-with-intercept fit (they'd have to sum to zero).
"""

import numpy as np
import pandas as pd
import pytest

from src.models import linear as linear_model


def test_smearing_factor_formula_matches_hand_calculation():
    # Hand-picked log-scale residuals: 0, ln(2), -ln(2), ln(4)
    # exp(residuals) = [1, 2, 0.5, 4] -> mean = 7.5 / 4 = 1.875
    residuals = np.array([0.0, np.log(2), -np.log(2), np.log(4)])
    expected = 1.875

    # Exact expression from linear.py's fit_tuned (y_train_log - log_pred_train
    # replaced directly by the hand-picked residuals here).
    smearing_factor = float(np.mean(np.exp(residuals)))

    assert smearing_factor == expected


def test_smearing_factor_is_one_when_ols_achieves_zero_residual():
    # 3 training rows, but one-hot encoding 4 categorical columns (each row
    # given a distinct category) plus 6 numeric columns yields far more
    # encoded parameters than rows -> the underlying least-squares problem
    # is underdetermined and generically fits every training point exactly.
    X_train = pd.DataFrame(
        {
            "task_type": ["engine", "avionics", "airframe"],
            "task_category": ["scheduled", "unscheduled", "inspection"],
            "shift_type": ["day", "evening", "night"],
            "day_of_week": ["Monday", "Tuesday", "Wednesday"],
            "estimated_duration_hours": [8.0, 4.0, 6.0],
            "rolling_efficiency_30d": [1.0, 1.1, 0.9],
            "task_type_experience_count": [0, 1, 2],
            "learning_curve_index": [0.0, 0.5, 1.0],
            "staffing_ratio": [0.8, 0.9, 1.0],
            "attendance_rate": [0.9, 0.95, 1.0],
        }
    )
    y_train = pd.Series([1.2, 0.9, 1.5])

    result = linear_model.fit_tuned(X_train, y_train)

    # Confirm the premise (exact fit) before trusting the derived expectation.
    preds = result.pipeline.log_pipeline.predict(X_train)
    np.testing.assert_allclose(preds, np.log(y_train.to_numpy()), atol=1e-9)

    assert result.best_params["smearing_factor"] == pytest.approx(1.0, abs=1e-9)
