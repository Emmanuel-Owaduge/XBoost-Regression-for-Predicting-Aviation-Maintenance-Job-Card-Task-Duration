"""Hand-verified tests for src/evaluation/metrics.py."""

from src.evaluation import metrics


def test_rmse_mae_mape_hand_verified():
    y_true = [1, 2, 3, 4]
    y_pred = [1, 2, 3, 5]
    # errors = [0, 0, 0, -1]

    assert metrics.rmse(y_true, y_pred) == 0.5  # sqrt(mean([0,0,0,1])) = sqrt(0.25)
    assert metrics.mae(y_true, y_pred) == 0.25  # mean([0,0,0,1])
    assert metrics.mape(y_true, y_pred) == 6.25  # mean([0,0,0,0.25]) * 100


def test_empirical_coverage_hand_verified():
    y_true = [1, 2, 3]
    lower = [0, 2.5, 2]
    upper = [2, 3, 4]
    # record 0: 1 in [0,2] -> covered
    # record 1: 2 in [2.5,3] -> NOT covered
    # record 2: 3 in [2,4] -> covered

    assert metrics.empirical_coverage(y_true, lower, upper) == 2 / 3
