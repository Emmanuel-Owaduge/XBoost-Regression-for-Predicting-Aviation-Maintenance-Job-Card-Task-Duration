"""Linear Regression: interpretable baseline, no hyperparameter search.

Fit on log(overrun_factor) rather than the raw target. The target is
generated as a product of multiplicative components (skill_effect x
learning_curve x log-normal noise — see data/generate.py), so a log
transform puts OLS in the space where its linearity assumption actually
holds. Random Forest and XGBoost stay on the raw scale (see
random_forest.py, xgboost_model.py) — tree splits don't carry that
distributional assumption, so they don't need the same fix.

Naively exponentiating log-space predictions back to the raw scale is not
unbiased (Jensen's inequality) — confirmed empirically via a positive mean
signed residual on train. Corrected here with Duan's smearing estimator:
smearing_factor = mean(exp(log-scale training residuals)); raw-scale
predictions = exp(log_pred) * smearing_factor. `SmearedLogLinearModel`
wraps this so `.predict()` still returns raw-scale predictions transparently
for every downstream caller (sanity_check.py, pipeline.py).
"""

import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from src import config as cfg
from src.features.preprocessing import build_preprocessor
from src.models.common import ModelResult


class SmearedLogLinearModel:
    """`.predict(X)` = exp(log_pipeline.predict(X)) * smearing_factor."""

    def __init__(self, log_pipeline: Pipeline, smearing_factor: float):
        self.log_pipeline = log_pipeline
        self.smearing_factor = smearing_factor

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.exp(self.log_pipeline.predict(X)) * self.smearing_factor


def fit_tuned(X_train: pd.DataFrame, y_train: pd.Series, cv: int = cfg.CV_FOLDS) -> ModelResult:
    """`cv` is accepted for interface consistency with random_forest/xgboost_model
    but unused — Linear Regression has no hyperparameters to tune per spec."""
    log_pipeline = Pipeline(
        [
            ("preprocess", build_preprocessor(scale_numeric=True)),
            ("model", LinearRegression()),
        ]
    )
    y_train_log = np.log(y_train.to_numpy())

    start = time.perf_counter()
    log_pipeline.fit(X_train, y_train_log)
    log_pred_train = log_pipeline.predict(X_train)
    smearing_factor = float(np.mean(np.exp(y_train_log - log_pred_train)))
    fit_time = time.perf_counter() - start

    model = SmearedLogLinearModel(log_pipeline, smearing_factor)
    n_coefficients = log_pipeline.named_steps["model"].coef_.shape[0]

    return ModelResult(
        name="linear_regression",
        pipeline=model,
        best_params={"smearing_factor": smearing_factor},
        single_fit_time_sec=fit_time,
        total_tuning_time_sec=fit_time,
        n_fits=1,
        complexity=float(n_coefficients),
    )
