"""Random Forest: secondary ensemble comparison, k-fold CV grid search."""

import time

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src import config as cfg
from src.features.preprocessing import build_preprocessor
from src.models.common import ModelResult


def fit_tuned(X_train: pd.DataFrame, y_train: pd.Series, cv: int = cfg.CV_FOLDS) -> ModelResult:
    pipeline = Pipeline(
        [
            ("preprocess", build_preprocessor(scale_numeric=False)),
            ("model", RandomForestRegressor(random_state=cfg.SEED)),
        ]
    )

    search = GridSearchCV(
        pipeline,
        cfg.RF_PARAM_GRID,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    start = time.perf_counter()
    search.fit(X_train, y_train)
    total_time = time.perf_counter() - start

    best_idx = search.best_index_
    single_fit_time = float(search.cv_results_["mean_fit_time"][best_idx])
    n_fits = len(search.cv_results_["params"]) * cv

    return ModelResult(
        name="random_forest",
        pipeline=search.best_estimator_,
        best_params=search.best_params_,
        single_fit_time_sec=single_fit_time,
        total_tuning_time_sec=total_time,
        n_fits=n_fits,
        complexity=float(search.best_params_["model__n_estimators"]),
    )
