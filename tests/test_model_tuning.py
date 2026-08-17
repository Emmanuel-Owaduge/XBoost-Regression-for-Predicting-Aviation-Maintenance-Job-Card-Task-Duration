"""Tests for the grid-search wiring in random_forest.py and xgboost_model.py.

RF_PARAM_GRID/XGB_PARAM_GRID are read as `cfg.RF_PARAM_GRID` inside fit_tuned
(not bound as a default argument), so monkeypatching them here to a small
grid actually takes effect and keeps these tests fast — unlike cfg.N_RECORDS
or cfg.CV_FOLDS, which are bound as default argument values at module import
time and would silently ignore a post-import monkeypatch (verified
separately; see the test-skill report). `cv` is passed explicitly below for
the same reason, sidestepping that gotcha entirely.
"""

from src import config as cfg
from src.data.generate import generate_dataset
from src.models import random_forest as rf_model
from src.models import xgboost_model as xgb_model

FEATURE_COLS = cfg.CATEGORICAL_FEATURES + cfg.NUMERIC_FEATURES


def _train_data(n_records=120, seed=11):
    df = generate_dataset(n_records=n_records, seed=seed)
    return df[FEATURE_COLS], df[cfg.TARGET_COL]


def test_random_forest_fit_tuned_grid_search_bookkeeping(monkeypatch):
    small_grid = {
        "model__n_estimators": [10, 20],
        "model__max_depth": [None, 3],
        "model__min_samples_leaf": [1],
    }
    monkeypatch.setattr(cfg, "RF_PARAM_GRID", small_grid)
    X, y = _train_data()

    result = rf_model.fit_tuned(X, y, cv=3)

    n_candidates = 2 * 2 * 1  # product of small_grid's option counts
    assert result.name == "random_forest"
    assert result.n_fits == n_candidates * 3
    assert set(result.best_params.keys()) == set(small_grid.keys())
    assert result.best_params["model__n_estimators"] in small_grid["model__n_estimators"]
    assert result.complexity == float(result.best_params["model__n_estimators"])
    assert result.single_fit_time_sec > 0
    assert result.total_tuning_time_sec >= result.single_fit_time_sec

    preds = result.pipeline.predict(X)
    assert len(preds) == len(X)


def test_xgboost_fit_tuned_grid_search_bookkeeping(monkeypatch):
    small_grid = {
        "model__n_estimators": [10, 20],
        "model__max_depth": [2, 3],
        "model__learning_rate": [0.1],
    }
    monkeypatch.setattr(cfg, "XGB_PARAM_GRID", small_grid)
    X, y = _train_data()

    result = xgb_model.fit_tuned(X, y, cv=3)

    n_candidates = 2 * 2 * 1
    assert result.name == "xgboost"
    assert result.n_fits == n_candidates * 3
    assert set(result.best_params.keys()) == set(small_grid.keys())
    assert result.best_params["model__n_estimators"] in small_grid["model__n_estimators"]
    assert result.complexity == float(result.best_params["model__n_estimators"])
    assert result.single_fit_time_sec > 0
    assert result.total_tuning_time_sec >= result.single_fit_time_sec

    preds = result.pipeline.predict(X)
    assert len(preds) == len(X)
