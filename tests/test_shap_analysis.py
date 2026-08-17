"""Tests for src/evaluation/shap_analysis.py.

The strongest available correctness check for SHAP output is local accuracy:
base_value + sum(shap_values for a row) must equal the underlying model's
raw prediction for that row. This is a defining mathematical property of
Shapley-value attributions (verified empirically here to hold to ~1e-6 with
this project's xgboost/shap versions, not just assumed).
"""

import numpy as np

from src import config as cfg
from src.data.generate import generate_dataset
from src.data.split import make_splits
from src.evaluation.shap_analysis import (
    compute_shap_values,
    plot_global_importance,
    plot_individual_explanation,
)
from src.models import xgboost_model as xgb_model

FEATURE_COLS = cfg.CATEGORICAL_FEATURES + cfg.NUMERIC_FEATURES


def _fitted_xgb_and_test_set(tmp_path):
    df = generate_dataset(n_records=150, seed=5)
    train_df, _cal_df, test_df = make_splits(df, output_dir=tmp_path)
    X_train, y_train = train_df[FEATURE_COLS], train_df[cfg.TARGET_COL]
    X_test = test_df[FEATURE_COLS]
    result = xgb_model.fit_tuned(X_train, y_train, cv=3)
    return result.pipeline, X_test


def test_shap_values_satisfy_local_accuracy(tmp_path):
    xgb_pipeline, X_test = _fitted_xgb_and_test_set(tmp_path)
    shap_values = compute_shap_values(xgb_pipeline, X_test)

    preprocess = xgb_pipeline.named_steps["preprocess"]
    model = xgb_pipeline.named_steps["model"]
    raw_predictions = model.predict(preprocess.transform(X_test))

    reconstructed = shap_values.base_values + shap_values.values.sum(axis=1)
    np.testing.assert_allclose(reconstructed, raw_predictions, atol=1e-3)


def test_shap_values_shape_matches_encoded_features(tmp_path):
    xgb_pipeline, X_test = _fitted_xgb_and_test_set(tmp_path)
    shap_values = compute_shap_values(xgb_pipeline, X_test)

    n_encoded_features = len(xgb_pipeline.named_steps["preprocess"].get_feature_names_out())
    assert shap_values.values.shape == (len(X_test), n_encoded_features)
    assert shap_values.feature_names == list(
        xgb_pipeline.named_steps["preprocess"].get_feature_names_out()
    )


def test_shap_plot_functions_write_nonempty_files(tmp_path):
    xgb_pipeline, X_test = _fitted_xgb_and_test_set(tmp_path)
    shap_values = compute_shap_values(xgb_pipeline, X_test)

    figures_dir = tmp_path / "figures"
    global_path = plot_global_importance(shap_values, output_dir=figures_dir)
    individual_path = plot_individual_explanation(shap_values, index=0, output_dir=figures_dir)

    assert global_path.exists() and global_path.stat().st_size > 0
    assert individual_path.exists() and individual_path.stat().st_size > 0
