"""SHAP global feature importance + individual prediction explanation for XGBoost."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap

from src import config as cfg


def compute_shap_values(xgb_pipeline, X_test: pd.DataFrame) -> shap.Explanation:
    preprocess = xgb_pipeline.named_steps["preprocess"]
    model = xgb_pipeline.named_steps["model"]

    X_transformed = preprocess.transform(X_test)
    feature_names = list(preprocess.get_feature_names_out())

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_transformed)
    shap_values.feature_names = feature_names
    return shap_values


def plot_global_importance(shap_values, output_dir: Path = cfg.FIGURES_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "shap_global_importance.png"
    plt.figure()
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def plot_individual_explanation(
    shap_values, index: int, output_dir: Path = cfg.FIGURES_DIR
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"shap_individual_{index}.png"
    plt.figure()
    shap.plots.waterfall(shap_values[index], show=False)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path
