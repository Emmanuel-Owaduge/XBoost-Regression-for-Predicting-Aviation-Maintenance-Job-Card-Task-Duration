"""Shared feature encoder used by all three models.

Implements spec/architecture.md section 4 (`features/preprocessing.py`).
One shared one-hot categorical encoding across all models (fair comparison,
at the cost of not using XGBoost's native categorical handling — see
spec/architecture.md section 5). `scale_numeric=True` adds StandardScaler
for Linear Regression; tree models don't need it.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config as cfg


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_transformer = (
        Pipeline([("scaler", StandardScaler())]) if scale_numeric else "passthrough"
    )
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cfg.CATEGORICAL_FEATURES,
            ),
            ("numeric", numeric_transformer, cfg.NUMERIC_FEATURES),
        ]
    )
