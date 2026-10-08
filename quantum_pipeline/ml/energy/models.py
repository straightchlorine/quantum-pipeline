"""Regressor pipelines and evaluation metrics for the energy estimator."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from quantum_pipeline.ml.energy.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from quantum_pipeline.ml.preprocessing import build_preprocessor


def build_ridge(alpha: float = 1.0) -> Pipeline:
    return Pipeline(
        [
            ('prep', build_preprocessor(list(NUMERIC_FEATURES), list(CATEGORICAL_FEATURES))),
            ('model', Ridge(alpha=alpha)),
        ]
    )


def build_xgboost(**kwargs: Any) -> Any:
    """XGBoost regressor behind the same one-hot preprocessor as the Ridge pipeline."""
    from xgboost import XGBRegressor

    defaults: dict[str, Any] = {
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        # NOTE XGBoost: raise it once folds are large enough to pay for the threads
        # (for smaller ones thread coordination costs more than tree building itself)
        'n_jobs': 1,
        'verbosity': 0,
    }
    defaults.update(kwargs)
    return Pipeline(
        [
            ('prep', build_preprocessor(list(NUMERIC_FEATURES), list(CATEGORICAL_FEATURES))),
            ('model', XGBRegressor(**defaults)),
        ]
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Evaluate energy predictions using absolute, squared-error, and R^2 metrics."""

    # mean absolute prediction error (same unit as energy target)
    mae = float(mean_absolute_error(y_true, y_pred))

    # root mean squared error; penalizes large prediction errors
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # measure of how much target variation is explained relative to predicting
    # the mean for the target every time
    # unefined for single observation (no variation to explain)
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float('nan')

    return {'mae': mae, 'rmse': rmse, 'r2': r2}
