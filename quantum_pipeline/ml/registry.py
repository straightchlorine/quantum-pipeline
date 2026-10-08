"""Canonical model identities, shared by both predictors.

Every place a model is named by string reads this mapping.
"""

from __future__ import annotations

MODEL_KEYS = {
    'XGBoost': 'xgboost',
    'RandomForest': 'random_forest',
    'LogisticRegression': 'logistic_regression',
    'Ridge': 'ridge',
}
