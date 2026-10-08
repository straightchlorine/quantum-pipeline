"""Classifier pipelines and evaluation metrics for convergence prediction."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from quantum_pipeline.ml.convergence.features import INTERACTION_FEATURES
from quantum_pipeline.ml.preprocessing import build_preprocessor


def build_xgboost_classifier(
    numeric_features: list[str],
    categorical_features: list[str],
    pos_weight: float = 1.0,
    **kwargs: Any,
) -> Pipeline:
    from xgboost import XGBClassifier

    defaults: dict[str, Any] = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'device': 'cpu',
        'n_estimators': 300,
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        # in case convergence class is underrepresented, give more weight to its
        # examples - in order to reduce model's tendency to favour non-convergence
        'scale_pos_weight': pos_weight,
        'random_state': 42,
        # NOTE XGBoost: raise it once folds are large enough to pay for the threads
        # (for smaller ones thread coordination costs more than tree building itself)
        'n_jobs': 1,
        'verbosity': 0,
    }

    # allow overrrides
    defaults.update(kwargs)
    return Pipeline(
        [
            # keep preprocessing within the pipeline (only on training data)
            ('prep', build_preprocessor(numeric_features, categorical_features)),
            ('model', XGBClassifier(**defaults)),
        ]
    )


def build_random_forest_classifier(
    numeric_features: list[str],
    categorical_features: list[str],
    **kwargs: Any,
) -> Pipeline:
    defaults: dict[str, Any] = {
        'n_estimators': 200,
        'max_depth': None,
        'min_samples_split': 5,
        # give more weight to minority class - in case convergence is underrepresented
        'class_weight': 'balanced',
        'random_state': 42,
        # enable paralel training (uses all cores)
        'n_jobs': -1,
    }
    defaults.update(kwargs)
    return Pipeline(
        [
            # keep preprocessing within the pipeline (only on training data)
            ('prep', build_preprocessor(numeric_features, categorical_features)),
            ('model', RandomForestClassifier(**defaults)),
        ]
    )


def build_logistic_regression_classifier(
    numeric_features: list[str],
    categorical_features: list[str],
    k: int,
    **kwargs: Any,
) -> Pipeline:
    """Build a Logistic Regression baseline with selected pairwise interactions.

    Interactions are limited to the:
        - horizon energy slope,
        - moving-energy standard deviation,
        - qubit count,
        - random-init flag.

    This let's the model express relationships between those features.

    For example, an energy slope may have different relationships with convergence for
    small and large problems, which can be represented by slope * qubit-count interaction.
    Purely linear model would assume that this relationship is the same regardless
    of the problem size.

    Remaining features are included as scaled linear terms.
    """

    # slope depends on the configured horizon
    slope_feat = f'energy_slope_first{k}'

    # create interaction for the features present in the current feature set
    # (and are selected)
    poly_features = [f for f in [slope_feat, *INTERACTION_FEATURES] if f in numeric_features]

    # all other metrics are linear terms, excluded from interaction expansion
    other_numeric = [f for f in numeric_features if f not in poly_features]

    transformers: list[tuple] = []
    if poly_features:
        transformers.append(
            (
                'poly_scaled',
                Pipeline(
                    [
                        # impute first: PolynomialFeatures cannot multiply a NaN
                        ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                        # standardize the inputs before forming interactions
                        # features should be comparable for the model
                        ('scaler', StandardScaler()),
                        (
                            'poly',
                            PolynomialFeatures(
                                degree=2,
                                include_bias=False,
                                # generate pairwise products, but not squared terms
                                interaction_only=True,
                            ),
                        ),
                    ]
                ),
                poly_features,
            )
        )

    # standardize other numeric features, so the scales are comparable
    if other_numeric:
        transformers.append(
            (
                'num',
                Pipeline(
                    [
                        ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                        ('scale', StandardScaler()),
                    ]
                ),
                other_numeric,
            )
        )
    if categorical_features:
        transformers.append(
            (
                'cat',
                OneHotEncoder(
                    # given data may contain a category not seen during training
                    # ignore it
                    handle_unknown='ignore',
                    sparse_output=False,
                ),
                categorical_features,
            )
        )

    # apply preprocessing appropriate to the selected groups of columns
    # discard others
    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

    logreg_defaults: dict[str, Any] = {
        'C': 1.0,
        # compensate for class imbalance (more weight to minority)
        'class_weight': 'balanced',
        'solver': 'lbfgs',
        # enough iterations to converge after feature expansion and encoding
        'max_iter': 1000,
        'random_state': 42,
    }
    logreg_defaults.update(kwargs)

    return Pipeline(
        [
            ('prep', preprocessor),
            ('model', LogisticRegression(**logreg_defaults)),
        ]
    )


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> dict[str, float]:
    """Evaluate classifier probabilities and threshold predictions.

    Returns:
        ROC AUC, average precision, Brier score and Matthews correlation coefficient.
        All four are NaN when the held-out fold contains only one class.
    """
    # roc auc is undefined when the held-out fold contains one class;
    # reporting NaN for all metrics in that fold for consistency
    if len(np.unique(y_true)) < 2:
        return {
            'roc_auc': float('nan'),
            'pr_auc': float('nan'),
            'brier_score': float('nan'),
            'mcc': float('nan'),
        }

    # matthews_corrcoef is calculated from binary predictions;
    # probabilities need to be converted to labels - 0.5 is used as threshold
    y_pred_labels = (y_pred_proba >= 0.5).astype(int)

    return {
        # measures how well positive examples are ranked above negative ones
        # across all classification thresholds
        'roc_auc': float(roc_auc_score(y_true, y_pred_proba)),
        # average_precision_score
        # summarizes the precision-recal curve across classification thresholds:
        # how many actual positives are found (recall)
        # against
        # how often positive predictions are correct (precision)
        'pr_auc': float(average_precision_score(y_true, y_pred_proba)),
        # measures the accuracy of the predicted probabilities;
        # lower values indicate probabilities closer to the observed binary outcomes
        'brier_score': float(brier_score_loss(y_true, y_pred_proba)),
        # evaluates the threshold predictions using confusion-matrix
        'mcc': float(matthews_corrcoef(y_true, y_pred_labels)),
    }
