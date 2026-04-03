"""Shared preprocessing utilities for ML modules."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Build column transformer: scale numerics, one-hot categoricals."""
    transformers: list[tuple] = [
        ('num', StandardScaler(), numeric_features),
    ]
    if categorical_features:
        transformers.append((
            'cat',
            OneHotEncoder(handle_unknown='ignore', sparse_output=False),
            categorical_features,
        ))
    return ColumnTransformer(transformers=transformers, remainder='drop')
