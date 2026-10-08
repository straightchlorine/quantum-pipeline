"""Shared preprocessing utilities for ML modules."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Build column transformer: scale numerics, one-hot categoricals."""
    transformers: list[tuple] = [
        (
            'num',
            # Pipeline fits every pipeline step on whatever rows it is handed.
            # In a LOMO fold the median is taken from the training molecules alone.
            #
            # Filling the frame up front would instead take one median across every
            # row - the held-out molecule included - and that number would then
            # appear in the data each fold trains on.
            #
            # The model would be learning from a value computed partly from
            # the runs it is about to be tested on.
            Pipeline(
                [
                    (
                        'impute',
                        # keep_empty_features: a column that is NaN for every run gets
                        # filled with 0;
                        # energy_delta_k1 is an example of this - step 1 has no
                        # predecessor to difference against
                        SimpleImputer(strategy='median', keep_empty_features=True),
                    ),
                    ('scale', StandardScaler()),
                ]
            ),
            numeric_features,
        ),
    ]
    if categorical_features:
        transformers.append(
            (
                'cat',
                OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                categorical_features,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder='drop')
