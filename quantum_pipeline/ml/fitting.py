"""Training helpers shared by both predictors: fold splitting and guarded fitting."""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any

from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import LeaveOneGroupOut

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np
    import pandas as pd

logger = logging.getLogger(__name__)


def fit_checked(model: Any, X: pd.DataFrame, y: np.ndarray, label: str) -> None:
    """Fit `model`, reporting a failure to converge as one log line naming `label`.

    Raised by the iterative solvers only - lbfgs in LogisticRegression, and Ridge's
    non-exact solvers.

    Suppressing the warning hides coefficients that never settled behind a decent score.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(X, y)

    if any(issubclass(c.category, ConvergenceWarning) for c in caught):
        logger.warning('%s did not converge; coefficients are unsettled', label)


def lomo_folds(
    df: pd.DataFrame,
    group_column: str = 'molecule_name',
    min_train: int = 5,
    min_test: int = 1,
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Yield `(held_out_group, train_positions, test_positions)` per LOMO fold.

    Leave-one-molecule-out: each group is held out once, so a run is never trained
    alongside another run of the same molecule. Folds too small to learn from or to
    score are skipped rather than returned, which is why callers see fewer folds than
    groups on sparse data.
    """
    groups = df[group_column].to_numpy()
    for train_idx, test_idx in LeaveOneGroupOut().split(df, groups=groups):
        if len(train_idx) < min_train or len(test_idx) < min_test:
            continue
        # the test side of a leave-one-group-out split is exactly one group
        yield str(groups[test_idx[0]]), train_idx, test_idx
