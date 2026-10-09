"""Result containers for the energy estimator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class EvaluationResult:
    """Evaluation metrics for one model at one trajectory completion fraction."""

    model_name: str
    completion_frac: float
    mae: float
    rmse: float
    r2: float
    per_molecule: dict[str, dict[str, float]] = field(default_factory=dict)
    """Mean training-set size across LOMO folds."""
    n_train: int = 0
    """Total number of out-of-fold predictions across all held-out molecules."""
    n_test: int = 0

    def __str__(self) -> str:
        return (
            f'{self.model_name} @ {int(self.completion_frac * 100)}%: '
            f'MAE={self.mae:.4f} Ha  RMSE={self.rmse:.4f} Ha  R²={self.r2:.4f} '
            f'(n_train~{self.n_train}/fold, n_test={self.n_test})'
        )


@dataclass
class LomoPredictions:
    """Out-of-fold prediction from leave-one-molecule-out sweep.

    Each sample predicted once: in the fold where its molecule is held out from
    training. Thus, `actual` and each array in `predicted` contain predictions
    for the same evaluation set and are aligned by sample
    """

    actual: np.ndarray
    predicted: dict[str, np.ndarray]
    per_molecule_actual: dict[str, list[float]]
    per_molecule_predicted: dict[str, dict[str, list[float]]]
    train_sizes: list[int]

    @property
    def mean_train_size(self) -> int:
        """Return the mean training-set size across LOMO folds."""
        return int(np.mean(self.train_sizes)) if self.train_sizes else 0


@dataclass
class EnergyEstimatorResults:
    """Aggregated evaluation results across models and completion fractions."""

    results: list[EvaluationResult] = field(default_factory=list)

    # store models by key {model_name}_{completion_frac}
    fitted_models: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = ['Energy Estimator - LOMO Cross-Validation Summary', '=' * 60]
        lines.extend(
            str(r) for r in sorted(self.results, key=lambda x: (x.completion_frac, x.model_name))
        )
        return '\n'.join(lines)

    _HIGHER_IS_BETTER = frozenset({'r2'})

    def best(self, metric: str = 'mae') -> EvaluationResult | None:
        if not self.results:
            return None

        # mae and rmse: lower is better
        # R^2: higher is better
        cmp = max if metric in self._HIGHER_IS_BETTER else min
        return cmp(self.results, key=lambda r: getattr(r, metric))
