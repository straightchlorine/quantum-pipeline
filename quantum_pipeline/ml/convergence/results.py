"""Result containers for the convergence predictor: one per model, horizon and fold."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class FoldResult:
    """Evaluation metrics for one model at one horizon on one LOMO fold."""

    model_name: str
    horizon_k: int
    held_out_molecule: str
    roc_auc: float
    pr_auc: float
    brier_score: float
    mcc: float

    # number of samples used to train this fold's model
    n_train: int

    # number of samples in the held-out molecule for this fold
    n_test: int

    def __str__(self) -> str:
        return (
            f'{self.model_name} K={self.horizon_k} [{self.held_out_molecule} held out]: '
            f'ROC-AUC={self.roc_auc:.4f}  PR-AUC={self.pr_auc:.4f}  '
            f'Brier={self.brier_score:.4f}  MCC={self.mcc:.4f} '
            f'(n_train={self.n_train}, n_test={self.n_test})'
        )


@dataclass
class ConvergencePredictorResults:
    """Aggregated evaluation results across models, horizons, and LOMO folds."""

    fold_results: list[FoldResult] = field(default_factory=list)

    # store models by key {model_name}_{horizon}
    fitted_models: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        if not self.fold_results:
            return 'Convergence Predictor - No results (insufficient data or folds)'

        lines = ['Convergence Predictor - LOMO Cross-Validation Summary', '=' * 60]

        df = pd.DataFrame(
            [
                {
                    'model': r.model_name,
                    'horizon_k': r.horizon_k,
                    'roc_auc': r.roc_auc,
                    'pr_auc': r.pr_auc,
                    'brier_score': r.brier_score,
                    'mcc': r.mcc,
                }
                for r in self.fold_results
                # exclude folds where roc auc is undefined:
                # they contain only one class; other metrics are excluded with
                # the fold for consistency
                if np.isfinite(r.roc_auc)
            ]
        )

        if df.empty:
            lines.append('No finite metrics to display.')
            return '\n'.join(lines)

        # average metrics across lomo folds for each model and horizon
        grouped = df.groupby(['model', 'horizon_k'])
        agg = grouped[['roc_auc', 'pr_auc', 'brier_score', 'mcc']].mean()

        # Report what each mean is built from.
        # --
        # A single-class test fold scores NaN and was dropped above, so a mean
        # can rest on fewer folds than the run produced silently.
        agg.insert(0, 'folds', grouped.size())
        agg['of'] = self._folds_attempted(agg.index)

        lines.append(agg.round(4).to_string())

        dropped = len(self.fold_results) - len(df)
        if dropped:
            lines.append(
                f'\n{dropped} of {len(self.fold_results)} fold results dropped as '
                'single-class (undefined metrics).'
            )

        return '\n'.join(lines)

    def _folds_attempted(self, index: Any) -> list[int]:
        """Folds recorded per (model, horizon), including the single-class ones."""
        attempted: dict[tuple[str, int], int] = {}
        for r in self.fold_results:
            key = (r.model_name, r.horizon_k)
            attempted[key] = attempted.get(key, 0) + 1
        return [attempted.get(tuple(key), 0) for key in index]

    _LOWER_IS_BETTER = frozenset({'brier_score'})

    def best(self, metric: str = 'roc_auc') -> FoldResult | None:
        """Return the best finite fold according to the selected metric."""
        finite = [r for r in self.fold_results if np.isfinite(getattr(r, metric, float('nan')))]
        if not finite:
            return None

        # brier score: lower is better
        # roc auc, pr auc, mcc: higher is better
        cmp = min if metric in self._LOWER_IS_BETTER else max
        return cmp(finite, key=lambda r: getattr(r, metric))
