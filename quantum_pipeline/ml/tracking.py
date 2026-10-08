"""MLflow experiment tracking for the VQE ML models.

    from quantum_pipeline.ml.tracking import tracker

    with tracker.run('convergence_predictor', params={'horizon_k': 10}):
        tracker.log_metrics({'roc_auc': 0.87})

Set `MLFLOW_TRACKING_URI` to override the default server, or to `mlruns` for local
file-based tracking with no server.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

DEFAULT_TRACKING_URI = 'http://localhost:5000'


def get_tracking_uri() -> str:
    """Read `MLFLOW_TRACKING_URI` at call time, so it can be set after import."""
    return os.environ.get('MLFLOW_TRACKING_URI', DEFAULT_TRACKING_URI)


class ExperimentTracker:
    """Wrapper around MLflow with lazy initialization.

    MLflow is imported and configured only when tracking is first used, so
    importing this module does not require the optional MLflow dependency.
    """

    def __init__(self, tracking_uri: str | None = None) -> None:
        self._tracking_uri = tracking_uri or get_tracking_uri()
        self._mlflow: Any = None

    @property
    def mlflow(self) -> Any:
        """The mlflow module, imported and pointed at the tracking URI on first use."""
        if self._mlflow is None:
            try:
                import mlflow as _mlflow
            except ImportError as exc:
                raise ImportError(
                    'mlflow is required for experiment tracking. '
                    'Install it with: pdm install -G ml'
                ) from exc
            _mlflow.set_tracking_uri(self._tracking_uri)
            self._mlflow = _mlflow
        return self._mlflow

    @contextmanager
    def run(
        self,
        experiment: str,
        run_name: str | None = None,
        params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
    ):
        """Open an MLflow run under `experiment`, logging `params`; yields the run."""
        self.mlflow.set_experiment(experiment)
        with self.mlflow.start_run(run_name=run_name, tags=tags) as active_run:
            if params:
                self.mlflow.log_params(params)
            yield active_run

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.mlflow.log_metrics(metrics, step=step)


tracker = ExperimentTracker()
