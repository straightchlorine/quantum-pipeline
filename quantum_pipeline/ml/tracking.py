"""
MLflow experiment tracking helpers for VQE ML model development.

Usage (in notebooks or training scripts):
    from quantum_pipeline.ml.tracking import tracker

    with tracker.run("convergence_predictor", params={"n_estimators": 100}):
        model.fit(X_train, y_train)
        tracker.log_metrics({"roc_auc": roc_auc, "accuracy": acc})
        tracker.log_model(model, "xgboost_model")

Environment variables:
    MLFLOW_TRACKING_URI - override default tracking URI (default: http://localhost:5000)
                           Use "mlruns" for local file-based tracking (no server required).
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import mlflow

_DEFAULT_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", _DEFAULT_TRACKING_URI)


class ExperimentTracker:
    """Thin wrapper around MLflow for VQE ML experiments.

    Avoids importing mlflow at module level so that the quantum_pipeline package
    can be imported without the ml extras installed.
    """

    def __init__(self, tracking_uri: str | None = None) -> None:
        self._tracking_uri = tracking_uri or get_tracking_uri()
        self._mlflow: "mlflow | None" = None

    # ------------------------------------------------------------------
    # Lazy MLflow import
    # ------------------------------------------------------------------

    @property
    def mlflow(self) -> "mlflow":
        if self._mlflow is None:
            try:
                import mlflow as _mlflow

                _mlflow.set_tracking_uri(self._tracking_uri)
                self._mlflow = _mlflow
            except ImportError as exc:
                raise ImportError(
                    "mlflow is required for experiment tracking. "
                    "Install it with: pdm install -G ml"
                ) from exc
        return self._mlflow

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def set_experiment(self, name: str) -> None:
        """Create or set the active experiment by name."""
        self.mlflow.set_experiment(name)

    @contextmanager
    def run(
        self,
        experiment: str,
        run_name: str | None = None,
        params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
    ):
        """Context manager that starts/ends an MLflow run.

        Args:
            experiment: Experiment name (created if it does not exist).
            run_name:   Optional human-readable run label.
            params:     Hyperparameters to log at run start.
            tags:       Key-value tags attached to the run.

        Yields:
            The active ``mlflow.ActiveRun`` handle.
        """
        self.set_experiment(experiment)
        with self.mlflow.start_run(run_name=run_name, tags=tags) as active_run:
            if params:
                self.mlflow.log_params(params)
            yield active_run

    def log_params(self, params: dict[str, Any]) -> None:
        self.mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.mlflow.log_metrics(metrics, step=step)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        self.mlflow.log_metric(key, value, step=step)

    def log_model(self, model: Any, artifact_path: str, **kwargs: Any) -> None:
        """Log a scikit-learn or XGBoost model as an MLflow artifact.

        Detects the flavour automatically based on the model type.
        Falls back to ``mlflow.sklearn`` for unknown types.
        """
        try:
            import xgboost  # noqa: F401

            if isinstance(model, xgboost.XGBModel):
                self.mlflow.xgboost.log_model(model, artifact_path, **kwargs)
                return
        except ImportError:
            pass

        self.mlflow.sklearn.log_model(model, artifact_path, **kwargs)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Log a local file (e.g. a plot or CSV) as an MLflow artifact."""
        self.mlflow.log_artifact(local_path, artifact_path=artifact_path)

    def log_figure(self, figure: Any, filename: str) -> None:
        """Log a matplotlib figure as an artifact."""
        self.mlflow.log_figure(figure, filename)

    # ------------------------------------------------------------------
    # Convenience: standard VQE ML metric sets
    # ------------------------------------------------------------------

    def log_classification_metrics(
        self,
        *,
        roc_auc: float | None = None,
        accuracy: float | None = None,
        f1: float | None = None,
        precision: float | None = None,
        recall: float | None = None,
        step: int | None = None,
    ) -> None:
        """Log the standard convergence-predictor evaluation metrics."""
        metrics = {
            k: v
            for k, v in {
                "roc_auc": roc_auc,
                "accuracy": accuracy,
                "f1": f1,
                "precision": precision,
                "recall": recall,
            }.items()
            if v is not None
        }
        if metrics:
            self.log_metrics(metrics, step=step)

    def log_regression_metrics(
        self,
        *,
        mse: float | None = None,
        mae: float | None = None,
        r2: float | None = None,
        step: int | None = None,
    ) -> None:
        """Log the standard energy-estimator evaluation metrics."""
        metrics = {
            k: v
            for k, v in {
                "mse": mse,
                "mae": mae,
                "r2": r2,
            }.items()
            if v is not None
        }
        if metrics:
            self.log_metrics(metrics, step=step)


# ---------------------------------------------------------------------------
# Module-level singleton - import and use directly in notebooks/scripts
# ---------------------------------------------------------------------------

tracker = ExperimentTracker()
