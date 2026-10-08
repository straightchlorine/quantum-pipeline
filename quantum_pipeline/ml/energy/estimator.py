"""Leave-one-molecule-out (LOMO) training and evaluation for the energy estimator.

`EnergyEstimator` is the entry point: `fit_evaluate` scores Ridge and XGBoost on
trajectories truncated at each completion fraction, then refits both on everything so
`predict` can serve new runs.

The regression target is the `final_energy` column. It is defined differently on real
and on synthetic data. The difference changes how good the errors look.

On real data `final_energy` is `vqe_results.minimum_energy`, renamed by
`docker/airflow/scripts/quantum_ml_feature_processing.py`. That is the lowest energy the
solver sampled at any iteration (`VQEResult.minimum`, picked by `VQESolver._best_step`),
not the energy of the step it stopped on. Every sample carries noise, so the lowest one
is usually a sample where the noise happened to dip below the level the run had settled
at.

On synthetic data `final_energy` is `Descent.e_final` from `ml/trajectory.py`.
`simulate_descent` first chooses the level the curve will settle at, then draws noisy
samples around it. The target is that chosen level, with no noise in it at all.

So the real target sits a little below the true settled level. However far the noise
dipped on a flat tail - there is the synthetic target settled.

Hypothesis, not measured:

Model therefore scores better on synthetic data than on real data by about
that noise margin. To measure it, compare a real run's `final_energy` with the mean of
its last few sampled energies; the gap is the dip.
TODO: prove/disprove

Neither definition is a reference energy, so a low error here means "predicted the best
energy this run would reach", not "predicted the true ground state". And synthetic
trajectories pick their outcome before the curve is drawn (`simulate_descent` in
`ml/trajectory.py`), so a score on them exercises this code and says nothing about
whether the features predict real runs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from quantum_pipeline.ml.energy.features import (
    ALL_FEATURES,
    COMPLETION_FRACS,
    extract_features_at_fraction,
)
from quantum_pipeline.ml.energy.models import (
    build_ridge,
    build_xgboost,
    compute_metrics,
)
from quantum_pipeline.ml.energy.results import (
    EnergyEstimatorResults,
    EvaluationResult,
    LomoPredictions,
)
from quantum_pipeline.ml.fitting import fit_checked, lomo_folds
from quantum_pipeline.ml.registry import MODEL_KEYS
from quantum_pipeline.ml.schema import RUN_KEY

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class EnergyEstimator:
    """Train and evaluate XGBoost and Ridge energy estimators on VQE trajectories.

    Evaluation is leave-one-molecule-out, so a score reports performance on a molecule
    the model never saw.

    One model is fitted per completion fraction, not one model taking the fraction as
    input: a trajectory truncated at 25% and one at 75% are different problems.

    For the same reason the fraction is not a feature: it would be constant over each
    model's training set, so the `StandardScaler` would map it to 0.

    Args:
        completion_fracs: How much of each trajectory the model may read, as a
            fraction. Each gets its own fitted model, keyed by `float(frac)`, so `1`
            and `1.0` agree, but a fraction that differs by float rounding does not
            (see `predict`).
        ridge_alpha: Ridge penalty weight; larger values shrink the coefficients
            harder. Ridge is the linear model XGBoost is compared against.
        xgb_params: XGBoost hyperparameter overrides, merged over the defaults in
            `models.build_xgboost`.
        use_mlflow: Whether to log runs to MLflow via the tracker singleton.
        experiment_name: MLflow experiment name.
    """

    def __init__(
        self,
        completion_fracs: list[float] | None = None,
        ridge_alpha: float = 1.0,
        xgb_params: dict[str, Any] | None = None,
        use_mlflow: bool = False,
        experiment_name: str = 'energy_estimator',
    ) -> None:
        self.completion_fracs = completion_fracs or COMPLETION_FRACS
        self.ridge_alpha = ridge_alpha
        self.xgb_params = xgb_params or {}
        self.use_mlflow = use_mlflow
        self.experiment_name = experiment_name

        # Populated by fit_evaluate and read by predict.
        # --
        # Public, and the same object fit_evaluate returns, so any `EnergyEstimatorResults`
        # (e.g. one loaded from disk) can be assigned here to predict without retraining.
        # fitted_models is keyed f'{MODEL_KEYS[name]}_{float(frac)}', e.g. 'xgboost_0.5'.
        self.results = EnergyEstimatorResults()

    def _build_models(self) -> dict[str, Pipeline]:
        """Fresh unfitted pipelines, keyed by the display names `MODEL_KEYS` maps.

        Rebuilt per fold rather than refitted in place. sklearn `fit` relearns from
        scratch either way, so this is for a clean object, not for isolation.
        """
        return {
            'Ridge': build_ridge(alpha=self.ridge_alpha),
            'XGBoost': build_xgboost(**self.xgb_params),
        }

    def _run_lomo_cv(
        self,
        df_feat: pd.DataFrame,
        molecules: np.ndarray,
    ) -> LomoPredictions:
        """Hold out each molecule in turn and collect out-of-fold predictions.

        A run is predicted at most once, in the fold where its molecule was held
        out.
        Thus the accumulated `actual` and `predicted` arrays stay index-aligned and
        together form the whole evaluation set.
        Metrics are computed over that pooled set rather than averaged per fold,
        which is why a molecule with more runs carries more weight.

        "At most" because `lomo_folds` skips a fold whose training side has fewer
        than `min_train` (default 5) rows. The skipped molecule's runs are then
        never predicted, and its entry in `per_molecule_actual` stays the empty
        list it was seeded with below.

        Args:
            df_feat: One row per run, as returned by `extract_features_at_fraction`,
                with `final_energy` already non-null on every row.
            molecules: Every `molecule_name` in `df_feat`. Used only to seed the
                per-molecule dicts, so a skipped fold leaves a key with an empty list
                rather than no key.
        """
        actuals: list[float] = []
        train_sizes: list[int] = []

        # building the pipelines and settling the keys.
        model_names = tuple(self._build_models())
        preds: dict[str, list[float]] = {name: [] for name in model_names}
        per_mol_preds: dict[str, dict[str, list[float]]] = {
            name: {m: [] for m in molecules} for name in model_names
        }
        per_mol_actual: dict[str, list[float]] = {m: [] for m in molecules}

        # training and then predicting each run at most once
        for held_out_mol, train_idx, test_idx in lomo_folds(df_feat):
            df_train = df_feat.iloc[train_idx]
            df_test = df_feat.iloc[test_idx]

            X_train = df_train[ALL_FEATURES]
            y_train = df_train['final_energy'].to_numpy()
            X_test = df_test[ALL_FEATURES]
            y_test = df_test['final_energy'].to_numpy()

            # going over created pipelines and running train + prediction
            for name, model in self._build_models().items():
                fit_checked(model, X_train, y_train, f'{name} held-out {held_out_mol} fold')
                y_pred = model.predict(X_test)
                preds[name].extend(y_pred.tolist())
                per_mol_preds[name][held_out_mol] = y_pred.tolist()

            # keeping data for the results
            actuals.extend(y_test.tolist())
            per_mol_actual[held_out_mol] = y_test.tolist()
            train_sizes.append(len(df_train))

        return LomoPredictions(
            actual=np.array(actuals),
            predicted={name: np.array(vals) for name, vals in preds.items()},
            per_molecule_actual=per_mol_actual,
            per_molecule_predicted=per_mol_preds,
            train_sizes=train_sizes,
        )

    def fit_evaluate(self, df_traj: pd.DataFrame) -> EnergyEstimatorResults:
        """Run full LOMO cross-validation across all configured completion fractions.

        Per completion fraction:
            - build features,
            - run leave-one-molecule-out CV,
            - record per-molecule and pooled MAE / RMSE / R^2,
            - refit on everything for inference.

        A fraction whose data cannot support evaluation (no target, fewer than 10 runs,
        fewer than 2 molecules) is skipped - one thin fraction does not abandon the others.

        An invalid fraction or missing required columns raises `ValueError` from
        `extract_features_at_fraction`. If every fraction is skipped the returned
        object is empty, and `predict` then raises for every key.

        Args:
            df_traj: Iteration-level trajectory DataFrame. Required columns are checked
                by `extract_features_at_fraction`. For anything to be scored it also
                needs a target - `final_energy`, or a per-row `minimum_energy` the
                builder falls back to, taking its value at the truncation point - and
                at least two distinct `molecule_name` values. Without `molecule_name`
                the builder labels every run 'UNKNOWN', so once the 10-row guard below
                passes, the >=2-molecules guard skips every fraction with a message
                about molecule count.

        Returns:
            `EnergyEstimatorResults` with evaluation metrics and fitted models. Also
            assigned to `self.results`, which is what `predict` reads.
        """
        results = EnergyEstimatorResults()
        self.results = results

        for frac in self.completion_fracs:
            logger.info('Evaluating at %.0f%% trajectory completion ...', frac * 100)
            df_feat = extract_features_at_fraction(df_traj, frac)

            # No target means prediction data reached a training method.
            # --
            # The feature builder always emits `final_energy`, as None when the input
            # has neither `final_energy` nor `minimum_energy`; that all-None column is
            # the usual trigger here. The `not in columns` half only fires when the
            # input had no runs, since an empty frame has no columns at all.
            if 'final_energy' not in df_feat.columns or df_feat['final_energy'].isnull().all():
                logger.warning('No target column (final_energy) at frac=%.2f - skipping', frac)
                continue

            # Checking the amount of samples with the target. Runs without it
            # can neither train nor score.
            df_feat = df_feat.dropna(subset=['final_energy'])
            if len(df_feat) < 10:
                logger.warning('Too few samples (%d) at frac=%.2f - skipping', len(df_feat), frac)
                continue

            molecules = df_feat['molecule_name'].unique()
            if len(molecules) < 2:
                logger.warning(
                    'Need >=2 molecules for LOMO-CV (got %d) - skipping', len(molecules)
                )
                continue

            cv = self._run_lomo_cv(df_feat, molecules)

            # Cannot trigger while the 10-row and 2-molecule guards above hold.
            # --
            # The smallest molecule has at most n/2 runs, so its fold keeps at
            # least 5 training rows and `lomo_folds` yields it.
            # Kept so a change to those guards fails quietly here and not later
            # in compute_metrics.
            # In such case fraction would get no results and no fitted model -
            # prediction would raise.
            if len(cv.actual) == 0:
                continue

            metrics_by_model = {
                name: compute_metrics(cv.actual, y_pred) for name, y_pred in cv.predicted.items()
            }

            for name, metrics in metrics_by_model.items():
                per_mol = {
                    mol: compute_metrics(
                        np.array(cv.per_molecule_actual[mol]),
                        np.array(cv.per_molecule_predicted[name][mol]),
                    )
                    for mol in molecules
                    # a molecule whose fold lomo_folds skipped still has its seeded
                    # empty list, and mean_absolute_error raises on an empty array
                    if cv.per_molecule_actual[mol]
                }
                results.results.append(
                    EvaluationResult(
                        model_name=name,
                        completion_frac=frac,
                        mae=metrics['mae'],
                        rmse=metrics['rmse'],
                        r2=metrics['r2'],
                        per_molecule=per_mol,
                        # no run is predicted in more than one fold, so n_test is the
                        # whole pooled evaluation set; n_train is the per-fold mean,
                        # floored to an int
                        n_train=cv.mean_train_size,
                        n_test=len(cv.actual),
                    )
                )

            if self.use_mlflow:
                self._track_fraction(frac, metrics_by_model)

            # Refit on every molecule. Never scored - the folds above did that - only
            # for predict, which looks them up by key.
            X_all = df_feat[ALL_FEATURES]
            y_all = df_feat['final_energy'].values
            for name, model in self._build_models().items():
                fit_checked(model, X_all, y_all, f'{name} frac={frac} full refit')
                results.fitted_models[f'{MODEL_KEYS[name]}_{float(frac)}'] = model

        return results

    def predict(
        self,
        df_traj: pd.DataFrame,
        completion_frac: float = 0.5,
        model: str = MODEL_KEYS['XGBoost'],
    ) -> pd.Series:
        """Predict final energy for new runs at a given trajectory completion fraction.

        Args:
            df_traj: Trajectory data (same format as `fit_evaluate`). `final_energy`
                may be absent; nothing here reads it. The builder still emits the
                column, as None or as its `minimum_energy` fallback.
            completion_frac: Fraction of trajectory available. Must be one of
                `completion_fracs` that `fit_evaluate` did not skip; the `ValueError`
                message lists the keys that were actually fitted.
            model: 'xgboost' or 'ridge' (a `MODEL_KEYS` value). Matched
                case-insensitively, so the display names work too because both keys
                are single words.

        Returns:
            Series of predicted energies indexed by experiment_id.

        Raises:
            ValueError: No model fitted for this `(model, completion_frac)` pair;
                `df_traj` has no runs; the fitted model expects feature columns the
                builder did not produce; or `df_traj` is missing `experiment_id`,
                `iteration_step` or `energy`. A `completion_frac` outside (0, 1]
                is never fitted, so it surfaces as the no-fitted-model case.
        """
        fitted_models = self.results.fitted_models

        # The key holds the fraction as a float: `1` and `1.0` agree, but a fraction
        # recomputed as 0.1 + 0.2 will not match 0.3. Pass back the value fit_evaluate
        # was given.
        key = f'{model.lower()}_{float(completion_frac)}'
        if key not in fitted_models:
            raise ValueError(
                f"No fitted model for '{model}' at fraction {completion_frac}. "
                f'Call fit_evaluate first, or pick one of: {sorted(fitted_models)}'
            )
        estimator = fitted_models[key]
        df_feat = extract_features_at_fraction(df_traj, completion_frac)

        # A frame with no runs makes the builder return `pd.DataFrame([])`, which has no
        # columns at all, so the check below would blame every feature.
        if df_feat.empty:
            raise ValueError('df_traj has no runs to predict')

        # `df_feat[cols]` raises a bare KeyError that names neither the fraction nor
        # the model.
        # --
        # Catches a `results` object loaded from disk whose model was fitted under
        # an older feature list.
        expected = list(getattr(estimator, 'feature_names_in_', ALL_FEATURES))
        missing = [f for f in expected if f not in df_feat.columns]
        if missing:
            raise ValueError(
                f'Feature mismatch at fraction {completion_frac}: the fitted {model} '
                f'expects columns {missing}, which df_traj did not produce.'
            )

        preds = estimator.predict(df_feat[expected])
        return pd.Series(preds, index=df_feat[RUN_KEY].values, name='predicted_energy')

    def _track_fraction(
        self,
        completion_frac: float,
        metrics_by_model: dict[str, dict[str, float]],
    ) -> None:
        """Log one MLflow run per completion fraction with both models' pooled metrics.

        Metric names are `f'{MODEL_KEYS[name]}_{metric}'`, e.g. `xgboost_mae`, so the
        prefix is the same string that keys `fitted_models` and that `predict` accepts
        as `model=`. One run per fraction rather than per model per fold, because the
        energy metrics are already pooled over all folds.

        Any failure is logged and swallowed: an uncaught tracking error here would abort
        `fit_evaluate` before it returns, skipping this fraction's refit and every later
        fraction (earlier fractions stay on `self.results`).

        Args:
            completion_frac: The fraction these metrics were computed at.
            metrics_by_model: Display name -> `compute_metrics` output, as built in
                `fit_evaluate`.
        """
        try:
            # Imported at call time so the test suite can monkeypatch
            # `quantum_pipeline.ml.tracking.tracker`; a module-level import would
            # have bound the original object before the patch.
            from quantum_pipeline.ml.tracking import tracker

            run_name = f'frac_{round(completion_frac * 100)}pct'
            with tracker.run(
                self.experiment_name,
                run_name=run_name,
                params={
                    'completion_frac': completion_frac,
                    'ridge_alpha': self.ridge_alpha,
                    **{f'xgb_{k}': v for k, v in self.xgb_params.items()},
                },
            ):
                tracker.log_metrics(
                    {
                        f'{MODEL_KEYS[name]}_{metric}': value
                        for name, metrics in metrics_by_model.items()
                        for metric, value in metrics.items()
                    }
                )
        except Exception:
            logger.warning('MLflow logging failed - continuing without tracking', exc_info=True)
