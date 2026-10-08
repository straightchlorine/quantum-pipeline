"""Leave-one-molecule-out (LOMO) training and evaluation for the convergence predictor.

`ConvergencePredictor` is the entry point: `fit_evaluate` scores XGBoost, Random Forest
and Logistic Regression at each horizon K, then refits all three on everything so
`predict_proba` can serve new runs.

A horizon is the number of opening iterations a model may read, and the question
at each one is "will this run converge, given only its first K iterations?".

Evaluation is leave-one-molecule-out (LOMO): each molecule is held out once and scored by
models trained on the others, so a score reports performance on a molecule the model never
saw. A random split would put sibling runs of the same molecule on both sides and flatter
the score.

Every scored (model, horizon, molecule) gets its own `FoldResult`, and
`ConvergencePredictorResults.summary` averages those, so a molecule with few runs
counts as much as one with many. A molecule whose runs all share one outcome has
no defined ROC AUC and is left out of those averages.

The classification target is the `converged` column. On real data it is scipy's
`OptimizeResult.success`, renamed by `docker/airflow/scripts/quantum_ml_feature_processing.py`:
the optimizer met its own stopping criterion, not that the energy is right.

A run that stalls far above the ground state can still be labelled converged.
The one case that forces the label to 0 is the pipeline's own evaluation cap
aborting a run (`VQESolver._make_truncated_result` sets `success=False`), so a 0
mixes "the optimizer never met its criterion" with "the pipeline stopped it".

On synthetic data (`convergence/synthetic.py`) the label is chosen first and the curve is
drawn to match it. A score on those trajectories therefore exercises this code and says
nothing about whether the features predict real runs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from quantum_pipeline.ml.convergence.features import (
    HORIZONS,
    compute_horizon_features,
    get_horizon_feature_names,
)
from quantum_pipeline.ml.convergence.models import (
    build_logistic_regression_classifier,
    build_random_forest_classifier,
    build_xgboost_classifier,
    evaluate_classifier,
)
from quantum_pipeline.ml.convergence.results import (
    ConvergencePredictorResults,
    FoldResult,
)
from quantum_pipeline.ml.fitting import fit_checked, lomo_folds
from quantum_pipeline.ml.registry import MODEL_KEYS
from quantum_pipeline.ml.schema import RUN_KEY

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class ConvergencePredictor:
    """Train and evaluate convergence classifiers on VQE trajectories.

    Each of XGBoost, Random Forest and Logistic Regression predicts whether a run will be
    labelled `converged` from its first K iterations only, so a run can be scored as soon
    as it reaches K iterations instead of waiting for its outcome.

    One set of models is fitted per horizon K, not one model taking K as input: many feature
    columns carry K in their names (`energy_slope_first10`), and the set itself differs by K
    (`improvement_ratio_k{k}` exists only for K >= 20, so the K=10 set lacks it).
    A horizon is therefore a different feature set, not a different value of one feature.

    Evaluation is leave-one-molecule-out, so a score reports performance on a molecule the
    model never saw.

    Args:
        horizons: Horizons K to evaluate, each an int number of opening iterations. Defaults
            to `HORIZONS` (10, 20, 50) when None; an empty list evaluates nothing. A run needs
            at least K iterations to be included at horizon K; shorter runs are dropped by
            `compute_horizon_features`.
        use_mlflow: Whether to log runs to MLflow via the tracker singleton.
        experiment_name: MLflow experiment name.
        xgb_params: XGBoost hyperparameter overrides, merged over the defaults in
            `models.build_xgboost_classifier`. Overriding `scale_pos_weight` pins one value
            for every fold, replacing the per-fold weight, and the `pos_weight` logged to
            MLflow then no longer matches what XGBoost used.
        rf_params: Random Forest overrides, merged over `models.build_random_forest_classifier`.
        logreg_params: Logistic Regression overrides, merged over
            `models.build_logistic_regression_classifier`.
    """

    def __init__(
        self,
        horizons: list[int] | None = None,
        use_mlflow: bool = False,
        experiment_name: str = 'convergence_predictor',
        xgb_params: dict[str, Any] | None = None,
        rf_params: dict[str, Any] | None = None,
        logreg_params: dict[str, Any] | None = None,
    ) -> None:
        self.horizons = list(HORIZONS) if horizons is None else list(horizons)
        self.use_mlflow = use_mlflow
        self.experiment_name = experiment_name
        self.xgb_params = xgb_params or {}
        self.rf_params = rf_params or {}
        self.logreg_params = logreg_params or {}

        # Populated by fit_evaluate and read by predict_proba.
        # --
        # Public, and the same object fit_evaluate returns, so results persisted to disk
        # (joblib is one way) can be assigned back and used to predict without retraining.
        # fit_evaluate replaces it with a fresh object on every call, so models fitted or
        # loaded earlier are dropped, even if the call then raises.
        #
        # What gets dumped must be the whole results object, not its `fitted_models` dict,
        # because predict_proba reads `self.results.fitted_models` and a bare dict has no
        # such attribute:
        #
        #     results = p.fit_evaluate(df_traj)
        #     joblib.dump(results, 'models.joblib')
        #
        #     p = ConvergencePredictor(horizons=[10])
        #     p.results = joblib.load('models.joblib')
        #     p.predict_proba(new_runs, horizon_k=10)
        #
        # fitted_models is keyed f'{MODEL_KEYS[name]}_{k}', e.g. 'xgboost_10'.
        self.results = ConvergencePredictorResults()

    def _build_models(
        self,
        numeric_features: list[str],
        categorical_features: list[str],
        pos_weight: float,
        k: int,
    ) -> dict[str, Pipeline]:
        """Fresh unfitted pipelines, keyed by the display names `MODEL_KEYS` maps.

        Called once per fold and again for the full refit, so the set of models and their
        order live here rather than being restated at each call site. A new XGBoost is
        needed per fold anyway: `scale_pos_weight` is fixed at construction and differs
        between folds.

        Args:
            numeric_features: Columns to median-impute and scale (Logistic Regression also
                multiplies a few of them pairwise, see `k`).
            categorical_features: Columns to one-hot encode.
            pos_weight: Passed to XGBoost only. The other two balance classes themselves by
                default (`class_weight='balanced'`, overridable through `rf_params` and
                `logreg_params`).
            k: Passed to Logistic Regression only, to name the horizon-specific slope column
                (`energy_slope_first{k}`) it pairs with the other interaction features
                (moving-window std, qubit count, random-init flag). The expectation that the
                slope reads differently for small and large molecules is a hypothesis, not a
                measured result; see `models.build_logistic_regression_classifier`.
        """
        return {
            'XGBoost': build_xgboost_classifier(
                numeric_features, categorical_features, pos_weight, **self.xgb_params
            ),
            'RandomForest': build_random_forest_classifier(
                numeric_features, categorical_features, **self.rf_params
            ),
            'LogisticRegression': build_logistic_regression_classifier(
                numeric_features, categorical_features, k, **self.logreg_params
            ),
        }

    def _run_lomo_fold(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        numeric_features: list[str],
        categorical_features: list[str],
        k: int,
    ) -> tuple[dict[str, np.ndarray], float]:
        """Fit all three classifiers on one fold's training molecules and predict the held-out one.

        Args:
            X_train: Feature rows of every molecule except the held-out one.
            y_train: 0/1 `converged` labels for `X_train`, with both classes present.
            X_test: Feature rows of the held-out molecule; only predicted on.
            numeric_features: Columns the pipelines median-impute and scale.
            categorical_features: Columns the pipelines one-hot encode.
            k: Horizon. Only Logistic Regression reads it (see `_build_models`); the other
                two take the feature lists as given. Also names the horizon in the
                non-convergence warning from `fit_checked`.

        Returns:
            Two values. A dict from display name ('XGBoost', 'RandomForest',
            'LogisticRegression') to the probability of convergence for each row of
            `X_test`, in order. And the class weight computed for this fold (n_neg / n_pos),
            which XGBoost is given unless `xgb_params` overrides `scale_pos_weight`. Scoring
            is left to the caller.
        """
        # Labels are 0/1 with 1 = converged; both counts feed the class weight below.
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())

        # XGBoost's scale_pos_weight: each positive row counts n_neg / n_pos times, so both
        # classes carry the same total weight.
        # --
        # Taken from this fold's training rows only, because the held-out molecule's labels
        # must not shape the model being tested on it. Random Forest and Logistic Regression
        # apply the same ratio themselves through class_weight='balanced'.
        #
        # n_pos is 0 only for a single-class training fold, which fit_evaluate skips before
        # calling here. The guard just avoids a ZeroDivisionError for a direct caller. It
        # would not make such a call work: on all-0 labels XGBoost and Random Forest fit, but
        # Random Forest's predict_proba then has one column, so `[:, 1]` raises IndexError
        # (and Logistic Regression's fit raises ValueError).
        fold_pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0

        models = self._build_models(numeric_features, categorical_features, fold_pos_weight, k)

        probas = {}
        for name, model in models.items():
            fit_checked(model, X_train, y_train, f'{name} K={k} fold')

            # Column 1 is class 1 (converged): predict_proba orders columns by sorted class.
            probas[name] = model.predict_proba(X_test)[:, 1]

        return probas, fold_pos_weight

    def fit_evaluate(self, df_traj: pd.DataFrame) -> ConvergencePredictorResults:
        """Run leave-one-molecule-out cross-validation at every configured horizon.

        Per horizon K:
            - build one feature row per run from its first K iterations,
            - hold out each molecule in turn, fit the three classifiers on the rest, and
              record ROC AUC, PR AUC, Brier score and MCC on the held-out molecule (see
              `models.evaluate_classifier`; MCC thresholds the probability at 0.5, the other
              three read it as is),
            - refit all three on every molecule for inference.

        A horizon whose data cannot support evaluation (no run reaching K, no `converged`
        labels, no `molecule_name`, fewer than 10 runs that reach K, fewer than 2 molecules,
        or every run sharing one outcome) is skipped with a warning, so one thin horizon does not
        abandon the others. A single fold is skipped the same way when its training side holds
        only one class. If every horizon is skipped the returned object is empty, and
        `predict_proba` then raises for every key.

        Args:
            df_traj: Iteration-level trajectory DataFrame, one row per iteration. It must
                have `experiment_id`, `iteration_step` and `energy`; the feature builder
                raises without them. To be scored a horizon also needs `converged` (read
                from each run's first row) and `molecule_name` with at least two distinct
                values. The other feature columns are optional: the builder recomputes or
                defaults what is missing, but a model fitted with a column cannot predict
                without it (see `predict_proba`).

        Returns:
            `ConvergencePredictorResults` with one `FoldResult` per scored (model, horizon,
            molecule) and the refit models in `fitted_models`. Also assigned to
            `self.results`, which is what `predict_proba` reads.

        Raises:
            ValueError: A required column is missing or a horizon is below 1 (both from
                `compute_horizon_features`). A null label in a run's first row also makes
                the builder raise, as ValueError or TypeError.
        """
        results = ConvergencePredictorResults()
        self.results = results

        # Each horizon is a separate problem. Nothing is shared across horizons except the
        # trajectory data they read from.
        for k in self.horizons:
            logger.info('Evaluating convergence predictor at horizon K=%d ...', k)

            df_feat = compute_horizon_features(df_traj, k)

            # With no run reaching K the builder returns a frame with no columns at all, so
            # this must be caught before the label check, which would blame a missing label.
            if df_feat.empty:
                logger.warning(
                    'No run reaches K=%d steps (%d runs) - skipping horizon',
                    k,
                    df_traj[RUN_KEY].nunique(),
                )
                continue

            # No label column means prediction data reached a training method.
            # --
            # The builder carries `converged` through only when df_traj has it, and a run
            # still in flight has no outcome yet.
            if 'converged' not in df_feat.columns or df_feat['converged'].isnull().all():
                logger.warning('No target column (converged) at K=%d - skipping', k)
                continue

            # Guard for a builder that starts tolerating null labels. Today it casts each
            # run's label with int(), which raises on a null, so this drops nothing and the
            # all-null test above cannot fire.
            df_feat = df_feat.dropna(subset=['converged'])

            # A floor on the whole horizon, not on a fold; lomo_folds applies its own
            # per-fold minimum afterwards.
            # --
            # 10 is twice that minimum (min_train=5). With at least two molecules (checked
            # below) the smallest holds at most half the runs, so the smallest molecule's
            # fold keeps 5 or more training rows and is never size-skipped, which means at
            # least one fold survives lomo_folds. Folds for larger molecules can still be
            # size-skipped (8 runs of one molecule plus 2 of another: holding out the 8-run
            # molecule leaves 2 training rows), and any fold whose training side holds one
            # class is skipped afterwards, so passing this floor does not guarantee a scored
            # fold.
            # The 10 is hard-coded; if the default min_train of lomo_folds changes, update it.
            if len(df_feat) < 10:
                logger.warning('Too few samples (%d) at K=%d - skipping', len(df_feat), k)
                continue

            # molecule_name is the grouping key: no column, no groups, no LOMO.
            if 'molecule_name' not in df_feat.columns:
                logger.warning('molecule_name column missing at K=%d - skipping', k)
                continue

            # LOMO needs a molecule to hold out and at least one other to train on.
            molecules = df_feat['molecule_name'].unique()
            if len(molecules) < 2:
                logger.warning(
                    'Need >=2 molecules for LOMO-CV (got %d) - skipping K=%d',
                    len(molecules),
                    k,
                )
                continue

            numeric_features, categorical_features = get_horizon_feature_names(
                k, list(df_feat.columns)
            )
            all_features = numeric_features + categorical_features

            # 1 where the optimizer met its own stopping criterion, which is weaker than
            # "found the ground state"; see the module docstring.
            y = df_feat['converged'].values.astype(int)

            # With one outcome every fold's training side has one class, so no fold is
            # scored, and neither the full refit nor its class weight is defined.
            if len(np.unique(y)) < 2:
                logger.warning('Only one converged outcome at K=%d - skipping', k)
                continue

            # Numeric NaNs stay: every pipeline starts with a median imputer, which is
            # fitted on whichever rows it is handed - the training fold.
            # --
            # Filling here would take one median over all rows, the held-out molecule
            # included, and every fold would then train on that number. A column that is
            # NaN throughout (energy_delta_k1: step 1 has no predecessor) is filled with 0
            # instead of dropped (keep_empty_features=True in the imputer, see
            # `preprocessing.build_preprocessor`), so the column stays and carries no signal.
            X = df_feat[all_features]

            # lomo_folds yields (held-out molecule, train rows, test rows) per molecule,
            # as row positions, so X.iloc and y[...] stay aligned with df_feat. It skips
            # a molecule whose training side has fewer than 5 rows.
            #
            # fold_idx counts the folds yielded, including ones skipped below, so MLflow run
            # names can show a gap.
            for fold_idx, (held_out_mol, train_idx, test_idx) in enumerate(lomo_folds(df_feat)):
                X_train, y_train = X.iloc[train_idx], y[train_idx]

                X_test, y_test = X.iloc[test_idx], y[test_idx]

                # A classifier needs both classes in its training rows. One class is left
                # when every other molecule shares an outcome, e.g. holding out the only
                # molecule that ever converged.
                # --
                # A held-out molecule with one class is not skipped: its ROC AUC is
                # undefined, so evaluate_classifier returns NaN for all four metrics and
                # `summary` leaves that fold out of the averages.
                if len(np.unique(y_train)) < 2:
                    logger.warning(
                        'K=%d fold %d: training set has only one class - skipping', k, fold_idx
                    )
                    continue

                probas, fold_pos_weight = self._run_lomo_fold(
                    X_train,
                    y_train,
                    X_test,
                    numeric_features,
                    categorical_features,
                    k,
                )

                for model_name, proba in probas.items():
                    metrics = evaluate_classifier(y_test, proba)
                    results.fold_results.append(
                        FoldResult(
                            model_name=model_name,
                            horizon_k=k,
                            held_out_molecule=held_out_mol,
                            roc_auc=metrics['roc_auc'],
                            pr_auc=metrics['pr_auc'],
                            brier_score=metrics['brier_score'],
                            mcc=metrics['mcc'],
                            n_train=len(X_train),
                            n_test=len(X_test),
                        )
                    )

                if self.use_mlflow:
                    # FoldResult keeps the display names ('XGBoost'); MLflow gets the
                    # MODEL_KEYS form ('xgboost') in its run names and `model` param.
                    self._track_fold(
                        k,
                        fold_idx,
                        held_out_mol,
                        {MODEL_KEYS[name]: proba for name, proba in probas.items()},
                        y_test,
                        len(X_train),
                        fold_pos_weight,
                    )

            # Refit on every molecule. Never scored - the folds above did that - only for
            # predict_proba, which looks them up by key.
            #
            # The class weight now comes from all rows: no held-out molecule is left to
            # protect.
            n_neg_all = int((y == 0).sum())
            n_pos_all = int((y == 1).sum())
            full_pos_weight = float(n_neg_all / n_pos_all)

            full_models = self._build_models(
                numeric_features, categorical_features, full_pos_weight, k
            )
            for name, full_model in full_models.items():
                fit_checked(full_model, X, y, f'{name} K={k} full refit')
                results.fitted_models[f'{MODEL_KEYS[name]}_{k}'] = full_model

        return results

    def predict_proba(
        self,
        df_traj: pd.DataFrame,
        horizon_k: int = 10,
        model: str = MODEL_KEYS['XGBoost'],
    ) -> pd.Series:
        """Predict the probability that each new run converges, from its first K iterations.

        Features are rebuilt with the same functions and horizon as in training, so the
        column names (which carry K) match what the fitted pipeline expects.

        Args:
            df_traj: Trajectory data in the same format as `fit_evaluate`. `converged` may
                be absent and normally is, since the run has no outcome yet. Predictions
                never use it, but the feature builder still casts each run's first-row value
                to int when the column is present, so a null there raises (ValueError, or
                TypeError for None / pd.NA): leave the column out rather than nulling it.
            horizon_k: Horizon of the model to use, as the int `fit_evaluate` was given.
                It is matched as text, so `10.0` does not find the model fitted for `10`.
                A horizon `fit_evaluate` skipped has no model.
            model: 'xgboost', 'random_forest' or 'logistic_regression', the `MODEL_KEYS`
                values of the three models this class fits ('ridge' belongs to the energy
                estimator). Case is ignored, but the display names are not accepted unless they
                are one word: 'RandomForest' lowercases to 'randomforest', which matches
                nothing, while 'XGBoost' happens to work.

        Returns:
            Series named `convergence_probability`, indexed by the run's experiment_id
            values (the index itself is unnamed): the probability that the optimizer
            reports success. A run shorter than `horizon_k`
            iterations is dropped by the feature builder and is absent, so the Series can
            be shorter than the number of runs in `df_traj`.

        Raises:
            ValueError: No model is fitted for this `(model, horizon_k)` pair; `df_traj` is
                missing `experiment_id`, `iteration_step` or `energy`; no run in `df_traj`
                reaches `horizon_k`; or the fitted model expects feature columns that
                `df_traj` did not produce.
            TypeError: `converged` is present and null in a run's first row as None or
                pd.NA (np.nan raises ValueError instead).
        """
        fitted_models = self.results.fitted_models

        # `model` must be one of the three keys this class fits; lower() only forgives case.
        key = f'{model.lower()}_{horizon_k}'
        if key not in fitted_models:
            raise ValueError(
                f"No fitted model for '{model}' at horizon K={horizon_k}. "
                f'Call fit_evaluate first, or pick one of: {sorted(fitted_models)}'
            )
        estimator = fitted_models[key]

        # Built exactly as in fit_evaluate. get_horizon_feature_names keeps only the columns
        # present, so an optional column missing from df_traj (optimizer, basis_set,
        # num_qubits) drops out of X here and the check below reports it. The `in
        # df_feat.columns` filter that follows is then redundant.
        df_feat = compute_horizon_features(df_traj, horizon_k)

        # With no run reaching K the builder returns a frame with no columns at all, so
        # the check below would list every column as missing.
        if df_feat.empty:
            raise ValueError(f'No run in df_traj reaches horizon K={horizon_k} steps')

        numeric_features, categorical_features = get_horizon_feature_names(
            horizon_k, list(df_feat.columns)
        )
        all_features = [
            f for f in (numeric_features + categorical_features) if f in df_feat.columns
        ]

        X = df_feat[all_features]

        # sklearn would also refuse a frame with a column missing, but its message names
        # only the column, not the horizon or model.
        # --
        # feature_names_in_ is every column the pipeline saw when fitted, forwarded from its
        # first step. A pipeline fitted on a bare array has none, the default [] applies,
        # and this check then passes without checking anything. A feature column the model
        # was not fitted on (say `optimizer` was absent in training) is harmless: the first
        # pipeline step selects only the columns it was fitted on.
        expected = list(getattr(estimator, 'feature_names_in_', []))
        missing = [f for f in expected if f not in X.columns]
        if missing:
            raise ValueError(
                f'Feature mismatch at horizon K={horizon_k}: the fitted {model} expects '
                f'{len(expected)} columns and df_traj produced {len(X.columns)}. '
                f'Missing: {missing}. The optional trajectory columns present when the '
                'model was trained must also be present now.'
            )

        proba = estimator.predict_proba(X)[:, 1]
        return pd.Series(
            proba,
            index=df_feat[RUN_KEY].values,
            name='convergence_probability',
        )

    def _track_fold(
        self,
        k: int,
        fold_idx: int,
        held_out_mol: str,
        probas: dict[str, np.ndarray],
        y_test: np.ndarray,
        n_train: int,
        pos_weight: float,
    ) -> None:
        """Log one MLflow run per model for a single held-out molecule at one horizon.

        One run per model per fold rather than pooled, because every fold has its own
        held-out molecule and class weight; averaging across folds happens in
        `ConvergencePredictorResults.summary`, not in MLflow.

        Run names are `{model}_lomo_k{k}_fold{fold_idx}_{molecule}`. The metrics are the
        four `evaluate_classifier` returns, recomputed here from the probabilities, with
        any non-finite value left out: a held-out molecule with one outcome scores NaN on
        all four, so its run carries parameters and no metrics.

        Any failure is logged and swallowed: an uncaught tracking error would abort
        `fit_evaluate` before it returns, skipping the remaining folds and every refit
        (results from folds already scored stay on `self.results`).

        Args:
            k: Horizon being evaluated.
            fold_idx: Position of this fold among those `lomo_folds` yielded.
            held_out_mol: Molecule held out as the test set for this fold.
            probas: `MODEL_KEYS` value ('xgboost') to predicted probability of convergence
                for each held-out row, as `fit_evaluate` builds it. Used verbatim as the
                run's `model` parameter and in its name.
            y_test: 0/1 `converged` labels for the held-out rows.
            n_train: Number of rows the models in this fold were trained on.
            pos_weight: The weight computed for XGBoost in this fold. Logged only, so it
                differs from what XGBoost used if `xgb_params` overrides `scale_pos_weight`.
                Logged on every model's run, including Random Forest and Logistic
                Regression, which balance classes through `class_weight` instead and
                ignore it.
        """
        try:
            # Imported at call time so the test suite can monkeypatch
            # `quantum_pipeline.ml.tracking.tracker`; a module-level import would
            # have bound the original object before the patch.
            from quantum_pipeline.ml.tracking import tracker

            for model_name, proba in probas.items():
                metrics = evaluate_classifier(y_test, proba)

                finite_metrics = {k2: v for k2, v in metrics.items() if np.isfinite(v)}

                with tracker.run(
                    self.experiment_name,
                    run_name=f'{model_name}_lomo_k{k}_fold{fold_idx}_{held_out_mol}',
                    params={
                        'model': model_name,
                        'horizon_k': k,
                        'held_out_molecule': held_out_mol,
                        'cv_strategy': 'lomo',
                        'pos_weight': pos_weight,
                        'n_train': n_train,
                    },
                ):
                    tracker.log_metrics(finite_metrics)
        except Exception:
            logger.warning('MLflow logging failed - continuing without tracking', exc_info=True)
