"""
Convergence predictor for VQE optimization trajectories.

Binary classifier: predicts whether a VQE run will converge given
the first K iterations (horizon-K prediction).

Models
------
- XGBoost classifier (primary)
- Random Forest (ensemble baseline)
- Logistic Regression with polynomial interaction terms (linear interpretable baseline)

Cross-validation
----------------
Leave-One-Molecule-Out (LOMO): train on all molecules except the held-out one.
Tests generalization across molecular complexity, not just across runs of the same molecule.
Evaluation follows Roberts et al. (2017) grouped cross-validation principles and
QUA-26 training protocol.

Usage
-----
    from quantum_pipeline.ml.convergence_predictor import ConvergencePredictor, generate_synthetic_trajectories

    # With real iteration data (ml_iteration_features table):
    df_traj = ...  # columns: run_id, iteration_step, energy, molecule_name, num_qubits,
                   #          optimizer, basis_set, init_strategy, converged,
                   #          energy_delta (optional), energy_moving_std_5 (optional),
                   #          steps_since_improvement (optional), is_new_minimum (optional),
                   #          parameter_delta_norm (optional), mean_param_delta_norm (optional)

    predictor = ConvergencePredictor()
    results = predictor.fit_evaluate(df_traj)
    print(results.summary())

    # With synthetic data (for development/testing):
    df_traj = generate_synthetic_trajectories(n_runs=500, seed=42)
    predictor = ConvergencePredictor()
    results = predictor.fit_evaluate(df_traj)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HORIZONS = [10, 20, 50]

CATEGORICAL_FEATURES = ['optimizer', 'basis_set']

# Run-level numeric features always merged into the horizon feature matrix
RUN_NUMERIC_FEATURES = [
    'num_qubits',
    'init_strategy_random',  # 1 if random init, 0 if HF
    'qubit_x_random',  # interaction: barren plateau severity x random init
    'mean_param_delta_norm',
]

# Key features for polynomial interaction in LogReg (horizon-independent part)
_POLY_BASE_FEATURES = [
    'energy_moving_std_5_at_k',
    'num_qubits',
    'init_strategy_random',
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FoldResult:
    """Metrics for a single model / horizon / LOMO fold."""

    model_name: str
    horizon_k: int
    held_out_molecule: str
    roc_auc: float
    pr_auc: float
    brier_score: float
    mcc: float
    n_train: int
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
    """Aggregated results across models, horizons, and LOMO folds."""

    fold_results: list[FoldResult] = field(default_factory=list)
    fitted_models: dict[str, Any] = field(default_factory=dict)  # {f'{model_name}_{k}': model}

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
                if np.isfinite(r.roc_auc)
            ]
        )

        if df.empty:
            lines.append('No finite metrics to display.')
        else:
            agg = df.groupby(['model', 'horizon_k'])[
                ['roc_auc', 'pr_auc', 'brier_score', 'mcc']
            ].mean()
            lines.append(agg.round(4).to_string())

        return '\n'.join(lines)

    _LOWER_IS_BETTER = frozenset({'brier_score'})

    def best(self, metric: str = 'roc_auc') -> FoldResult | None:
        """Return the fold result with the best value of metric."""
        finite = [r for r in self.fold_results if np.isfinite(getattr(r, metric, float('nan')))]
        if not finite:
            return None
        cmp = min if metric in self._LOWER_IS_BETTER else max
        return cmp(finite, key=lambda r: getattr(r, metric))


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


def generate_synthetic_trajectories(
    n_runs: int = 300,
    min_iter: int = 55,
    max_iter: int = 100,
    seed: int = 42,
    molecules: list[dict] | None = None,
) -> pd.DataFrame:
    """Generate VQE-like trajectory data with convergence labels.

    Simulates energy descent using an exponential decay model:
        E(t) = E_final + A * exp(-k * t) + σ * noise(t)

    Convergence probability scales with molecular complexity (qubit count) and
    initialization strategy, mirroring barren plateau effects (McClean et al., 2018).

    Args:
        n_runs: Total number of VQE runs to simulate.
        min_iter: Minimum iterations per run (must be >= 55 for K=50 horizon tests).
        max_iter: Maximum iterations per run.
        seed: Random seed for reproducibility.
        molecules: List of molecule spec dicts. Defaults to 4-molecule test set.

    Returns:
        DataFrame with columns: run_id, molecule_name, num_qubits, optimizer, basis_set,
        init_strategy, iteration_step (1-indexed), energy, energy_delta,
        energy_moving_avg_5, energy_moving_std_5, cumulative_min_energy,
        steps_since_improvement, is_new_minimum, parameter_delta_norm,
        mean_param_delta_norm, converged.
    """
    rng = np.random.default_rng(seed)

    if molecules is None:
        molecules = [
            {'name': 'H2', 'num_qubits': 4, 'e_fci': -1.1175, 'e_local': -0.78},
            {'name': 'LiH', 'num_qubits': 6, 'e_fci': -7.882, 'e_local': -7.50},
            {'name': 'H2O', 'num_qubits': 8, 'e_fci': -75.012, 'e_local': -74.60},
            {'name': 'BeH2', 'num_qubits': 10, 'e_fci': -15.834, 'e_local': -15.60},
        ]

    optimizers = ['COBYLA', 'L-BFGS-B', 'Nelder-Mead', 'SLSQP']
    basis_sets = ['sto-3g', 'cc-pVDZ']
    init_strategies = ['random', 'hf']

    all_rows: list[dict] = []
    runs_per_mol = max(1, n_runs // len(molecules))

    for mol in molecules:
        for run_idx in range(runs_per_mol):
            run_id = f'syn_{mol["name"]}_{run_idx:04d}'
            optimizer = optimizers[run_idx % len(optimizers)]
            basis_set = basis_sets[run_idx % len(basis_sets)]
            init_strategy = init_strategies[run_idx % len(init_strategies)]
            n_iter = int(rng.integers(max(min_iter, 10), max(max_iter + 1, min_iter + 1)))

            # Convergence probability: lower for more qubits and random init
            base_prob = max(0.15, 0.90 - mol['num_qubits'] * 0.04)
            if init_strategy == 'random':
                base_prob *= 0.65
            converged = bool(rng.random() < base_prob)

            e_fci = mol['e_fci']
            e_local = mol['e_local']

            if converged:
                e_final = e_fci + rng.uniform(0.0, 0.02)
                plateau_frac = 1.0
                noise_scale = abs(e_fci) * 0.001
            else:
                e_final = e_local + rng.uniform(-0.05, 0.10)
                plateau_frac = float(rng.uniform(0.2, 0.6))
                noise_scale = abs(e_fci) * 0.005

            e_start = e_local + rng.uniform(0.2, 0.5) * abs(e_local - e_fci)
            amplitude = e_start - e_final
            k_rate = -np.log(0.01) / (n_iter * plateau_frac + 1e-9)

            run_rows: list[dict] = []
            param_deltas: list[float] = []
            prev_energy = e_start
            cummin = e_start
            running_min = e_start
            steps_since_imp = 0
            energy_window: list[float] = []

            for step in range(1, n_iter + 1):  # 1-indexed per protocol
                t = float(step - 1)
                e_ideal = e_final + amplitude * np.exp(-k_rate * t)
                noise = float(rng.normal(0.0, noise_scale))
                e = e_ideal + noise

                # Occasional uphill moves for gradient-free optimizers
                if optimizer in ('COBYLA', 'Nelder-Mead') and rng.random() < 0.04:
                    e += abs(noise) * 1.5

                delta = e - prev_energy
                param_delta = float(rng.exponential(0.1) * np.exp(-k_rate * t))
                param_deltas.append(param_delta)

                is_new_min = 1 if e < running_min - 1e-8 else 0
                if is_new_min:
                    running_min = e
                    steps_since_imp = 0
                else:
                    steps_since_imp += 1

                cummin = min(cummin, e)
                energy_window.append(e)
                if len(energy_window) > 5:
                    energy_window.pop(0)

                moving_avg = float(np.mean(energy_window))
                moving_std = float(np.std(energy_window)) if len(energy_window) > 1 else 0.0

                run_rows.append(
                    {
                        'run_id': run_id,
                        'molecule_name': mol['name'],
                        'num_qubits': mol['num_qubits'],
                        'optimizer': optimizer,
                        'basis_set': basis_set,
                        'init_strategy': init_strategy,
                        'iteration_step': step,
                        'energy': e,
                        'energy_delta': delta,
                        'energy_moving_avg_5': moving_avg,
                        'energy_moving_std_5': moving_std,
                        'cumulative_min_energy': cummin,
                        'steps_since_improvement': steps_since_imp,
                        'is_new_minimum': is_new_min,
                        'parameter_delta_norm': param_delta,
                        'converged': int(converged),
                        'mean_param_delta_norm': 0.0,  # placeholder; filled after loop
                    }
                )

                prev_energy = e

            # Back-fill mean_param_delta_norm (run-level stat)
            mean_pdelta = float(np.mean(param_deltas)) if param_deltas else 0.0
            for row in run_rows:
                row['mean_param_delta_norm'] = mean_pdelta

            all_rows.extend(run_rows)

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def compute_horizon_features(df_iter: pd.DataFrame, k: int) -> pd.DataFrame:
    """Compute convergence predictor features from the first K iterations.

    No lookahead: only uses iterations 1..K. Merges horizon-K trajectory features
    with run-level metadata features.

    Args:
        df_iter: Iteration-level DataFrame. Required columns: run_id, iteration_step, energy.
            Optional (enhances features): energy_delta, energy_moving_std_5,
            steps_since_improvement, is_new_minimum, parameter_delta_norm,
            converged, molecule_name, num_qubits, optimizer, basis_set, init_strategy,
            mean_param_delta_norm.
        k: Prediction horizon (number of first iterations to use). Must be >= 1.

    Returns:
        DataFrame with one row per run_id containing horizon-K features plus
        run-level metadata. If 'converged' is present in df_iter, it is included
        as the target column.
    """
    required = {'run_id', 'iteration_step', 'energy'}
    missing = required - set(df_iter.columns)
    if missing:
        raise ValueError(f'df_iter missing required columns: {missing}')
    if k < 1:
        raise ValueError(f'k must be >= 1, got {k}')

    df_k = df_iter[df_iter['iteration_step'] <= k].copy()
    feature_rows: list[dict] = []

    for run_id, grp in df_k.groupby('run_id'):
        grp = grp.sort_values('iteration_step').reset_index(drop=True)
        n_obs = len(grp)
        energies = grp['energy'].values.astype(float)

        feat: dict[str, Any] = {'run_id': run_id}

        # --Per-iteration energy delta features (first 5 iterations) --------
        for i in range(1, min(6, k + 1)):
            row_i = grp[grp['iteration_step'] == i]
            if len(row_i) > 0 and 'energy_delta' in grp.columns:
                feat[f'energy_delta_k{i}'] = float(row_i['energy_delta'].iloc[0])
            else:
                feat[f'energy_delta_k{i}'] = float('nan')

        # --Energy slope over first K iterations (OLS) -----------------------
        if n_obs >= 2:
            x = np.arange(n_obs, dtype=float)
            feat[f'energy_slope_first{k}'] = float(np.polyfit(x, energies, 1)[0])
        else:
            feat[f'energy_slope_first{k}'] = 0.0

        # --Rolling energy std at iteration K (barren plateau proxy) ---------
        if 'energy_moving_std_5' in grp.columns:
            row_k = grp[grp['iteration_step'] == k]
            if len(row_k) > 0:
                feat['energy_moving_std_5_at_k'] = float(row_k['energy_moving_std_5'].iloc[-1])
            else:
                window = energies[-min(5, n_obs) :]
                feat['energy_moving_std_5_at_k'] = (
                    float(np.std(window)) if len(window) > 1 else 0.0
                )
        else:
            window = energies[-min(5, n_obs) :]
            feat['energy_moving_std_5_at_k'] = float(np.std(window)) if len(window) > 1 else 0.0

        # --Steps since improvement at iteration K ----------------------------
        if 'steps_since_improvement' in grp.columns:
            row_k = grp[grp['iteration_step'] == k]
            if len(row_k) > 0:
                feat[f'steps_since_improvement_at_k{k}'] = float(
                    row_k['steps_since_improvement'].iloc[-1]
                )
            else:
                feat[f'steps_since_improvement_at_k{k}'] = float(k)  # impute: stagnant
        else:
            feat[f'steps_since_improvement_at_k{k}'] = float(k)

        # --Longest plateau in first K iterations ----------------------------
        if 'is_new_minimum' in grp.columns:
            is_new_min = grp['is_new_minimum'].values
            max_plateau = 0
            run_len = 0
            for val in is_new_min:
                if val == 0:
                    run_len += 1
                    max_plateau = max(max_plateau, run_len)
                else:
                    run_len = 0
            feat[f'longest_plateau_first{k}'] = float(max_plateau)
        else:
            feat[f'longest_plateau_first{k}'] = float(k)

        # --Parameter movement signals ----------------------------------------
        if 'parameter_delta_norm' in grp.columns:
            param_deltas = grp['parameter_delta_norm'].values.astype(float)
            feat[f'param_delta_norm_mean_k{k}'] = float(np.mean(param_deltas))
            feat[f'param_delta_norm_std_k{k}'] = (
                float(np.std(param_deltas)) if len(param_deltas) > 1 else 0.0
            )
        else:
            feat[f'param_delta_norm_mean_k{k}'] = 0.0
            feat[f'param_delta_norm_std_k{k}'] = 0.0

        # --Improvement saturation (K >= 20) ---------------------------------
        if k >= 20:
            if 'energy_delta' in grp.columns:
                mid = k // 2
                first_half = float(grp[grp['iteration_step'] <= mid]['energy_delta'].sum())
                second_half = float(grp[grp['iteration_step'] > mid]['energy_delta'].sum())
                feat[f'improvement_ratio_k{k}'] = first_half / (first_half + second_half + 1e-12)
            else:
                feat[f'improvement_ratio_k{k}'] = 0.5  # neutral default

        # --Run-level metadata ------------------------------------------------
        for col in ('molecule_name', 'num_qubits', 'optimizer', 'basis_set', 'init_strategy'):
            if col in grp.columns:
                feat[col] = grp[col].iloc[0]

        if 'parameter_delta_norm' in grp.columns:
            feat['mean_param_delta_norm'] = float(grp['parameter_delta_norm'].mean())
        else:
            feat['mean_param_delta_norm'] = 0.0

        # Derived features
        init_random = 1 if str(feat.get('init_strategy', 'random')).lower() == 'random' else 0
        feat['init_strategy_random'] = float(init_random)
        feat['qubit_x_random'] = float(feat.get('num_qubits', 0)) * float(init_random)

        # --Target ------------------------------------------------------------
        if 'converged' in grp.columns:
            feat['converged'] = int(grp['converged'].iloc[0])

        feature_rows.append(feat)

    return pd.DataFrame(feature_rows)


def get_horizon_feature_names(k: int, available_cols: list[str]) -> tuple[list[str], list[str]]:
    """Return (numeric_features, categorical_features) for a given horizon K.

    Only includes features that are present in available_cols.
    """
    numeric: list[str] = list(RUN_NUMERIC_FEATURES)

    # Per-iteration energy delta features
    numeric.extend(f'energy_delta_k{i}' for i in range(1, min(6, k + 1)))

    numeric.append(f'energy_slope_first{k}')
    numeric.append('energy_moving_std_5_at_k')
    numeric.append(f'steps_since_improvement_at_k{k}')
    numeric.append(f'longest_plateau_first{k}')
    numeric.append(f'param_delta_norm_mean_k{k}')
    numeric.append(f'param_delta_norm_std_k{k}')
    if k >= 20:
        numeric.append(f'improvement_ratio_k{k}')

    numeric = [f for f in numeric if f in available_cols]
    categorical = [f for f in CATEGORICAL_FEATURES if f in available_cols]

    return numeric, categorical


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


def _build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    from quantum_pipeline.ml.preprocessing import build_preprocessor

    return build_preprocessor(numeric_features, categorical_features)


def _build_xgboost_clf(
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
        'scale_pos_weight': pos_weight,
        'random_state': 42,
        'verbosity': 0,
    }
    defaults.update(kwargs)
    return Pipeline(
        [
            ('prep', _build_preprocessor(numeric_features, categorical_features)),
            ('model', XGBClassifier(**defaults)),
        ]
    )


def _build_rf_clf(
    numeric_features: list[str],
    categorical_features: list[str],
    **kwargs: Any,
) -> Pipeline:
    defaults: dict[str, Any] = {
        'n_estimators': 200,
        'max_depth': None,
        'min_samples_split': 5,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1,
    }
    defaults.update(kwargs)
    return Pipeline(
        [
            ('prep', _build_preprocessor(numeric_features, categorical_features)),
            ('model', RandomForestClassifier(**defaults)),
        ]
    )


def _build_logreg_clf(
    numeric_features: list[str],
    categorical_features: list[str],
    k: int,
    **kwargs: Any,
) -> Pipeline:
    """LogReg with polynomial interactions on 4 key features (per QUA-26 protocol §4.3)."""
    slope_feat = f'energy_slope_first{k}'
    poly_features = [f for f in [slope_feat, *_POLY_BASE_FEATURES] if f in numeric_features]
    other_numeric = [f for f in numeric_features if f not in poly_features]

    transformers: list[tuple] = []
    if poly_features:
        transformers.append(
            (
                'poly_scaled',
                Pipeline(
                    [
                        ('scaler', StandardScaler()),
                        (
                            'poly',
                            PolynomialFeatures(
                                degree=2, include_bias=False, interaction_only=True
                            ),
                        ),
                    ]
                ),
                poly_features,
            )
        )
    if other_numeric:
        transformers.append(('num', StandardScaler(), other_numeric))
    if categorical_features:
        transformers.append(
            (
                'cat',
                OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

    logreg_defaults: dict[str, Any] = {
        'C': 1.0,
        'class_weight': 'balanced',
        'solver': 'lbfgs',
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


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _evaluate_classifier(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> dict[str, float]:
    """Compute ROC-AUC, PR-AUC, Brier score, and MCC."""
    y_pred_labels = (y_pred_proba >= 0.5).astype(int)

    if len(np.unique(y_true)) < 2:
        return {
            'roc_auc': float('nan'),
            'pr_auc': float('nan'),
            'brier_score': float('nan'),
            'mcc': float('nan'),
        }

    return {
        'roc_auc': float(roc_auc_score(y_true, y_pred_proba)),
        'pr_auc': float(average_precision_score(y_true, y_pred_proba)),
        'brier_score': float(brier_score_loss(y_true, y_pred_proba)),
        'mcc': float(matthews_corrcoef(y_true, y_pred_labels)),
    }


# ---------------------------------------------------------------------------
# Main predictor class
# ---------------------------------------------------------------------------


class ConvergencePredictor:
    """Train and evaluate convergence classifiers on VQE trajectories.

    Trains XGBoost, Random Forest, and Logistic Regression classifiers using
    Leave-One-Molecule-Out (LOMO) cross-validation for each configured
    prediction horizon K (number of first iterations available).

    Args:
        horizons: Prediction horizons K in iterations. Defaults to [10, 20, 50].
        use_mlflow: Whether to log runs to MLflow via the tracker singleton.
        experiment_name: MLflow experiment name.
        xgb_params: XGBoost hyperparameter overrides.
        rf_params: RandomForest hyperparameter overrides.
        logreg_params: LogisticRegression hyperparameter overrides.
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
        self.horizons = horizons or HORIZONS
        self.use_mlflow = use_mlflow
        self.experiment_name = experiment_name
        self.xgb_params = xgb_params or {}
        self.rf_params = rf_params or {}
        self.logreg_params = logreg_params or {}
        self._fitted_models: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    @staticmethod
    def _impute_nans(X: pd.DataFrame, numeric_features: list[str]) -> pd.DataFrame:
        """Impute NaN values with column median (0.0 if all NaN)."""
        X = X.copy()
        for col in numeric_features:
            if col in X.columns and X[col].isnull().any():
                X[col] = X[col].fillna(X[col].median() if X[col].notna().any() else 0.0)
        return X

    def _run_lomo_fold(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        numeric_features: list[str],
        categorical_features: list[str],
        k: int,
    ) -> dict[str, np.ndarray]:
        """Train XGBoost, RF, and LogReg on one LOMO fold, return predicted probabilities."""
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        fold_pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0

        models = {
            'XGBoost': _build_xgboost_clf(
                numeric_features, categorical_features, fold_pos_weight, **self.xgb_params
            ),
            'RandomForest': _build_rf_clf(
                numeric_features, categorical_features, **self.rf_params
            ),
            'LogisticRegression': _build_logreg_clf(
                numeric_features, categorical_features, k, **self.logreg_params
            ),
        }

        probas = {}
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            for name, model in models.items():
                model.fit(X_train, y_train)
                probas[name] = model.predict_proba(X_test)[:, 1]

        return probas

    def fit_evaluate(self, df_traj: pd.DataFrame) -> ConvergencePredictorResults:
        """Run LOMO cross-validation across all configured horizons.

        For each horizon K:
        1. Compute horizon-K features from iteration-level data.
        2. Run Leave-One-Molecule-Out CV with GroupKFold.
        3. Train XGBoost, RF, LogReg per fold.
        4. Collect per-fold metrics (ROC-AUC, PR-AUC, Brier, MCC).
        5. Store final models trained on all data for inference.

        Args:
            df_traj: Iteration-level trajectory DataFrame. Must contain:
                run_id, iteration_step, energy, converged, molecule_name.

        Returns:
            :class:`ConvergencePredictorResults` with fold results and fitted models.
        """
        results = ConvergencePredictorResults()

        for k in self.horizons:
            logger.info('Evaluating convergence predictor at horizon K=%d ...', k)
            df_feat = compute_horizon_features(df_traj, k)

            if 'converged' not in df_feat.columns or df_feat['converged'].isnull().all():
                logger.warning('No target column (converged) at K=%d - skipping', k)
                continue

            df_feat = df_feat.dropna(subset=['converged'])
            if len(df_feat) < 10:
                logger.warning('Too few samples (%d) at K=%d - skipping', len(df_feat), k)
                continue

            if 'molecule_name' not in df_feat.columns:
                logger.warning('molecule_name column missing at K=%d - skipping', k)
                continue

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

            y = df_feat['converged'].values.astype(int)
            X = self._impute_nans(df_feat[all_features], numeric_features)
            groups = df_feat['molecule_name'].values

            n_splits = min(len(molecules), 9)
            lomo = GroupKFold(n_splits=n_splits)

            for fold_idx, (train_idx, test_idx) in enumerate(lomo.split(X, y, groups=groups)):
                held_out_mol = str(df_feat.iloc[test_idx]['molecule_name'].iloc[0])

                X_train, y_train = X.iloc[train_idx], y[train_idx]
                X_test, y_test = X.iloc[test_idx], y[test_idx]

                if len(X_train) < 5 or len(X_test) < 1:
                    continue

                if len(np.unique(y_train)) < 2:
                    logger.warning(
                        'K=%d fold %d: training set has only one class - skipping', k, fold_idx
                    )
                    continue

                probas = self._run_lomo_fold(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    numeric_features,
                    categorical_features,
                    k,
                )

                for model_name, proba in probas.items():
                    metrics = _evaluate_classifier(y_test, proba)
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
                    # Compute pos_weight for logging from this fold's training set
                    n_neg = int((y_train == 0).sum())
                    n_pos = int((y_train == 1).sum())
                    log_pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0
                    self._log_fold_to_mlflow(
                        k,
                        fold_idx,
                        held_out_mol,
                        {n.lower().replace(' ', '_'): p for n, p in probas.items()},
                        y_test,
                        len(X_train),
                        log_pos_weight,
                    )

            # Store models trained on all data for inference
            n_neg_all = int((y == 0).sum())
            n_pos_all = int((y == 1).sum())
            full_pos_weight = float(n_neg_all / n_pos_all) if n_pos_all > 0 else 1.0

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=ConvergenceWarning)
                xgb_full = _build_xgboost_clf(
                    numeric_features, categorical_features, full_pos_weight, **self.xgb_params
                )
                rf_full = _build_rf_clf(numeric_features, categorical_features, **self.rf_params)
                logreg_full = _build_logreg_clf(
                    numeric_features, categorical_features, k, **self.logreg_params
                )
                xgb_full.fit(X, y)
                rf_full.fit(X, y)
                logreg_full.fit(X, y)

            results.fitted_models[f'xgboost_{k}'] = xgb_full
            results.fitted_models[f'random_forest_{k}'] = rf_full
            results.fitted_models[f'logistic_regression_{k}'] = logreg_full
            self._fitted_models.update(results.fitted_models)

        return results

    def predict_proba(
        self,
        df_traj: pd.DataFrame,
        horizon_k: int = 10,
        model: str = 'xgboost',
    ) -> pd.Series:
        """Predict convergence probability for new runs.

        Must call :meth:`fit_evaluate` before predict_proba.

        Args:
            df_traj: Iteration-level trajectory data (same format as ``fit_evaluate``).
            horizon_k: Horizon used during training. Must match one of :attr:`horizons`.
            model: 'xgboost', 'random_forest', or 'logistic_regression'.

        Returns:
            Series of convergence probabilities in [0, 1], indexed by run_id.
        """
        key = f'{model.lower()}_{horizon_k}'
        if key not in self._fitted_models:
            raise ValueError(
                f"No fitted model for '{model}' at horizon K={horizon_k}. "
                f'Available: {sorted(self._fitted_models)}'
            )

        df_feat = compute_horizon_features(df_traj, horizon_k)
        numeric_features, categorical_features = get_horizon_feature_names(
            horizon_k, list(df_feat.columns)
        )
        all_features = [
            f for f in (numeric_features + categorical_features) if f in df_feat.columns
        ]

        X = self._impute_nans(df_feat[all_features], numeric_features)

        proba = self._fitted_models[key].predict_proba(X)[:, 1]
        return pd.Series(
            proba,
            index=df_feat['run_id'].values,
            name='convergence_probability',
        )

    # ------------------------------------------------------------------
    # MLflow integration
    # ------------------------------------------------------------------

    def _log_fold_to_mlflow(
        self,
        k: int,
        fold_idx: int,
        held_out_mol: str,
        probas: dict[str, np.ndarray],
        y_test: np.ndarray,
        n_train: int,
        pos_weight: float,
    ) -> None:
        try:
            from quantum_pipeline.ml.tracking import tracker

            for model_name, proba in probas.items():
                metrics = _evaluate_classifier(y_test, proba)
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
