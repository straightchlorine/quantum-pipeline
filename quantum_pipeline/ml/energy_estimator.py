"""
Energy estimator for VQE ground-state energy prediction from partial trajectories.

Predicts final ground-state energy (in Hartree) given a VQE optimization trajectory
truncated at a configurable completion fraction (25%, 50%, 75%).

Models
------
- XGBoost regressor (primary)
- Ridge Regression (baseline, with StandardScaler)

Cross-validation
----------------
Leave-One-Molecule-Out (LOMO): train on all molecules except the held-out one.
Tests generalization across molecular complexity, not just across runs of the same molecule.

Usage
-----
    from quantum_pipeline.ml.energy_estimator import EnergyEstimator, generate_synthetic_trajectories

    # With real iteration data (ml_iteration_features table):
    df_traj = ...  # columns: experiment_id, iteration_step, energy, molecule_name, num_qubits,
                   #          optimizer, basis_set, ansatz_reps, converged (optional),
                   #          cumulative_min_energy (optional), energy_delta (optional),
                   #          parameter_delta_norm (optional)

    estimator = EnergyEstimator()
    results = estimator.fit_evaluate(df_traj, completion_fracs=[0.25, 0.5, 0.75])
    print(results.summary())

    # With synthetic data (for development/testing):
    df_traj = generate_synthetic_trajectories(n_runs=500, seed=42)
    estimator = EnergyEstimator()
    results = estimator.fit_evaluate(df_traj)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COMPLETION_FRACS = [0.25, 0.50, 0.75]

# Features derived from the trajectory snapshot at completion fraction p
NUMERIC_FEATURES = [
    'cumulative_min_energy',
    'current_energy',
    'energy_slope',
    'energy_improvement_rate',
    'energy_delta_mean',
    'energy_delta_std',
    'energy_moving_std',
    'steps_since_improvement',
    'num_steps_observed',
    'num_qubits',
    'ansatz_reps',
    'trajectory_fraction',
]

CATEGORICAL_FEATURES = ['optimizer', 'basis_set']

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    """Metrics for a single model at a single trajectory completion fraction."""

    model_name: str
    completion_frac: float
    mae: float
    rmse: float
    r2: float
    per_molecule: dict[str, dict[str, float]] = field(default_factory=dict)
    n_train: int = 0
    n_test: int = 0

    def __str__(self) -> str:
        return (
            f'{self.model_name} @ {int(self.completion_frac * 100)}%: '
            f'MAE={self.mae:.4f} Ha  RMSE={self.rmse:.4f} Ha  R²={self.r2:.4f} '
            f'(n_train={self.n_train}, n_test={self.n_test})'
        )


@dataclass
class EnergyEstimatorResults:
    """Aggregated results across models and completion fractions."""

    results: list[EvaluationResult] = field(default_factory=list)
    fitted_models: dict[str, Any] = field(default_factory=dict)  # {model_name: fitted model}

    def summary(self) -> str:
        lines = ['Energy Estimator — LOMO Cross-Validation Summary', '=' * 60]
        for r in sorted(self.results, key=lambda x: (x.completion_frac, x.model_name)):
            lines.append(str(r))
        return '\n'.join(lines)

    def best(self, metric: str = 'mae') -> EvaluationResult | None:
        if not self.results:
            return None
        return min(self.results, key=lambda r: getattr(r, metric))


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


def generate_synthetic_trajectories(
    n_runs: int = 300,
    max_iter: int = 150,
    seed: int = 42,
    molecules: list[dict] | None = None,
) -> pd.DataFrame:
    """Generate VQE-like trajectory data for development and testing.

    Simulates energy descent using an exponential decay model:
        E(t) = E_final + A * exp(-k * t) + σ * noise(t)

    Converged runs reach near E_final; non-converged runs plateau at a sub-optimal
    intermediate energy, which is the primary ML signal.

    Args:
        n_runs: Total number of VQE runs to simulate.
        max_iter: Maximum iterations per run.
        seed: Random seed for reproducibility.
        molecules: List of molecule spec dicts. Defaults to 4-molecule test set.

    Returns:
        DataFrame with columns: experiment_id, molecule_name, num_qubits, optimizer,
        basis_set, ansatz_reps, iteration_step, energy, cumulative_min_energy,
        energy_delta, parameter_delta_norm, final_energy.
    """
    rng = np.random.default_rng(seed)

    if molecules is None:
        molecules = [
            {'name': 'H2', 'num_qubits': 4, 'e_fci': -1.1175, 'e_local': -0.78},
            {'name': 'LiH', 'num_qubits': 8, 'e_fci': -7.882, 'e_local': -7.50},
            {'name': 'H2O', 'num_qubits': 10, 'e_fci': -75.012, 'e_local': -74.60},
            {'name': 'BeH2', 'num_qubits': 8, 'e_fci': -15.834, 'e_local': -15.60},
        ]

    optimizers = ['COBYLA', 'L-BFGS-B', 'Nelder-Mead', 'SLSQP']
    basis_sets = ['sto-3g', 'cc-pVDZ']

    rows = []
    runs_per_mol = n_runs // len(molecules)

    for mol in molecules:
        for run_idx in range(runs_per_mol):
            experiment_id = f'syn_{mol["name"]}_{run_idx:04d}'
            optimizer = optimizers[run_idx % len(optimizers)]
            basis_set = basis_sets[run_idx % len(basis_sets)]
            ansatz_reps = rng.integers(1, 4)
            n_iter = rng.integers(30, max_iter + 1)

            # Decide convergence (70% converge for simplicity)
            converged = rng.random() < 0.70

            e_fci = mol['e_fci']
            e_local = mol['e_local']

            if converged:
                e_final = e_fci + rng.uniform(0.0, 0.05)
                plateau_frac = 1.0  # reaches minimum
                noise_scale = abs(e_fci) * 0.002
            else:
                # Trapped in local minimum
                e_final = e_local + rng.uniform(-0.1, 0.1)
                plateau_frac = rng.uniform(0.3, 0.7)  # plateaus early
                noise_scale = abs(e_fci) * 0.01

            # Starting energy: random initialization near top of landscape
            e_start = e_local + rng.uniform(0.3, 0.6) * abs(e_local - e_fci)
            amplitude = e_start - e_final
            k_rate = -np.log(0.01) / (n_iter * plateau_frac + 1e-9)

            energies = []
            prev_energy = e_start
            cummin = e_start
            steps_since_imp = 0

            for step in range(n_iter):
                t = step
                e_ideal = e_final + amplitude * np.exp(-k_rate * t)
                noise = rng.normal(0.0, noise_scale)
                e = e_ideal + noise

                # Add occasional uphill moves (gradient-free optimizers)
                if optimizer in ('COBYLA', 'Nelder-Mead') and rng.random() < 0.05:
                    e += abs(noise) * 2.0

                energies.append(e)
                cummin = min(cummin, e)

                delta = abs(e - prev_energy) if step > 0 else 0.0
                param_delta = rng.exponential(0.1) * np.exp(-k_rate * t)

                rows.append({
                    'experiment_id': experiment_id,
                    'molecule_name': mol['name'],
                    'num_qubits': mol['num_qubits'],
                    'optimizer': optimizer,
                    'basis_set': basis_set,
                    'ansatz_reps': int(ansatz_reps),
                    'iteration_step': step,
                    'energy': e,
                    'cumulative_min_energy': cummin,
                    'energy_delta': delta,
                    'parameter_delta_norm': param_delta,
                    'final_energy': e_final,
                    'converged': converged,
                })
                prev_energy = e

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_features_at_fraction(
    df_traj: pd.DataFrame,
    completion_frac: float,
) -> pd.DataFrame:
    """Extract run-level ML features from trajectories truncated at completion_frac.

    For each experiment, takes only the first ``ceil(total_steps * completion_frac)``
    steps and derives summary statistics that the regressor will use to predict
    the final ground-state energy.

    Args:
        df_traj: Trajectory DataFrame. Required columns: experiment_id, iteration_step,
            energy. Optional: cumulative_min_energy, energy_delta, parameter_delta_norm,
            molecule_name, num_qubits, optimizer, basis_set, ansatz_reps, final_energy.
        completion_frac: Fraction in (0, 1]. E.g. 0.25 = first quarter of trajectory.

    Returns:
        One-row-per-experiment feature DataFrame with columns matching ALL_FEATURES
        plus 'experiment_id', 'molecule_name', 'final_energy'.
    """
    if not 0 < completion_frac <= 1.0:
        raise ValueError(f'completion_frac must be in (0, 1], got {completion_frac}')

    required = {'experiment_id', 'iteration_step', 'energy'}
    missing = required - set(df_traj.columns)
    if missing:
        raise ValueError(f'df_traj missing required columns: {missing}')

    feature_rows = []

    for exp_id, grp in df_traj.groupby('experiment_id'):
        grp = grp.sort_values('iteration_step').reset_index(drop=True)
        total_steps = len(grp)
        cutoff = max(1, int(np.ceil(total_steps * completion_frac)))
        obs = grp.iloc[:cutoff]

        energies = obs['energy'].values.astype(float)
        n_obs = len(energies)

        # ── Cumulative minimum ────────────────────────────────────────────
        if 'cumulative_min_energy' in obs.columns:
            cummin = obs['cumulative_min_energy'].iloc[-1]
        else:
            cummin = float(np.min(energies))

        current_energy = float(energies[-1])

        # ── Energy slope (linear regression) ────────────────────────────
        if n_obs >= 2:
            x = np.arange(n_obs, dtype=float)
            slope, _ = np.polyfit(x, energies, deg=1)
        else:
            slope = 0.0

        # ── Improvement rate ─────────────────────────────────────────────
        total_improvement = float(energies[0] - cummin)  # positive = improved
        energy_improvement_rate = total_improvement / n_obs if n_obs > 0 else 0.0

        # ── Energy delta statistics ──────────────────────────────────────
        if 'energy_delta' in obs.columns:
            deltas = obs['energy_delta'].dropna().values.astype(float)
        elif n_obs >= 2:
            deltas = np.abs(np.diff(energies))
        else:
            deltas = np.array([0.0])

        energy_delta_mean = float(np.mean(deltas)) if len(deltas) > 0 else 0.0
        energy_delta_std = float(np.std(deltas)) if len(deltas) > 1 else 0.0

        # ── Moving std over last 5 steps (plateau detection) ────────────
        window = min(5, n_obs)
        energy_moving_std = float(np.std(energies[-window:])) if window > 1 else 0.0

        # ── Steps since improvement ──────────────────────────────────────
        running_min = energies[0]
        steps_since_imp = 0
        for val in energies:
            if val < running_min - 1e-8:
                running_min = val
                steps_since_imp = 0
            else:
                steps_since_imp += 1

        # ── Metadata ────────────────────────────────────────────────────
        num_qubits = int(grp['num_qubits'].iloc[0]) if 'num_qubits' in grp.columns else 0
        ansatz_reps = int(grp['ansatz_reps'].iloc[0]) if 'ansatz_reps' in grp.columns else 1
        optimizer = str(grp['optimizer'].iloc[0]) if 'optimizer' in grp.columns else 'UNKNOWN'
        basis_set = str(grp['basis_set'].iloc[0]) if 'basis_set' in grp.columns else 'sto-3g'
        molecule_name = str(grp['molecule_name'].iloc[0]) if 'molecule_name' in grp.columns else 'UNKNOWN'

        # ── Target ──────────────────────────────────────────────────────
        final_energy: float | None = None
        if 'final_energy' in grp.columns:
            final_energy = float(grp['final_energy'].iloc[0])
        elif 'minimum_energy' in grp.columns:
            final_energy = float(grp['minimum_energy'].iloc[-1])

        feature_rows.append({
            'experiment_id': exp_id,
            'molecule_name': molecule_name,
            # Numeric features
            'cumulative_min_energy': cummin,
            'current_energy': current_energy,
            'energy_slope': float(slope),
            'energy_improvement_rate': energy_improvement_rate,
            'energy_delta_mean': energy_delta_mean,
            'energy_delta_std': energy_delta_std,
            'energy_moving_std': energy_moving_std,
            'steps_since_improvement': float(steps_since_imp),
            'num_steps_observed': float(n_obs),
            'num_qubits': float(num_qubits),
            'ansatz_reps': float(ansatz_reps),
            'trajectory_fraction': float(completion_frac),
            # Categorical features
            'optimizer': optimizer,
            'basis_set': basis_set,
            # Target
            'final_energy': final_energy,
        })

    return pd.DataFrame(feature_rows)


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------


def _build_preprocessor() -> ColumnTransformer:
    """Build column transformer: scale numerics, one-hot categoricals."""
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES),
            (
                'cat',
                OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder='drop',
    )


def _build_ridge(alpha: float = 1.0) -> Pipeline:
    return Pipeline([
        ('prep', _build_preprocessor()),
        ('model', Ridge(alpha=alpha)),
    ])


def _build_xgboost(**kwargs: Any) -> Any:
    """Build XGBoost regressor pipeline (no scaling needed for tree models)."""
    from xgboost import XGBRegressor  # noqa: PLC0415

    defaults: dict[str, Any] = {
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
    }
    defaults.update(kwargs)
    return Pipeline([
        ('prep', _build_preprocessor()),
        ('model', XGBRegressor(**defaults)),
    ])


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float('nan')
    return {'mae': mae, 'rmse': rmse, 'r2': r2}


# ---------------------------------------------------------------------------
# Main estimator class
# ---------------------------------------------------------------------------


class EnergyEstimator:
    """Train and evaluate XGBoost and Ridge energy estimators on VQE trajectories.

    Evaluation uses Leave-One-Molecule-Out (LOMO) cross-validation to test
    generalization to unseen molecules.

    Args:
        completion_fracs: Trajectory completion fractions to evaluate.
        ridge_alpha: Regularization strength for Ridge baseline.
        xgb_params: XGBoost hyperparameter overrides.
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
        self._fitted_models: dict[str, Any] = {}  # keyed by (model_name, frac)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit_evaluate(self, df_traj: pd.DataFrame) -> EnergyEstimatorResults:
        """Run full LOMO cross-validation across all configured completion fractions.

        For each completion fraction:
        1. Extract features for all experiments.
        2. Run leave-one-molecule-out CV — per fold, train on all other molecules.
        3. Report per-molecule and aggregate MAE / RMSE / R².

        Args:
            df_traj: Iteration-level trajectory DataFrame. See
                :func:`extract_features_at_fraction` for column spec.

        Returns:
            :class:`EnergyEstimatorResults` with evaluation metrics and fitted models.
        """
        results = EnergyEstimatorResults()

        for frac in self.completion_fracs:
            logger.info('Evaluating at %.0f%% trajectory completion …', frac * 100)
            df_feat = extract_features_at_fraction(df_traj, frac)

            if 'final_energy' not in df_feat.columns or df_feat['final_energy'].isnull().all():
                logger.warning(
                    'No target column (final_energy) at frac=%.2f — skipping', frac
                )
                continue

            df_feat = df_feat.dropna(subset=['final_energy'])
            if len(df_feat) < 10:
                logger.warning('Too few samples (%d) at frac=%.2f — skipping', len(df_feat), frac)
                continue

            molecules = df_feat['molecule_name'].unique()
            if len(molecules) < 2:
                logger.warning(
                    'Need ≥2 molecules for LOMO-CV (got %d) — skipping', len(molecules)
                )
                continue

            ridge_preds, xgb_preds, actuals, mol_labels = [], [], [], []
            per_mol_ridge: dict[str, list] = {m: [] for m in molecules}
            per_mol_xgb: dict[str, list] = {m: [] for m in molecules}
            per_mol_actual: dict[str, list] = {m: [] for m in molecules}

            for held_out_mol in molecules:
                train_mask = df_feat['molecule_name'] != held_out_mol
                test_mask = ~train_mask

                df_train = df_feat[train_mask]
                df_test = df_feat[test_mask]

                if len(df_train) < 5 or len(df_test) < 1:
                    continue

                X_train = df_train[ALL_FEATURES]
                y_train = df_train['final_energy'].values
                X_test = df_test[ALL_FEATURES]
                y_test = df_test['final_energy'].values

                # ── Ridge ───────────────────────────────────────────────
                ridge = _build_ridge(alpha=self.ridge_alpha)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    ridge.fit(X_train, y_train)
                y_pred_ridge = ridge.predict(X_test)

                # ── XGBoost ─────────────────────────────────────────────
                xgb = _build_xgboost(**self.xgb_params)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    xgb.fit(X_train, y_train)
                y_pred_xgb = xgb.predict(X_test)

                # Accumulate
                ridge_preds.extend(y_pred_ridge.tolist())
                xgb_preds.extend(y_pred_xgb.tolist())
                actuals.extend(y_test.tolist())
                mol_labels.extend([held_out_mol] * len(y_test))

                per_mol_ridge[held_out_mol] = y_pred_ridge.tolist()
                per_mol_xgb[held_out_mol] = y_pred_xgb.tolist()
                per_mol_actual[held_out_mol] = y_test.tolist()

            if not actuals:
                continue

            y_true_arr = np.array(actuals)
            ridge_arr = np.array(ridge_preds)
            xgb_arr = np.array(xgb_preds)

            # Aggregate metrics
            ridge_metrics = _compute_metrics(y_true_arr, ridge_arr)
            xgb_metrics = _compute_metrics(y_true_arr, xgb_arr)

            # Per-molecule metrics
            per_mol_ridge_metrics = {
                mol: _compute_metrics(np.array(per_mol_actual[mol]), np.array(per_mol_ridge[mol]))
                for mol in molecules
                if per_mol_actual[mol]
            }
            per_mol_xgb_metrics = {
                mol: _compute_metrics(np.array(per_mol_actual[mol]), np.array(per_mol_xgb[mol]))
                for mol in molecules
                if per_mol_actual[mol]
            }

            n_total = len(y_true_arr)
            n_train_avg = int(n_total * (len(molecules) - 1) / len(molecules))

            results.results.append(
                EvaluationResult(
                    model_name='Ridge',
                    completion_frac=frac,
                    mae=ridge_metrics['mae'],
                    rmse=ridge_metrics['rmse'],
                    r2=ridge_metrics['r2'],
                    per_molecule=per_mol_ridge_metrics,
                    n_train=n_train_avg,
                    n_test=n_total - n_train_avg,
                )
            )
            results.results.append(
                EvaluationResult(
                    model_name='XGBoost',
                    completion_frac=frac,
                    mae=xgb_metrics['mae'],
                    rmse=xgb_metrics['rmse'],
                    r2=xgb_metrics['r2'],
                    per_molecule=per_mol_xgb_metrics,
                    n_train=n_train_avg,
                    n_test=n_total - n_train_avg,
                )
            )

            if self.use_mlflow:
                self._log_to_mlflow(frac, ridge_metrics, xgb_metrics)

            # Store models trained on ALL data at this fraction for later inference
            ridge_full = _build_ridge(alpha=self.ridge_alpha)
            xgb_full = _build_xgboost(**self.xgb_params)
            X_all = df_feat[ALL_FEATURES]
            y_all = df_feat['final_energy'].values
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                ridge_full.fit(X_all, y_all)
                xgb_full.fit(X_all, y_all)
            results.fitted_models[f'ridge_{frac}'] = ridge_full
            results.fitted_models[f'xgboost_{frac}'] = xgb_full
            self._fitted_models[f'ridge_{frac}'] = ridge_full
            self._fitted_models[f'xgboost_{frac}'] = xgb_full

        return results

    def predict(
        self,
        df_traj: pd.DataFrame,
        completion_frac: float = 0.5,
        model: str = 'xgboost',
    ) -> pd.Series:
        """Predict final energy for new runs at a given trajectory completion fraction.

        Args:
            df_traj: Trajectory data (same format as ``fit_evaluate``).
            completion_frac: Fraction of trajectory available. Must match a fraction
                used during training (one of 0.25, 0.50, 0.75).
            model: 'xgboost' or 'ridge'.

        Returns:
            Series of predicted energies indexed by experiment_id.
        """
        key = f'{model.lower()}_{completion_frac}'
        if key not in self._fitted_models:
            raise ValueError(
                f"No fitted model for '{model}' at fraction {completion_frac}. "
                f'Available: {list(self._fitted_models)}'
            )
        df_feat = extract_features_at_fraction(df_traj, completion_frac)
        X = df_feat[ALL_FEATURES]
        preds = self._fitted_models[key].predict(X)
        return pd.Series(preds, index=df_feat['experiment_id'].values, name='predicted_energy')

    # ------------------------------------------------------------------
    # MLflow integration
    # ------------------------------------------------------------------

    def _log_to_mlflow(
        self,
        completion_frac: float,
        ridge_metrics: dict[str, float],
        xgb_metrics: dict[str, float],
    ) -> None:
        try:
            from quantum_pipeline.ml.tracking import tracker  # noqa: PLC0415

            run_name = f'frac_{int(completion_frac * 100)}pct'
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
                        'ridge_mae': ridge_metrics['mae'],
                        'ridge_rmse': ridge_metrics['rmse'],
                        'ridge_r2': ridge_metrics['r2'],
                        'xgb_mae': xgb_metrics['mae'],
                        'xgb_rmse': xgb_metrics['rmse'],
                        'xgb_r2': xgb_metrics['r2'],
                    }
                )
        except Exception:  # noqa: BLE001
            logger.debug('MLflow logging failed — continuing without tracking', exc_info=True)
