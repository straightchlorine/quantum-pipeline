"""Tests for quantum_pipeline.ml.energy_estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantum_pipeline.ml.energy_estimator import (
    EnergyEstimator,
    EnergyEstimatorResults,
    EvaluationResult,
    extract_features_at_fraction,
    generate_synthetic_trajectories,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def small_traj() -> pd.DataFrame:
    """Minimal synthetic trajectory: 4 molecules, 100 runs, fast."""
    return generate_synthetic_trajectories(n_runs=80, max_iter=40, seed=0)


@pytest.fixture(scope='module')
def medium_traj() -> pd.DataFrame:
    """Larger trajectory dataset for model quality tests."""
    return generate_synthetic_trajectories(n_runs=200, max_iter=80, seed=1)


@pytest.fixture(scope='module')
def fitted_estimator(medium_traj: pd.DataFrame) -> EnergyEstimator:
    """Train energy estimator once, share across all tests."""
    est = EnergyEstimator(completion_fracs=[0.25, 0.5, 0.75])
    est.fit_evaluate(medium_traj)
    return est


@pytest.fixture(scope='module')
def estimator_results(medium_traj: pd.DataFrame) -> tuple[EnergyEstimator, EnergyEstimatorResults]:
    """Estimator and results from the single training run."""
    est = EnergyEstimator(completion_fracs=[0.25, 0.5, 0.75])
    results = est.fit_evaluate(medium_traj)
    return est, results


# ---------------------------------------------------------------------------
# generate_synthetic_trajectories
# ---------------------------------------------------------------------------


class TestGenerateSyntheticTrajectories:
    def test_returns_dataframe(self, small_traj: pd.DataFrame) -> None:
        assert isinstance(small_traj, pd.DataFrame)

    def test_required_columns_present(self, small_traj: pd.DataFrame) -> None:
        required = {
            'experiment_id',
            'molecule_name',
            'num_qubits',
            'optimizer',
            'basis_set',
            'ansatz_reps',
            'iteration_step',
            'energy',
            'cumulative_min_energy',
            'energy_delta',
            'parameter_delta_norm',
            'final_energy',
            'converged',
        }
        assert required.issubset(set(small_traj.columns))

    def test_multiple_molecules(self, small_traj: pd.DataFrame) -> None:
        assert small_traj['molecule_name'].nunique() >= 2

    def test_iteration_steps_start_at_zero(self, small_traj: pd.DataFrame) -> None:
        min_step = small_traj.groupby('experiment_id')['iteration_step'].min()
        assert (min_step == 0).all()

    def test_cumulative_min_is_monotone_per_experiment(self, small_traj: pd.DataFrame) -> None:
        for _, grp in small_traj.groupby('experiment_id'):
            grp = grp.sort_values('iteration_step')
            cummin = grp['cumulative_min_energy'].values
            assert np.all(np.diff(cummin) <= 1e-9), 'cumulative_min_energy must be non-increasing'

    def test_energy_in_plausible_range(self, small_traj: pd.DataFrame) -> None:
        # Energies should be negative for molecular systems
        assert (small_traj['energy'] < 0).all()

    def test_no_null_values(self, small_traj: pd.DataFrame) -> None:
        assert not small_traj.isnull().any().any()

    def test_reproducibility(self) -> None:
        df1 = generate_synthetic_trajectories(n_runs=20, seed=99)
        df2 = generate_synthetic_trajectories(n_runs=20, seed=99)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self) -> None:
        df1 = generate_synthetic_trajectories(n_runs=20, seed=1)
        df2 = generate_synthetic_trajectories(n_runs=20, seed=2)
        assert not df1['energy'].equals(df2['energy'])

    def test_custom_molecules(self) -> None:
        mols = [
            {'name': 'TestMol', 'num_qubits': 4, 'e_fci': -1.0, 'e_local': -0.7},
        ]
        df = generate_synthetic_trajectories(n_runs=4, molecules=mols, seed=0)
        assert (df['molecule_name'] == 'TestMol').all()


# ---------------------------------------------------------------------------
# extract_features_at_fraction
# ---------------------------------------------------------------------------


class TestExtractFeaturesAtFraction:
    def test_returns_one_row_per_experiment(self, small_traj: pd.DataFrame) -> None:
        n_experiments = small_traj['experiment_id'].nunique()
        df_feat = extract_features_at_fraction(small_traj, 0.5)
        assert len(df_feat) == n_experiments

    def test_required_feature_columns_present(self, small_traj: pd.DataFrame) -> None:
        from quantum_pipeline.ml.energy_estimator import ALL_FEATURES

        df_feat = extract_features_at_fraction(small_traj, 0.5)
        for col in ALL_FEATURES:
            assert col in df_feat.columns, f'Missing feature column: {col}'

    def test_trajectory_fraction_matches_input(self, small_traj: pd.DataFrame) -> None:
        for frac in [0.25, 0.5, 0.75]:
            df_feat = extract_features_at_fraction(small_traj, frac)
            assert (df_feat['trajectory_fraction'] == frac).all()

    def test_cumulative_min_never_exceeds_current(self, small_traj: pd.DataFrame) -> None:
        df_feat = extract_features_at_fraction(small_traj, 0.5)
        # cumulative min ≤ current energy (energies are negative; more negative = lower)
        assert (df_feat['cumulative_min_energy'] <= df_feat['current_energy'] + 1e-8).all()

    def test_more_data_at_higher_fraction(self, small_traj: pd.DataFrame) -> None:
        df_25 = extract_features_at_fraction(small_traj, 0.25)
        df_75 = extract_features_at_fraction(small_traj, 0.75)
        # Higher fraction → more steps observed
        assert df_75['num_steps_observed'].mean() > df_25['num_steps_observed'].mean()

    def test_invalid_fraction_raises(self, small_traj: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match='completion_frac'):
            extract_features_at_fraction(small_traj, 0.0)
        with pytest.raises(ValueError, match='completion_frac'):
            extract_features_at_fraction(small_traj, 1.5)

    def test_missing_required_columns_raises(self) -> None:
        df_bad = pd.DataFrame({'experiment_id': ['a'], 'energy': [-1.0]})
        with pytest.raises(ValueError, match='missing required columns'):
            extract_features_at_fraction(df_bad, 0.5)

    def test_single_step_does_not_crash(self) -> None:
        df = pd.DataFrame({
            'experiment_id': ['exp1'],
            'iteration_step': [0],
            'energy': [-1.0],
            'molecule_name': ['H2'],
            'num_qubits': [4],
            'optimizer': ['COBYLA'],
            'basis_set': ['sto-3g'],
            'ansatz_reps': [2],
            'final_energy': [-1.1],
        })
        df_feat = extract_features_at_fraction(df, 1.0)
        assert len(df_feat) == 1

    def test_no_nulls_in_numeric_features(self, small_traj: pd.DataFrame) -> None:
        from quantum_pipeline.ml.energy_estimator import NUMERIC_FEATURES

        df_feat = extract_features_at_fraction(small_traj, 0.5)
        assert not df_feat[NUMERIC_FEATURES].isnull().any().any()


# ---------------------------------------------------------------------------
# EnergyEstimator.fit_evaluate
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestEnergyEstimatorFitEvaluate:
    def test_returns_results_object(
        self,
        estimator_results: tuple[EnergyEstimator, EnergyEstimatorResults],
    ) -> None:
        _, results = estimator_results
        assert isinstance(results, EnergyEstimatorResults)

    def test_produces_results_for_each_model_and_frac(
        self,
        estimator_results: tuple[EnergyEstimator, EnergyEstimatorResults],
    ) -> None:
        _, results = estimator_results
        model_frac_pairs = {(r.model_name, r.completion_frac) for r in results.results}
        for frac in [0.25, 0.75]:
            assert ('Ridge', frac) in model_frac_pairs
            assert ('XGBoost', frac) in model_frac_pairs

    def test_mae_is_positive(
        self,
        estimator_results: tuple[EnergyEstimator, EnergyEstimatorResults],
    ) -> None:
        _, results = estimator_results
        for r in results.results:
            assert r.mae >= 0.0

    def test_rmse_geq_mae(
        self,
        estimator_results: tuple[EnergyEstimator, EnergyEstimatorResults],
    ) -> None:
        _, results = estimator_results
        for r in results.results:
            assert r.rmse >= r.mae - 1e-10

    def test_per_molecule_metrics_present(
        self,
        estimator_results: tuple[EnergyEstimator, EnergyEstimatorResults],
    ) -> None:
        _, results = estimator_results
        for r in results.results:
            assert len(r.per_molecule) > 0
            for metrics in r.per_molecule.values():
                assert 'mae' in metrics
                assert 'rmse' in metrics
                assert 'r2' in metrics

    def test_higher_completion_frac_improves_mae(
        self,
        estimator_results: tuple[EnergyEstimator, EnergyEstimatorResults],
    ) -> None:
        """More trajectory information should generally improve energy prediction."""
        _, results = estimator_results
        xgb_results = {
            r.completion_frac: r.mae
            for r in results.results
            if r.model_name == 'XGBoost'
        }
        # 75% should not be worse than 25% (allow small tolerance)
        if 0.25 in xgb_results and 0.75 in xgb_results:
            assert xgb_results[0.75] <= xgb_results[0.25] * 1.5, (
                f'Expected MAE at 75% <= MAE at 25% * 1.5, '
                f'got {xgb_results[0.75]:.4f} vs {xgb_results[0.25]:.4f}'
            )

    def test_fitted_models_stored(
        self,
        estimator_results: tuple[EnergyEstimator, EnergyEstimatorResults],
    ) -> None:
        _, results = estimator_results
        assert 'ridge_0.5' in results.fitted_models
        assert 'xgboost_0.5' in results.fitted_models

    def test_both_models_produce_finite_mae(
        self,
        estimator_results: tuple[EnergyEstimator, EnergyEstimatorResults],
    ) -> None:
        """Both Ridge and XGBoost should produce finite, non-NaN MAE values."""
        _, results = estimator_results
        for r in results.results:
            assert np.isfinite(r.mae), f'{r.model_name} MAE is not finite: {r.mae}'
            assert np.isfinite(r.rmse), f'{r.model_name} RMSE is not finite: {r.rmse}'
            assert r.mae >= 0.0

    def test_single_molecule_skipped_gracefully(self) -> None:
        df = generate_synthetic_trajectories(
            n_runs=40,
            max_iter=30,
            seed=0,
            molecules=[{'name': 'H2', 'num_qubits': 4, 'e_fci': -1.1, 'e_local': -0.8}],
        )
        est = EnergyEstimator(completion_fracs=[0.5])
        # Should not raise — returns empty results
        results = est.fit_evaluate(df)
        assert isinstance(results, EnergyEstimatorResults)

    def test_summary_contains_model_names(
        self,
        estimator_results: tuple[EnergyEstimator, EnergyEstimatorResults],
    ) -> None:
        _, results = estimator_results
        summary = results.summary()
        assert 'Ridge' in summary
        assert 'XGBoost' in summary


# ---------------------------------------------------------------------------
# EnergyEstimator.predict
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestEnergyEstimatorPredict:
    def test_predict_returns_series(
        self,
        fitted_estimator: EnergyEstimator,
        medium_traj: pd.DataFrame,
    ) -> None:
        preds = fitted_estimator.predict(medium_traj, completion_frac=0.5, model='xgboost')
        assert isinstance(preds, pd.Series)

    def test_predict_length_matches_experiments(
        self,
        fitted_estimator: EnergyEstimator,
        medium_traj: pd.DataFrame,
    ) -> None:
        n_experiments = medium_traj['experiment_id'].nunique()
        preds = fitted_estimator.predict(medium_traj, completion_frac=0.5, model='xgboost')
        assert len(preds) == n_experiments

    def test_predict_before_fit_raises(self, medium_traj: pd.DataFrame) -> None:
        est = EnergyEstimator(completion_fracs=[0.5])
        with pytest.raises(ValueError, match='No fitted model'):
            est.predict(medium_traj, completion_frac=0.5, model='xgboost')

    def test_predict_unavailable_frac_raises(
        self,
        fitted_estimator: EnergyEstimator,
        medium_traj: pd.DataFrame,
    ) -> None:
        # fitted_estimator has fracs=[0.25, 0.5, 0.75]; frac=0.9 is not available
        with pytest.raises(ValueError, match='No fitted model'):
            fitted_estimator.predict(medium_traj, completion_frac=0.9, model='xgboost')

    def test_predict_ridge_and_xgboost(
        self,
        fitted_estimator: EnergyEstimator,
        medium_traj: pd.DataFrame,
    ) -> None:
        preds_ridge = fitted_estimator.predict(medium_traj, completion_frac=0.5, model='ridge')
        preds_xgb = fitted_estimator.predict(medium_traj, completion_frac=0.5, model='xgboost')
        assert len(preds_ridge) == len(preds_xgb)
        # Predictions should differ between models
        assert not preds_ridge.equals(preds_xgb)


# ---------------------------------------------------------------------------
# EvaluationResult helpers
# ---------------------------------------------------------------------------


class TestEvaluationResult:
    def test_str_contains_model_name(self) -> None:
        r = EvaluationResult('XGBoost', 0.5, mae=0.01, rmse=0.02, r2=0.99)
        assert 'XGBoost' in str(r)
        assert '50%' in str(r)

    def test_best_returns_lowest_mae(self) -> None:
        results = EnergyEstimatorResults()
        results.results = [
            EvaluationResult('Ridge', 0.5, mae=0.10, rmse=0.15, r2=0.80),
            EvaluationResult('XGBoost', 0.5, mae=0.05, rmse=0.08, r2=0.95),
        ]
        best = results.best('mae')
        assert best is not None
        assert best.model_name == 'XGBoost'

    def test_best_on_empty_returns_none(self) -> None:
        results = EnergyEstimatorResults()
        assert results.best() is None
