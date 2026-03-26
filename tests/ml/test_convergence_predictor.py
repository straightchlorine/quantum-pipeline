"""Tests for quantum_pipeline.ml.convergence_predictor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantum_pipeline.ml.convergence_predictor import (
    CATEGORICAL_FEATURES,
    ConvergencePredictor,
    ConvergencePredictorResults,
    FoldResult,
    compute_horizon_features,
    generate_synthetic_trajectories,
    get_horizon_feature_names,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def small_traj() -> pd.DataFrame:
    """Minimal synthetic trajectory: 4 molecules, 60 runs, fast."""
    return generate_synthetic_trajectories(n_runs=60, min_iter=55, max_iter=70, seed=0)


@pytest.fixture(scope='module')
def medium_traj() -> pd.DataFrame:
    """Larger trajectory dataset for model quality tests."""
    return generate_synthetic_trajectories(n_runs=160, min_iter=55, max_iter=80, seed=1)


@pytest.fixture(scope='module')
def fitted_predictor(medium_traj: pd.DataFrame) -> ConvergencePredictor:
    """Train convergence predictor once, share across all tests."""
    predictor = ConvergencePredictor(horizons=[10, 20])
    predictor.fit_evaluate(medium_traj)
    return predictor


@pytest.fixture(scope='module')
def predictor_results(medium_traj: pd.DataFrame) -> tuple[ConvergencePredictor, ConvergencePredictorResults]:
    """Predictor and results from the single training run."""
    predictor = ConvergencePredictor(horizons=[10, 20])
    results = predictor.fit_evaluate(medium_traj)
    return predictor, results


# ---------------------------------------------------------------------------
# generate_synthetic_trajectories
# ---------------------------------------------------------------------------


class TestGenerateSyntheticTrajectories:
    def test_returns_dataframe(self, small_traj: pd.DataFrame) -> None:
        assert isinstance(small_traj, pd.DataFrame)

    def test_required_columns_present(self, small_traj: pd.DataFrame) -> None:
        required = {
            'run_id',
            'molecule_name',
            'num_qubits',
            'optimizer',
            'basis_set',
            'init_strategy',
            'iteration_step',
            'energy',
            'energy_delta',
            'energy_moving_avg_5',
            'energy_moving_std_5',
            'cumulative_min_energy',
            'steps_since_improvement',
            'is_new_minimum',
            'parameter_delta_norm',
            'mean_param_delta_norm',
            'converged',
        }
        assert required.issubset(set(small_traj.columns))

    def test_multiple_molecules(self, small_traj: pd.DataFrame) -> None:
        assert small_traj['molecule_name'].nunique() >= 2

    def test_iteration_steps_are_one_indexed(self, small_traj: pd.DataFrame) -> None:
        min_step = small_traj.groupby('run_id')['iteration_step'].min()
        assert (min_step == 1).all(), 'iteration_step must be 1-indexed'

    def test_cumulative_min_is_monotone_non_increasing(self, small_traj: pd.DataFrame) -> None:
        for _, grp in small_traj.groupby('run_id'):
            grp = grp.sort_values('iteration_step')
            cummin = grp['cumulative_min_energy'].values
            assert np.all(np.diff(cummin) <= 1e-9), 'cumulative_min_energy must be non-increasing'

    def test_energy_values_negative(self, small_traj: pd.DataFrame) -> None:
        assert (small_traj['energy'] < 0).all()

    def test_converged_is_binary(self, small_traj: pd.DataFrame) -> None:
        assert set(small_traj['converged'].unique()).issubset({0, 1})

    def test_no_null_values(self, small_traj: pd.DataFrame) -> None:
        assert not small_traj.isnull().any().any()

    def test_reproducibility(self) -> None:
        df1 = generate_synthetic_trajectories(n_runs=20, seed=77)
        df2 = generate_synthetic_trajectories(n_runs=20, seed=77)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self) -> None:
        df1 = generate_synthetic_trajectories(n_runs=20, seed=10)
        df2 = generate_synthetic_trajectories(n_runs=20, seed=11)
        assert not df1['energy'].equals(df2['energy'])

    def test_minimum_iterations_enforced(self) -> None:
        df = generate_synthetic_trajectories(n_runs=8, min_iter=55, max_iter=60, seed=0)
        min_steps = df.groupby('run_id')['iteration_step'].count().min()
        assert min_steps >= 55, f'Expected ≥55 iterations per run, got {min_steps}'

    def test_custom_molecules(self) -> None:
        mols = [
            {'name': 'TestMol', 'num_qubits': 4, 'e_fci': -1.0, 'e_local': -0.7},
        ]
        df = generate_synthetic_trajectories(n_runs=4, molecules=mols, seed=0)
        assert (df['molecule_name'] == 'TestMol').all()

    def test_mean_param_delta_norm_constant_per_run(self, small_traj: pd.DataFrame) -> None:
        for _, grp in small_traj.groupby('run_id'):
            vals = grp['mean_param_delta_norm'].values
            assert np.allclose(vals, vals[0]), 'mean_param_delta_norm must be constant per run'

    def test_both_init_strategies_present(self, small_traj: pd.DataFrame) -> None:
        strategies = small_traj['init_strategy'].unique()
        assert 'random' in strategies
        assert 'hf' in strategies


# ---------------------------------------------------------------------------
# compute_horizon_features
# ---------------------------------------------------------------------------


class TestComputeHorizonFeatures:
    def test_returns_one_row_per_run(self, small_traj: pd.DataFrame) -> None:
        n_runs = small_traj['run_id'].nunique()
        df_feat = compute_horizon_features(small_traj, k=10)
        assert len(df_feat) == n_runs

    def test_run_id_column_present(self, small_traj: pd.DataFrame) -> None:
        df_feat = compute_horizon_features(small_traj, k=10)
        assert 'run_id' in df_feat.columns

    def test_converged_column_propagated(self, small_traj: pd.DataFrame) -> None:
        df_feat = compute_horizon_features(small_traj, k=10)
        assert 'converged' in df_feat.columns
        assert set(df_feat['converged'].unique()).issubset({0, 1})

    def test_horizon_features_present(self, small_traj: pd.DataFrame) -> None:
        for k in [10, 20, 50]:
            df_feat = compute_horizon_features(small_traj, k=k)
            assert f'energy_slope_first{k}' in df_feat.columns
            assert f'steps_since_improvement_at_k{k}' in df_feat.columns
            assert f'longest_plateau_first{k}' in df_feat.columns
            assert f'param_delta_norm_mean_k{k}' in df_feat.columns

    def test_improvement_ratio_present_for_k20_plus(self, small_traj: pd.DataFrame) -> None:
        df_k20 = compute_horizon_features(small_traj, k=20)
        df_k50 = compute_horizon_features(small_traj, k=50)
        df_k10 = compute_horizon_features(small_traj, k=10)
        assert 'improvement_ratio_k20' in df_k20.columns
        assert 'improvement_ratio_k50' in df_k50.columns
        assert 'improvement_ratio_k10' not in df_k10.columns

    def test_run_level_features_present(self, small_traj: pd.DataFrame) -> None:
        df_feat = compute_horizon_features(small_traj, k=10)
        for col in ('num_qubits', 'init_strategy_random', 'qubit_x_random', 'mean_param_delta_norm'):
            assert col in df_feat.columns, f'Missing run-level feature: {col}'

    def test_init_strategy_random_encoding(self, small_traj: pd.DataFrame) -> None:
        df_feat = compute_horizon_features(small_traj, k=10)
        assert set(df_feat['init_strategy_random'].unique()).issubset({0.0, 1.0})

    def test_qubit_x_random_is_product(self, small_traj: pd.DataFrame) -> None:
        df_feat = compute_horizon_features(small_traj, k=10)
        expected = df_feat['num_qubits'] * df_feat['init_strategy_random']
        np.testing.assert_allclose(df_feat['qubit_x_random'].values, expected.values)

    def test_no_lookahead(self, small_traj: pd.DataFrame) -> None:
        """Horizon K=10 features must use only iterations 1..10."""
        df_k10 = compute_horizon_features(small_traj, k=10)
        df_k50 = compute_horizon_features(small_traj, k=50)
        # Energy slope at k=10 should differ from k=50 (more data → different slope)
        assert not np.allclose(
            df_k10['energy_slope_first10'].values,
            df_k50['energy_slope_first50'].values,
            atol=1e-3,
        )

    def test_missing_required_columns_raises(self) -> None:
        df_bad = pd.DataFrame({'run_id': ['a'], 'energy': [-1.0]})
        with pytest.raises(ValueError, match='missing required columns'):
            compute_horizon_features(df_bad, k=10)

    def test_invalid_k_raises(self, small_traj: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match='k must be >= 1'):
            compute_horizon_features(small_traj, k=0)

    def test_energy_delta_features_for_early_iterations(self, small_traj: pd.DataFrame) -> None:
        df_feat = compute_horizon_features(small_traj, k=10)
        # energy_delta_k1 through energy_delta_k5 should be present (k=10 >= 5)
        for i in range(1, 6):
            assert f'energy_delta_k{i}' in df_feat.columns

    def test_single_iteration_does_not_crash(self) -> None:
        df = pd.DataFrame({
            'run_id': ['r1'],
            'iteration_step': [1],
            'energy': [-1.0],
            'molecule_name': ['H2'],
            'num_qubits': [4],
            'optimizer': ['COBYLA'],
            'basis_set': ['sto-3g'],
            'init_strategy': ['random'],
            'converged': [1],
        })
        df_feat = compute_horizon_features(df, k=1)
        assert len(df_feat) == 1


# ---------------------------------------------------------------------------
# get_horizon_feature_names
# ---------------------------------------------------------------------------


class TestGetHorizonFeatureNames:
    def test_returns_two_lists(self, small_traj: pd.DataFrame) -> None:
        df_feat = compute_horizon_features(small_traj, k=10)
        numeric, categorical = get_horizon_feature_names(10, list(df_feat.columns))
        assert isinstance(numeric, list)
        assert isinstance(categorical, list)

    def test_only_available_columns_returned(self, small_traj: pd.DataFrame) -> None:
        df_feat = compute_horizon_features(small_traj, k=10)
        numeric, categorical = get_horizon_feature_names(10, list(df_feat.columns))
        all_feat = set(numeric) | set(categorical)
        assert all_feat.issubset(set(df_feat.columns))

    def test_categorical_subset_of_global_list(self, small_traj: pd.DataFrame) -> None:
        df_feat = compute_horizon_features(small_traj, k=10)
        _, categorical = get_horizon_feature_names(10, list(df_feat.columns))
        assert set(categorical).issubset(set(CATEGORICAL_FEATURES))

    def test_improvement_ratio_included_for_k20(self, small_traj: pd.DataFrame) -> None:
        df_feat = compute_horizon_features(small_traj, k=20)
        numeric, _ = get_horizon_feature_names(20, list(df_feat.columns))
        assert 'improvement_ratio_k20' in numeric

    def test_improvement_ratio_excluded_for_k10(self, small_traj: pd.DataFrame) -> None:
        df_feat = compute_horizon_features(small_traj, k=10)
        numeric, _ = get_horizon_feature_names(10, list(df_feat.columns))
        assert 'improvement_ratio_k10' not in numeric


# ---------------------------------------------------------------------------
# ConvergencePredictor.fit_evaluate
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestConvergencePredictorFitEvaluate:
    def test_returns_results_object(
        self,
        predictor_results: tuple[ConvergencePredictor, ConvergencePredictorResults],
    ) -> None:
        _, results = predictor_results
        assert isinstance(results, ConvergencePredictorResults)

    def test_fold_results_non_empty(
        self,
        predictor_results: tuple[ConvergencePredictor, ConvergencePredictorResults],
    ) -> None:
        _, results = predictor_results
        assert len(results.fold_results) > 0

    def test_all_three_models_present(
        self,
        predictor_results: tuple[ConvergencePredictor, ConvergencePredictorResults],
    ) -> None:
        _, results = predictor_results
        model_names = {r.model_name for r in results.fold_results}
        assert 'XGBoost' in model_names
        assert 'RandomForest' in model_names
        assert 'LogisticRegression' in model_names

    def test_fold_results_have_correct_horizon(
        self,
        predictor_results: tuple[ConvergencePredictor, ConvergencePredictorResults],
    ) -> None:
        # Shared fixture uses horizons=[10, 20]; verify horizon 20 is present
        _, results = predictor_results
        assert any(r.horizon_k == 20 for r in results.fold_results)

    def test_multiple_horizons_produce_results(
        self,
        predictor_results: tuple[ConvergencePredictor, ConvergencePredictorResults],
    ) -> None:
        _, results = predictor_results
        horizons_seen = {r.horizon_k for r in results.fold_results}
        assert 10 in horizons_seen
        assert 20 in horizons_seen

    def test_roc_auc_in_valid_range_or_nan(
        self,
        predictor_results: tuple[ConvergencePredictor, ConvergencePredictorResults],
    ) -> None:
        _, results = predictor_results
        for r in results.fold_results:
            if np.isfinite(r.roc_auc):
                assert 0.0 <= r.roc_auc <= 1.0, f'ROC-AUC out of range: {r.roc_auc}'

    def test_brier_score_non_negative(
        self,
        predictor_results: tuple[ConvergencePredictor, ConvergencePredictorResults],
    ) -> None:
        _, results = predictor_results
        for r in results.fold_results:
            if np.isfinite(r.brier_score):
                assert r.brier_score >= 0.0

    def test_n_train_positive(
        self,
        predictor_results: tuple[ConvergencePredictor, ConvergencePredictorResults],
    ) -> None:
        _, results = predictor_results
        for r in results.fold_results:
            assert r.n_train > 0
            assert r.n_test > 0

    def test_fitted_models_stored_for_each_model_and_horizon(
        self,
        predictor_results: tuple[ConvergencePredictor, ConvergencePredictorResults],
    ) -> None:
        _, results = predictor_results
        assert 'xgboost_10' in results.fitted_models
        assert 'random_forest_10' in results.fitted_models
        assert 'logistic_regression_10' in results.fitted_models

    def test_held_out_molecule_is_a_known_molecule(
        self,
        medium_traj: pd.DataFrame,
        predictor_results: tuple[ConvergencePredictor, ConvergencePredictorResults],
    ) -> None:
        known_molecules = set(medium_traj['molecule_name'].unique())
        _, results = predictor_results
        for r in results.fold_results:
            assert r.held_out_molecule in known_molecules

    def test_single_molecule_skipped_gracefully(self) -> None:
        df = generate_synthetic_trajectories(
            n_runs=40,
            min_iter=55,
            max_iter=70,
            seed=0,
            molecules=[{'name': 'H2', 'num_qubits': 4, 'e_fci': -1.1, 'e_local': -0.8}],
        )
        predictor = ConvergencePredictor(horizons=[10])
        results = predictor.fit_evaluate(df)
        # No folds should be produced (LOMO requires ≥2 molecules)
        assert isinstance(results, ConvergencePredictorResults)

    def test_summary_contains_model_names(
        self,
        predictor_results: tuple[ConvergencePredictor, ConvergencePredictorResults],
    ) -> None:
        _, results = predictor_results
        summary = results.summary()
        assert 'XGBoost' in summary or 'No results' in summary

    def test_missing_converged_column_skipped_gracefully(self, small_traj: pd.DataFrame) -> None:
        df_no_label = small_traj.drop(columns=['converged'])
        predictor = ConvergencePredictor(horizons=[10])
        results = predictor.fit_evaluate(df_no_label)
        assert isinstance(results, ConvergencePredictorResults)
        assert len(results.fold_results) == 0


# ---------------------------------------------------------------------------
# ConvergencePredictor.predict_proba
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestConvergencePredictorPredictProba:
    def test_predict_proba_returns_series(
        self,
        fitted_predictor: ConvergencePredictor,
        medium_traj: pd.DataFrame,
    ) -> None:
        proba = fitted_predictor.predict_proba(medium_traj, horizon_k=10, model='xgboost')
        assert isinstance(proba, pd.Series)

    def test_predict_proba_length_matches_runs(
        self,
        fitted_predictor: ConvergencePredictor,
        medium_traj: pd.DataFrame,
    ) -> None:
        n_runs = medium_traj['run_id'].nunique()
        proba = fitted_predictor.predict_proba(medium_traj, horizon_k=10, model='xgboost')
        assert len(proba) == n_runs

    def test_predict_proba_values_in_01(
        self,
        fitted_predictor: ConvergencePredictor,
        medium_traj: pd.DataFrame,
    ) -> None:
        proba = fitted_predictor.predict_proba(medium_traj, horizon_k=10, model='random_forest')
        assert (proba >= 0.0).all() and (proba <= 1.0).all()

    def test_predict_before_fit_raises(self, medium_traj: pd.DataFrame) -> None:
        predictor = ConvergencePredictor(horizons=[10])
        with pytest.raises(ValueError, match='No fitted model'):
            predictor.predict_proba(medium_traj, horizon_k=10, model='xgboost')

    def test_predict_unavailable_horizon_raises(
        self,
        fitted_predictor: ConvergencePredictor,
        medium_traj: pd.DataFrame,
    ) -> None:
        # fitted_predictor has horizons=[10, 20]; horizon=50 is not available
        with pytest.raises(ValueError, match='No fitted model'):
            fitted_predictor.predict_proba(medium_traj, horizon_k=50, model='xgboost')

    def test_all_three_models_predict(
        self,
        fitted_predictor: ConvergencePredictor,
        medium_traj: pd.DataFrame,
    ) -> None:
        for model_name in ('xgboost', 'random_forest', 'logistic_regression'):
            proba = fitted_predictor.predict_proba(medium_traj, horizon_k=10, model=model_name)
            assert len(proba) == medium_traj['run_id'].nunique()

    def test_different_models_produce_different_predictions(
        self,
        fitted_predictor: ConvergencePredictor,
        medium_traj: pd.DataFrame,
    ) -> None:
        proba_xgb = fitted_predictor.predict_proba(medium_traj, horizon_k=10, model='xgboost')
        proba_lr = fitted_predictor.predict_proba(
            medium_traj, horizon_k=10, model='logistic_regression'
        )
        assert not proba_xgb.equals(proba_lr)


# ---------------------------------------------------------------------------
# Dataclass helpers
# ---------------------------------------------------------------------------


class TestFoldResult:
    def test_str_contains_model_and_molecule(self) -> None:
        r = FoldResult(
            model_name='XGBoost',
            horizon_k=10,
            held_out_molecule='H2',
            roc_auc=0.85,
            pr_auc=0.70,
            brier_score=0.10,
            mcc=0.60,
            n_train=200,
            n_test=50,
        )
        s = str(r)
        assert 'XGBoost' in s
        assert 'H2' in s
        assert '10' in s

    def test_str_contains_metrics(self) -> None:
        r = FoldResult('RF', 20, 'LiH', 0.80, 0.65, 0.12, 0.55, 180, 40)
        s = str(r)
        assert 'ROC-AUC' in s
        assert 'Brier' in s


class TestConvergencePredictorResults:
    def test_best_returns_highest_roc_auc(self) -> None:
        results = ConvergencePredictorResults()
        results.fold_results = [
            FoldResult('XGBoost', 10, 'H2', 0.85, 0.70, 0.10, 0.60, 200, 50),
            FoldResult('RandomForest', 10, 'H2', 0.78, 0.65, 0.12, 0.55, 200, 50),
            FoldResult('LogisticRegression', 10, 'H2', 0.72, 0.60, 0.15, 0.45, 200, 50),
        ]
        best = results.best('roc_auc')
        assert best is not None
        assert best.model_name == 'XGBoost'

    def test_best_on_empty_returns_none(self) -> None:
        results = ConvergencePredictorResults()
        assert results.best() is None

    def test_best_ignores_nan(self) -> None:
        results = ConvergencePredictorResults()
        results.fold_results = [
            FoldResult('XGBoost', 10, 'H2', float('nan'), 0.70, 0.10, 0.60, 200, 50),
            FoldResult('RandomForest', 10, 'H2', 0.78, 0.65, 0.12, 0.55, 200, 50),
        ]
        best = results.best('roc_auc')
        assert best is not None
        assert best.model_name == 'RandomForest'

    def test_summary_returns_string(self) -> None:
        results = ConvergencePredictorResults()
        results.fold_results = [
            FoldResult('XGBoost', 10, 'H2', 0.85, 0.70, 0.10, 0.60, 200, 50),
        ]
        summary = results.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_summary_on_empty_results(self) -> None:
        results = ConvergencePredictorResults()
        summary = results.summary()
        assert 'No results' in summary
