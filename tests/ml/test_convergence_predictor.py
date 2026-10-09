"""Tests for quantum_pipeline.ml.convergence."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import pytest

from quantum_pipeline.ml.convergence.features import (
    CATEGORICAL_FEATURES,
    HORIZONS,
    compute_horizon_features,
    get_horizon_feature_names,
)
from quantum_pipeline.ml.convergence.predictor import ConvergencePredictor
from quantum_pipeline.ml.convergence.results import (
    ConvergencePredictorResults,
    FoldResult,
)
from quantum_pipeline.ml.convergence.synthetic import generate_synthetic_trajectories
from quantum_pipeline.ml.schema import FIRST_ITERATION

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
def predictor_results(
    medium_traj: pd.DataFrame,
) -> tuple[ConvergencePredictor, ConvergencePredictorResults]:
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
            'experiment_id',
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
        min_step = small_traj.groupby('experiment_id')['iteration_step'].min()
        assert (min_step == 1).all(), 'iteration_step must be 1-indexed'

    def test_cumulative_min_is_monotone_non_increasing(self, small_traj: pd.DataFrame) -> None:
        for _, grp in small_traj.groupby('experiment_id'):
            grp = grp.sort_values('iteration_step')
            cummin = grp['cumulative_min_energy'].values
            assert np.all(np.diff(cummin) <= 1e-9), 'cumulative_min_energy must be non-increasing'

    def test_energy_values_negative(self, small_traj: pd.DataFrame) -> None:
        assert (small_traj['energy'] < 0).all()

    def test_converged_is_binary(self, small_traj: pd.DataFrame) -> None:
        assert set(small_traj['converged'].unique()).issubset({0, 1})

    def test_step_one_nulls_match_the_real_table(self, small_traj: pd.DataFrame) -> None:
        """Spark/VQESolver null these three on step 1 (no previous row); nothing else."""
        first_steps = small_traj['iteration_step'] == FIRST_ITERATION
        step_one_null = ['energy_delta', 'energy_moving_std_5', 'parameter_delta_norm']

        for col in step_one_null:
            assert small_traj.loc[first_steps, col].isnull().all(), col
            assert small_traj.loc[~first_steps, col].notnull().all(), col
        assert not small_traj.drop(columns=step_one_null).isnull().any().any()

    def test_moving_std_is_the_sample_std(self) -> None:
        """Spark `stddev` divides by n-1: the second row of a window is |e1 - e0| / sqrt(2)."""
        df = generate_synthetic_trajectories(n_runs=5, seed=0)
        run = df[df['experiment_id'] == df['experiment_id'].iloc[0]]
        e = run['energy'].values
        expected = abs(e[1] - e[0]) / np.sqrt(2)

        assert np.isclose(run['energy_moving_std_5'].iloc[1], expected)

    def test_iteration_range_valid_below_the_floor(self) -> None:
        """min_iter and max_iter both under 10 floor at 10 steps instead of raising."""
        df = generate_synthetic_trajectories(n_runs=5, min_iter=3, max_iter=5, seed=0)

        assert (df.groupby('experiment_id')['iteration_step'].count() == 10).all()

    def test_optimizer_basis_init_are_not_collinear(self) -> None:
        """Every optimizer x basis x init combination occurs, not just a round-robin few."""
        df = generate_synthetic_trajectories(n_runs=400, seed=0)
        runs = df.drop_duplicates('experiment_id')
        combos = runs.groupby(['optimizer', 'basis_set', 'init_strategy']).ngroups

        assert combos == 4 * 2 * 2

    def test_random_init_probability_floor_is_final(self) -> None:
        """The 0.15 floor bounds random-init N2 too: 0.10 x 0.65 would otherwise sit near 0.1."""
        mol = {'name': 'Wide', 'num_qubits': 40, 'e_fci': -1.0, 'e_local': -0.7}
        df = generate_synthetic_trajectories(n_runs=4000, molecules=[mol], seed=0)
        runs = df.drop_duplicates('experiment_id')
        rate = runs.loc[runs['init_strategy'] == 'random', 'converged'].mean()

        assert 0.12 < rate < 0.18

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
        min_steps = df.groupby('experiment_id')['iteration_step'].count().min()
        assert min_steps >= 55, f'Expected ≥55 iterations per run, got {min_steps}'

    def test_custom_molecules(self) -> None:
        mols = [
            {'name': 'TestMol', 'num_qubits': 4, 'e_fci': -1.0, 'e_local': -0.7},
        ]
        df = generate_synthetic_trajectories(n_runs=4, molecules=mols, seed=0)
        assert (df['molecule_name'] == 'TestMol').all()

    def test_mean_param_delta_norm_constant_per_run(self, small_traj: pd.DataFrame) -> None:
        for _, grp in small_traj.groupby('experiment_id'):
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
        n_runs = small_traj['experiment_id'].nunique()
        df_feat = compute_horizon_features(small_traj, k=10)
        assert len(df_feat) == n_runs

    def test_experiment_id_column_present(self, small_traj: pd.DataFrame) -> None:
        df_feat = compute_horizon_features(small_traj, k=10)
        assert 'experiment_id' in df_feat.columns

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
        for col in (
            'num_qubits',
            'init_strategy_random',
            'qubit_x_random',
            'mean_param_delta_norm',
        ):
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
        df_bad = pd.DataFrame({'experiment_id': ['a'], 'energy': [-1.0]})
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
        df = pd.DataFrame(
            {
                'experiment_id': ['r1'],
                'iteration_step': [1],
                'energy': [-1.0],
                'molecule_name': ['H2'],
                'num_qubits': [4],
                'optimizer': ['COBYLA'],
                'basis_set': ['sto3g'],
                'init_strategy': ['random'],
                'converged': [1],
            }
        )
        df_feat = compute_horizon_features(df, k=1)
        assert len(df_feat) == 1


class TestParamDeltaNullFirstStep:
    """Step 1 has no previous point, so the real table holds None there."""

    @staticmethod
    def _run(k: int) -> pd.DataFrame:
        steps = list(range(FIRST_ITERATION, FIRST_ITERATION + k))
        return pd.DataFrame(
            {
                'experiment_id': ['r1'] * k,
                'iteration_step': steps,
                'energy': [-1.0 - 0.01 * i for i in range(k)],
                'parameter_delta_norm': [None, *[0.5, 0.25, 0.125, 0.0625][: k - 1]],
                'molecule_name': ['H2'] * k,
                'num_qubits': [4] * k,
                'optimizer': ['COBYLA'] * k,
                'basis_set': ['sto3g'] * k,
                'init_strategy': ['random'] * k,
                'converged': [1] * k,
            }
        )

    def test_null_first_step_is_skipped(self) -> None:
        df_feat = compute_horizon_features(self._run(5), k=5)
        assert df_feat['param_delta_norm_mean_k5'].iloc[0] == pytest.approx(0.234375)
        assert np.isfinite(df_feat['param_delta_norm_std_k5'].iloc[0])

    def test_no_measured_step_is_nan_without_warning(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            df_feat = compute_horizon_features(self._run(1), k=1)
        assert np.isnan(df_feat['param_delta_norm_mean_k1'].iloc[0])
        assert np.isnan(df_feat['param_delta_norm_std_k1'].iloc[0])


class TestShortRunHandling:
    """Runs shorter than K must not be featurised as if they had stalled."""

    @staticmethod
    def _run(experiment_id: str, n_steps: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                'experiment_id': [experiment_id] * n_steps,
                'iteration_step': range(FIRST_ITERATION, n_steps + FIRST_ITERATION),
                'energy': np.linspace(-1.0, -1.5, n_steps),
                'molecule_name': ['H2'] * n_steps,
                'num_qubits': [4] * n_steps,
                'converged': [1] * n_steps,
            }
        )

    def test_short_runs_are_dropped(self) -> None:
        df = pd.concat([self._run('long', 30), self._run('short', 5)], ignore_index=True)

        df_feat = compute_horizon_features(df, k=20)

        assert df_feat['experiment_id'].tolist() == ['long']

    def test_all_runs_shorter_than_horizon_gives_empty_frame(self) -> None:
        df_feat = compute_horizon_features(self._run('short', 5), k=50)

        assert df_feat.empty

    def test_opt_out_keeps_short_runs(self) -> None:
        df = pd.concat([self._run('long', 30), self._run('short', 5)], ignore_index=True)

        df_feat = compute_horizon_features(df, k=20, require_full_horizon=False)

        assert sorted(df_feat['experiment_id']) == ['long', 'short']


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
    def test_horizon_beyond_every_run_warns_about_length(
        self, small_traj: pd.DataFrame, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            ConvergencePredictor(horizons=[10_000]).fit_evaluate(small_traj)
        assert 'No run reaches K=10000 steps' in caplog.text
        assert 'No target column' not in caplog.text

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

    @pytest.mark.parametrize('label', [0, 1])
    def test_single_outcome_horizon_skipped_not_raised(
        self, small_traj: pd.DataFrame, label: int
    ) -> None:
        df = small_traj.copy()
        df['converged'] = label
        predictor = ConvergencePredictor(horizons=[10, 20])
        results = predictor.fit_evaluate(df)
        assert results.fold_results == []
        assert results.fitted_models == {}

    def test_single_outcome_horizon_does_not_abandon_later_horizons(
        self, small_traj: pd.DataFrame
    ) -> None:
        # Horizon 60 keeps only the runs of at least 60 iterations; label those all 1, so
        # that horizon is single-class while horizon 10 (every run) stays mixed.
        df = small_traj.copy()
        n_iter = df.groupby('experiment_id')['iteration_step'].transform('count')
        long_run = n_iter >= 60
        assert df.loc[long_run, 'experiment_id'].nunique() >= 10
        parity = df.groupby('experiment_id').ngroup() % 2
        df['converged'] = np.where(long_run, 1, parity)
        results = ConvergencePredictor(horizons=[60, 10]).fit_evaluate(df)
        assert {r.horizon_k for r in results.fold_results} == {10}
        assert 'xgboost_10' in results.fitted_models

    def test_default_horizons_not_aliased(self) -> None:
        a = ConvergencePredictor()
        a.horizons.append(99)
        assert 99 not in ConvergencePredictor().horizons
        assert 99 not in HORIZONS

    def test_empty_horizons_kept(self) -> None:
        assert ConvergencePredictor(horizons=[]).horizons == []

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
        n_runs = medium_traj['experiment_id'].nunique()
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

    def test_predict_horizon_beyond_every_run_raises(
        self, fitted_predictor: ConvergencePredictor, medium_traj: pd.DataFrame
    ) -> None:
        short = medium_traj[medium_traj['iteration_step'] < 5]
        with pytest.raises(ValueError, match='No run in df_traj reaches horizon K=10'):
            fitted_predictor.predict_proba(short, horizon_k=10, model='xgboost')

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
            assert len(proba) == medium_traj['experiment_id'].nunique()

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


class TestMovingStdFallback:
    """The recomputed `energy_moving_std_5_at_k` must equal the Spark column's value."""

    @pytest.mark.parametrize('k', [1, 2, 3, 5, 10, 20])
    def test_fallback_equals_column(self, small_traj: pd.DataFrame, k: int) -> None:
        with_col = compute_horizon_features(small_traj, k=k)
        without_col = compute_horizon_features(
            small_traj.drop(columns=['energy_moving_std_5']), k=k
        )
        np.testing.assert_allclose(
            without_col['energy_moving_std_5_at_k'].to_numpy(),
            with_col['energy_moving_std_5_at_k'].to_numpy(),
            equal_nan=True,
        )
        # k=1 is a one-row window: NaN in the column, so NaN (not 0.0) in the fallback
        assert with_col['energy_moving_std_5_at_k'].isna().all() == (k == 1)

    def test_fallback_short_run_uses_sample_std_of_tail(self, small_traj: pd.DataFrame) -> None:
        # k beyond every run, so the "never reached step k" branch computes from the tail
        out = compute_horizon_features(
            small_traj.drop(columns=['energy_moving_std_5']), k=1000, require_full_horizon=False
        )
        run = small_traj[small_traj['experiment_id'] == out['experiment_id'].iloc[0]]
        tail = run.sort_values('iteration_step')['energy'].to_numpy()[-5:]
        assert out['energy_moving_std_5_at_k'].iloc[0] == pytest.approx(np.std(tail, ddof=1))
