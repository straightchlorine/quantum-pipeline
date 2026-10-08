"""Feature extraction for the convergence predictor.

Collapses iteration-level rows into one row per run using only the first K iterations.
The `iteration_step <= k` filter is what enforces no-lookahead: any feature that read
past K would let the model cheat and make reported accuracy meaningless.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from quantum_pipeline.ml.schema import RUN_KEY, require_columns

logger = logging.getLogger(__name__)

HORIZONS = [10, 20, 50]

CATEGORICAL_FEATURES = ['optimizer', 'basis_set']

# Features constant for the entire run.
RUN_LEVEL_FEATURES = [
    # stand-in for problem difficulty; hypothesis, not a measured cost
    'num_qubits',
    # 1 if random init, 0 if HF
    'init_strategy_random',
    # tests the hypothesis that random init hurts more as qubit count grows
    'qubit_x_random',
    # average distance the params moved per step
    'mean_param_delta_norm',
]

# Expanded into pairwise products, so the linear baseline can weigh them together.
# --
# A flat energy curve is ambiguous alone - converged, or stuck on a barren plateau.
# It only reads as a plateau when the run is also wide and randomly initialised.
# Tree models reach the same combinations by splitting, so they skip this.
INTERACTION_FEATURES = [
    'energy_moving_std_5_at_k',
    'num_qubits',
    'init_strategy_random',
]


def _spark_stddev(window: np.ndarray) -> float:
    """Sample std (ddof=1) of `window`, NaN when it has fewer than two rows.

    Matches Spark's `stddev` over the rolling window in `ml_iteration_features`.
    """
    return float(np.std(window, ddof=1)) if len(window) > 1 else float('nan')


def compute_horizon_features(
    df_iter: pd.DataFrame, k: int, require_full_horizon: bool = True
) -> pd.DataFrame:
    """Collapse iteration-level rows into one feature row per run.

    Reads no iteration past K, so the features stay usable as an early-abort signal.

    Runs shorter than K are dropped. The stagnation features are read from the row at
    step K, which such a run never reaches, so the only alternative is to substitute a
    value - and substituting the worst one labels a run that finished early as the most
    stalled. Backwards, and it smuggles in run length, which already tracks the label.

    Args:
        df_iter: Iteration-level rows from `ml_iteration_features`. Needs
            experiment_id, iteration_step and energy; the other feature columns
            fall back to recomputation or a neutral default when absent.
        k: Prediction horizon in iterations, >= 1.
        require_full_horizon: Set False to keep short runs, substituting their missing
            horizon features instead of dropping the run.

    Returns:
        One row per experiment_id, plus `converged` when df_iter carries it. Empty
        when no run reaches K.

    Raises:
        ValueError: If a required column is missing or k < 1.
    """
    require_columns(df_iter.columns, {RUN_KEY, 'iteration_step', 'energy'}, 'df_iter')
    if k < 1:
        raise ValueError(f'k must be >= 1, got {k}')

    # The no-lookahead cut. Everything below reads only these rows, so no feature can
    # see the outcome it is meant to predict. copy() keeps the later filter off a view.
    within_horizon = df_iter[df_iter['iteration_step'] <= k].copy()

    if require_full_horizon:
        # max step survives the truncation above, so it tells us how far the run got
        steps_per_run = within_horizon.groupby(RUN_KEY)['iteration_step'].max()
        complete = steps_per_run[steps_per_run >= k].index
        dropped = len(steps_per_run) - len(complete)
        if dropped:
            logger.info('K=%d: dropping %d run(s) shorter than the horizon', k, dropped)
        within_horizon = within_horizon[within_horizon[RUN_KEY].isin(complete)]

    feature_rows: list[dict] = []

    # move through rows within experiments, fulfilling the horizon condition
    for experiment_id, run_rows in within_horizon.groupby(RUN_KEY):
        # Sorting, since the table has no ordering guarantee;
        # every feature blow thus assumes chronological rows.
        run_rows = run_rows.sort_values('iteration_step').reset_index(drop=True)
        n_steps = len(run_rows)
        energies = run_rows['energy'].values.astype(float)

        features: dict[str, Any] = {RUN_KEY: experiment_id}

        # How much energy each of the first five steps gained, as five features.
        # Should convey the shape of the opening, apart from the average.
        for i in range(1, min(6, k + 1)):
            step_row = run_rows[run_rows['iteration_step'] == i]

            # feature not built if lack of column
            # or run is shorter than i
            if len(step_row) > 0 and 'energy_delta' in run_rows.columns:
                features[f'energy_delta_k{i}'] = float(step_row['energy_delta'].iloc[0])
            else:
                # NaN, not 0.0 - a literal 0.0 would claim no energy change
                features[f'energy_delta_k{i}'] = float('nan')

        # Descent rate across the entire interval.
        # Approach is fitting a straight line to the energies and keeping its gradient.
        # --
        # Negative means still improving, near zero flat.
        # Since x is step position, unit is Hartree/step
        if n_steps >= 2:
            x = np.arange(n_steps, dtype=float)
            features[f'energy_slope_first{k}'] = float(np.polyfit(x, energies, 1)[0])
        else:
            # One point defines no line; 0.0 says "no trend", which is what we know.
            features[f'energy_slope_first{k}'] = 0.0

        # How much energy varied over the final five steps, as a single feature.
        # --
        # Near zero means flat (converged/plateau).
        # INTERACTION_FEATURES pairings are used to separate the two.
        if 'energy_moving_std_5' in run_rows.columns:
            # spark calculates this per row (rolling), value at the horizon
            # represents the final window
            horizon_row = run_rows[run_rows['iteration_step'] == k]
            if len(horizon_row) > 0:
                features['energy_moving_std_5_at_k'] = float(
                    horizon_row['energy_moving_std_5'].iloc[-1]
                )
            else:
                # didn't reach step k, can happen if require_full_horizon=False
                # likely won't be used - handling computes over what the tail has
                features['energy_moving_std_5_at_k'] = _spark_stddev(energies[-5:])
        else:
            # Column absent, so recompute it: sample std (n-1) over the last five
            # rows, NaN on a one-row window (k=1).
            # The NaN is what the column holds there too, and the median imputer
            # fills it exactly as it does in training.
            features['energy_moving_std_5_at_k'] = _spark_stddev(energies[-5:])

        # How long the run had gone without a new best energy by step K.
        # --
        # A snapshot at K, so a stall earlier in the window that has since recovered
        # leaves no trace here. longest_plateau_first{k} below is what catches those.
        if 'steps_since_improvement' in run_rows.columns:
            horizon_row = run_rows[run_rows['iteration_step'] == k]
            if len(horizon_row) > 0:
                features[f'steps_since_improvement_at_k{k}'] = float(
                    horizon_row['steps_since_improvement'].iloc[-1]
                )
            else:
                features[f'steps_since_improvement_at_k{k}'] = float(
                    k
                )  # no row at K: substitute the worst value
        else:
            features[f'steps_since_improvement_at_k{k}'] = float(k)

        # Worst stall in the entire interval.
        # --
        # Defined as the longest, unbroken run of steps that failed to set a
        # new minimum.
        # This registers a run that froze and recovered.
        if 'is_new_minimum' in run_rows.columns:
            is_new_min = run_rows['is_new_minimum'].values
            max_plateau = 0
            run_len = 0
            for val in is_new_min:
                if val == 0:
                    run_len += 1
                    max_plateau = max(max_plateau, run_len)
                else:
                    run_len = 0  # a new minimum ends the current stretch
            features[f'longest_plateau_first{k}'] = float(max_plateau)
        else:
            features[f'longest_plateau_first{k}'] = float(k)

        # How far the circuit parameters travelled per step.
        # --
        # Mean step size and how it varied.
        #
        # The first step has no previous point, so `parameter_delta_norm` is null there
        # and the float cast makes it NaN. Plain `np.mean` / `np.std` would propagate that
        # NaN into every run and leave the imputer a dead column; the nan-aware versions
        # skip it. A window with no measured step (k=1) stays NaN, and the count guard
        # avoids numpy's empty-slice RuntimeWarning.
        #
        # Hypothesis: a stalled optimizer shows small, even steps.
        # TODO: prove/disprove
        if 'parameter_delta_norm' in run_rows.columns:
            param_deltas = run_rows['parameter_delta_norm'].values.astype(float)
            n_measured = int(np.count_nonzero(~np.isnan(param_deltas)))
            features[f'param_delta_norm_mean_k{k}'] = (
                float(np.nanmean(param_deltas)) if n_measured else float('nan')
            )
            if n_measured == 0:
                features[f'param_delta_norm_std_k{k}'] = float('nan')
            else:
                features[f'param_delta_norm_std_k{k}'] = (
                    float(np.nanstd(param_deltas)) if n_measured > 1 else 0.0
                )
        else:
            features[f'param_delta_norm_mean_k{k}'] = 0.0
            features[f'param_delta_norm_std_k{k}'] = 0.0

        # Share of the window's improvement that happened in the first half.
        # --
        # Run that dropped early then was flat is 1.0.
        # Steady descent ~0.5.
        #
        # Omitted below K=20, where each half holds fewer than 10 steps and its sum
        # is highly prone to noise. A chosen floor, not a derived one.
        if k >= 20:
            if 'energy_delta' in run_rows.columns:
                mid = k // 2
                first_half = float(
                    run_rows[run_rows['iteration_step'] <= mid]['energy_delta'].sum()
                )
                second_half = float(
                    run_rows[run_rows['iteration_step'] > mid]['energy_delta'].sum()
                )

                # net change across the whole window
                # if run came back to it's starting energy, it's ~0
                # use 0.5 to represent a steady descent (division would cause an extreme ratio)
                total = first_half + second_half
                features[f'improvement_ratio_k{k}'] = (
                    first_half / total if abs(total) > 1e-9 else 0.5
                )
            else:
                features[f'improvement_ratio_k{k}'] = 0.5  # neutral default

        # constant for the whole run, so any row carries it
        for col in ('molecule_name', 'num_qubits', 'optimizer', 'basis_set', 'init_strategy'):
            if col in run_rows.columns:
                features[col] = run_rows[col].iloc[0]

        if 'parameter_delta_norm' in run_rows.columns:
            features['mean_param_delta_norm'] = float(run_rows['parameter_delta_norm'].mean())
        else:
            features['mean_param_delta_norm'] = 0.0

        # 'random' is the current pipeline default
        # TODO: measure performance impact and consider change to HF init
        init_random = 1 if str(features.get('init_strategy', 'random')).lower() == 'random' else 0
        features['init_strategy_random'] = float(init_random)
        features['qubit_x_random'] = float(features.get('num_qubits', 0)) * float(init_random)

        # the target, carried through only when training data was passed
        if 'converged' in run_rows.columns:
            features['converged'] = int(run_rows['converged'].iloc[0])

        feature_rows.append(features)

    return pd.DataFrame(feature_rows)


def get_horizon_feature_names(k: int, available_cols: list[str]) -> tuple[list[str], list[str]]:
    """Return (numeric_features, categorical_features) for a given horizon K.

    Only includes features that are present in available_cols.
    """
    numeric: list[str] = list(RUN_LEVEL_FEATURES)

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
