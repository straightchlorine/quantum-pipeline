"""Feature extraction for the energy estimator.

Collapses a trajectory into one row per run, reading only the first
`ceil(total_steps * completion_frac)` iterations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantum_pipeline.ml.schema import RUN_KEY, require_columns

COMPLETION_FRACS = [0.25, 0.50, 0.75]

# features derived from the trajectory observed up to completion fraction p
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
]

CATEGORICAL_FEATURES = ['optimizer', 'basis_set']

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def extract_features_at_fraction(
    df_traj: pd.DataFrame,
    completion_frac: float,
) -> pd.DataFrame:
    """Extract run-level ML features from trajectories truncated at completion_frac.

    For each experiment, takes only the first `ceil(total_steps * completion_frac)`
    steps and derives summary statistics for the regressor.

    Args:
        df_traj: Trajectory DataFrame. Required columns: experiment_id, iteration_step,
            energy. Optional: cumulative_min_energy, energy_delta, parameter_delta_norm,
            molecule_name, num_qubits, optimizer, basis_set, ansatz_reps, final_energy.
        completion_frac: Fraction in (0, 1]. E.g. 0.25 = first quarter of trajectory.

    Returns:
        One-row-per-experiment feature DataFrame with columns matching ALL_FEATURES
        plus 'experiment_id', 'molecule_name' and 'final_energy'.
    """
    if not 0 < completion_frac <= 1.0:
        raise ValueError(f'completion_frac must be in (0, 1], got {completion_frac}')

    require_columns(df_traj.columns, {RUN_KEY, 'iteration_step', 'energy'}, 'df_traj')

    feature_rows = []

    # move through rows within experiments, constrained by the fraction
    for experiment_id, run_rows in df_traj.groupby(RUN_KEY):
        # sorting, every row below assumes chronological order
        run_rows = run_rows.sort_values('iteration_step').reset_index(drop=True)

        # calculate horizon as a `completion_frac` of it's length
        # take trajectory up to that point
        total_steps = len(run_rows)
        cutoff = max(1, int(np.ceil(total_steps * completion_frac)))
        observed = run_rows.iloc[:cutoff]

        energies = observed['energy'].values.astype(float)
        n_steps = len(energies)

        # What is the lower enery reached within the observed portion.
        if 'cumulative_min_energy' in observed.columns:
            cummin = observed['cumulative_min_energy'].iloc[-1]
        else:
            cummin = float(np.min(energies))

        # energy at horizon
        current_energy = float(energies[-1])

        # What is the direction and the rate of change.
        # --
        # Fitting a straight line through the observed trajectory.
        # Negative slope implies energy decreasing, positive - increasing.
        if n_steps >= 2:
            x = np.arange(n_steps, dtype=float)
            slope, _ = np.polyfit(x, energies, deg=1)
        else:
            slope = 0.0

        # How much lower is best observed energy, compared with the starting one.
        total_improvement = float(energies[0] - cummin)  # positive = improved

        # How much improvement per observed iteration, in energy units.
        energy_improvement_rate = total_improvement / n_steps if n_steps > 0 else 0.0

        # How much does the energy change between consecutive steps. Signed: negative
        # is a step down. The mean is therefore net drift, not typical step size.
        if 'energy_delta' in observed.columns:
            # if available
            deltas = observed['energy_delta'].dropna().values.astype(float)
        elif n_steps >= 2:
            # E_k - E_{k-1}, signed like the `energy_delta` column above
            deltas = np.diff(energies)
        else:
            deltas = np.array([0.0])

        # Mean of the energy changes.
        energy_delta_mean = float(np.mean(deltas)) if len(deltas) > 0 else 0.0

        # Standard deviation of energy changes.
        energy_delta_std = float(np.std(deltas)) if len(deltas) > 1 else 0.0

        # Defining standard deviation of the last 5 steps as a plateu signal.
        # --
        # The module's own feature, always computed here and never read from the
        # Spark `energy_moving_std_5` column, so it keeps numpy's population std (n)
        # and 0.0 on a one-row window; training and prediction share this definition.
        window = min(5, n_steps)
        energy_moving_std = float(np.std(energies[-window:])) if window > 1 else 0.0

        # How many steps have passed since th elast new lowest energy was found.
        running_min = energies[0]
        steps_since_imp = 0
        for val in energies:
            if val < running_min - 1e-8:
                running_min = val
                steps_since_imp = 0
            else:
                steps_since_imp += 1

        # collect metadata
        num_qubits = int(run_rows['num_qubits'].iloc[0]) if 'num_qubits' in run_rows.columns else 0
        ansatz_reps = (
            int(run_rows['ansatz_reps'].iloc[0]) if 'ansatz_reps' in run_rows.columns else 1
        )
        optimizer = (
            str(run_rows['optimizer'].iloc[0]) if 'optimizer' in run_rows.columns else 'UNKNOWN'
        )
        basis_set = (
            str(run_rows['basis_set'].iloc[0]) if 'basis_set' in run_rows.columns else 'sto3g'
        )
        molecule_name = (
            str(run_rows['molecule_name'].iloc[0])
            if 'molecule_name' in run_rows.columns
            else 'UNKNOWN'
        )

        # Run's final energy, or the minimum observed as a fallback.
        # Defined as prediction target.
        final_energy: float | None = None
        if 'final_energy' in run_rows.columns:
            final_energy = float(run_rows['final_energy'].iloc[0])
        elif 'minimum_energy' in observed.columns:
            final_energy = float(observed['minimum_energy'].iloc[-1])

        feature_rows.append(
            {
                RUN_KEY: experiment_id,
                'molecule_name': molecule_name,
                'cumulative_min_energy': cummin,
                'current_energy': current_energy,
                'energy_slope': float(slope),
                'energy_improvement_rate': energy_improvement_rate,
                'energy_delta_mean': energy_delta_mean,
                'energy_delta_std': energy_delta_std,
                'energy_moving_std': energy_moving_std,
                'steps_since_improvement': float(steps_since_imp),
                'num_steps_observed': float(n_steps),
                'num_qubits': float(num_qubits),
                'ansatz_reps': float(ansatz_reps),
                'optimizer': optimizer,
                'basis_set': basis_set,
                'final_energy': final_energy,
            }
        )

    return pd.DataFrame(feature_rows)
