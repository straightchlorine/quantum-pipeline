"""Synthetic convergence trajectories for development and tests.

Not production code: the curves are an exponential-decay caricature of VQE descent,
shaped to exercise the horizon features and the leave-one-molecule-out cross-validation
(each fold holds one molecule out) without a pipeline run.

A horizon K is the number of opening steps a feature may see: `compute_horizon_features`
builds one row per run from the steps up to K only (it reads no step past K).

The frame mimics the Iceberg table `ml_iteration_features` (one row per VQE iteration,
written by the Spark job `docker/airflow/scripts/quantum_ml_feature_processing.py`)
closely enough for `compute_horizon_features` and `ConvergencePredictor.fit_evaluate` to
run on it. The known gaps are listed at the end of the `generate_synthetic_trajectories`
docstring.

The descent curve itself comes from `quantum_pipeline.ml.trajectory.simulate_descent`,
shared with the energy generator. This module adds what the convergence model needs on
top of it: the outcome label and the per-step columns the Spark job derives with window
functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantum_pipeline.ml.schema import FIRST_ITERATION, RUN_KEY, default_molecules
from quantum_pipeline.ml.trajectory import BASIS_SETS, OPTIMIZERS, simulate_descent


def generate_synthetic_trajectories(
    n_runs: int = 300,
    min_iter: int = 55,
    max_iter: int = 100,
    seed: int = 42,
    molecules: list[dict] | None = None,
) -> pd.DataFrame:
    """Generate VQE-like trajectories with convergence labels, one row per iteration.

    Each run is one curve from `quantum_pipeline.ml.trajectory.simulate_descent`.
    The label is assigned first and the curve drawn to match it.

    In this frame `converged` also decides where the curve ends: `simulate_descent`
    decays a converged run to just above `e_fci` (the full configuration interaction
    energy: the exact ground-state energy within the basis set) and a stalled one to near
    `e_local` (a local minimum above it), so label and final energy level are tied.

    In the real table `converged` is scipy's `OptimizeResult.success`, renamed by the
    Spark job (`docker/airflow/scripts/quantum_ml_feature_processing.py`) from
    `vqe_results.success`: the optimizer met its own stopping criterion.

    This says nothing about the energy being right; a run stuck on a barren
    plateau reports success too. The one exception is that the solver forces it
    False when the pipeline's own evaluation cap aborts the run
    (`VQESolver._make_truncated_result`).

    The tie above is an artefact of how these rows are made, not something to
    expect in real data.

    Args:
        n_runs: Target total. Each molecule gets `max(1, n_runs // len(molecules))`
            runs, so the real total is a multiple of `len(molecules)`: with the five
            default molecules `n_runs=8` gives 5 runs and `n_runs=4` also gives 5.
        min_iter: Minimum iterations per run; the lower bound actually used is
            `max(min_iter, 10)`. Keep it above the largest horizon you intend to test:
            by default `compute_horizon_features` drops each run shorter than K, so with
            `min_iter` below 50 the K=50 tests would lose every run drawn shorter than
            that, and all of them once `max_iter` is below 50 too. `min_iter=55` keeps
            every run.
        max_iter: Maximum iterations per run, inclusive. Below the effective lower bound
            `max(min_iter, 10)` it is ignored and every run gets exactly that many steps.
        seed: Seed for `numpy.random.default_rng`; same seed, same frame.
        molecules: Molecule spec dicts (`name`, `num_qubits`, `e_fci`, `e_local`).
            Defaults to `quantum_pipeline.ml.schema.default_molecules`.

    Returns:
        DataFrame with one row per iteration and the columns: experiment_id,
        molecule_name, num_qubits, optimizer, basis_set, init_strategy, iteration_step
        (1-indexed, `schema.FIRST_ITERATION`), energy, energy_delta (NaN on step 1,
        from the None the solver records),
        energy_moving_avg_5, energy_moving_std_5 (NaN on step 1), cumulative_min_energy,
        steps_since_improvement, is_new_minimum, parameter_delta_norm (NaN on step 1),
        mean_param_delta_norm, converged.

    Where this frame differs from the real table `ml_iteration_features`: `converged` and
    `is_new_minimum` are int 0/1 here and boolean there, which `compute_horizon_features`
    absorbs with `int()` and `== 0`. `mean_param_delta_norm` is not a column of that table
    at all (`ml_run_summary` has it, as a run-level mean). The comments at the lines that
    fill these columns have the detail.
    """
    rng = np.random.default_rng(seed)

    if molecules is None:
        molecules = default_molecules()

    init_strategies = ['random', 'hf']

    all_rows: list[dict] = []
    runs_per_mol = max(1, n_runs // len(molecules))

    for mol in molecules:
        for run_idx in range(runs_per_mol):
            experiment_id = f'syn_{mol["name"]}_{run_idx:04d}'
            # Independent draws, so every combination can occur. A round-robin over
            # these short lists would lock optimizer, basis_set and init_strategy
            # together (collinear columns no model can separate).
            optimizer = str(rng.choice(OPTIMIZERS))
            basis_set = str(rng.choice(BASIS_SETS))
            init_strategy = str(rng.choice(init_strategies))

            # Draw n_iter from low to high inclusive; `integers` excludes its upper bound,
            # hence the +1. `low` carries the 10-step floor, and `high` is kept above it
            # so the interval is never empty: when max_iter < low every run gets exactly
            # `low` steps.
            low = max(min_iter, 10)
            n_iter = int(rng.integers(low, max(max_iter + 1, low + 1)))

            # Qubit term: 0.90 falling by 0.04 per qubit. For the defaults in
            # schema.SYNTHETIC_MOLECULES it gives H2 0.74, LiH 0.42, H2O 0.34, NH3 0.26
            # and N2 0.10.
            base_prob = 0.90 - mol['num_qubits'] * 0.04

            # The random-init penalty is multiplicative, so its absolute size is largest
            # on small molecules: an interaction for `qubit_x_random` to find, but with
            # the opposite sign to the hypothesis in `features.py`.
            if init_strategy == 'random':
                base_prob *= 0.65

            # Floor applied last, so that even the widest molecule converges sometimes
            # whatever the init: it bounds the final probability.
            base_prob = max(0.15, base_prob)

            # Label first, curve second: the reverse of reality (the `simulate_descent`
            # docstring has the consequences).
            converged = bool(rng.random() < base_prob)

            descent = simulate_descent(rng, mol, n_iter, optimizer, converged)
            param_deltas = descent.param_deltas

            # Rebuilds, step by step, the columns that `build_iteration_features`
            # (docker/airflow/scripts/quantum_ml_feature_processing.py) derives with Spark
            # window functions: rolling mean/std, the new-minimum flag and the
            # no-improvement counter.
            # `cumulative_min_energy` is not one of them: the solver computes it
            # per step (`VQESolver.compute_energy`) and Spark only reads it, so `cummin`
            # below stands in for the solver, not for Spark.
            #
            # NOTE: No shared code - kept in sync by hand.
            run_rows: list[dict] = []
            prev_energy: float | None = None

            # `cummin` is exact and becomes the cumulative_min_energy column; `running_min`
            # below decides is_new_minimum with a tolerance.
            cummin = descent.energies[0]

            # 1e-8 Ha (hartree, the energy unit; 10 nanohartree) tolerance on
            # is_new_minimum, so a wiggle under it does not reset the no-improvement
            # counter.
            # --
            # Spark has none (strict `<` on cumulative_min_energy), so the two differ
            # only on improvements under 1e-8 Ha, far below the noise simulate_descent
            # adds. +inf so step 1 always registers a new minimum, as in the Spark job.
            running_min = float('inf')
            steps_since_imp = 0
            energy_window: list[float] = []

            for step, e in enumerate(descent.energies, start=FIRST_ITERATION):
                # None on step 1, as `VQESolver.compute_energy` records it: there is no
                # previous parameter vector to measure from. `astype(float)` in
                # `compute_horizon_features` turns it into NaN, which `np.mean` then
                # propagates into `param_delta_norm_mean_k{k}`.
                param_delta = (
                    None if step == FIRST_ITERATION else param_deltas[step - FIRST_ITERATION]
                )

                # None on step 1, matching `VQESolver.compute_energy` (vqe_solver.py): no
                # earlier iteration to difference against, and 0.0 would claim the energy
                # did not move. energy/synthetic.py follows this line.
                delta = None if prev_energy is None else e - prev_energy

                # Strict improvement beyond the tolerance on `running_min`
                # (`cummin` is exact).
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

                # Sample std (divides by n-1) and NaN for a one-row window, the same
                # definition as Spark's `stddev`.
                moving_std = (
                    float(np.std(energy_window, ddof=1)) if len(energy_window) > 1 else np.nan
                )

                run_rows.append(
                    {
                        RUN_KEY: experiment_id,
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
                        # Placeholder, overwritten below once the run's rows exist.
                        # Two tests read this column:
                        # `test_required_columns_present` and
                        # `test_mean_param_delta_norm_constant_per_run`
                        # (tests/ml/test_convergence_predictor.py).
                        # `test_run_level_features_present` also names it, but
                        # only checks that the `compute_horizon_features` output
                        # has a column of that name, so it never reads this one.
                        'mean_param_delta_norm': 0.0,
                    }
                )

                prev_energy = e

            # Whole-run mean written onto every row, so at any horizon K < n_iter it would
            # leak steps after K if a feature read it. `compute_horizon_features` does
            # not: it recomputes the mean from `parameter_delta_norm` within the horizon.
            # Skips the null on step 1, like Spark's `avg`.
            # --
            # The real `ml_iteration_features` has no such column (it carries the
            # cumulative `mean_param_change`; the run-level mean `mean_param_delta_norm`
            # is in `ml_run_summary`).
            mean_pdelta = float(np.mean(param_deltas[1:])) if len(param_deltas) > 1 else 0.0
            for row in run_rows:
                row['mean_param_delta_norm'] = mean_pdelta

            all_rows.extend(run_rows)

    return pd.DataFrame(all_rows)
