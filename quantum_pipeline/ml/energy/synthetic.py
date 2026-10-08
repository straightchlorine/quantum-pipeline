"""Synthetic energy trajectories for development and tests.

Not production code: an exponential-decay caricature of VQE descent, drawn by
`quantum_pipeline.ml.trajectory.simulate_descent` and laid out like the Iceberg table
`ml_iteration_features`, so `extract_features_at_fraction` and `EnergyEstimator.fit_evaluate`
can run without a pipeline run.

This module adds what the energy estimator needs on top of the curve: the `final_energy`
target and the raw per-step columns. The sibling `quantum_pipeline.ml.convergence.synthetic`
draws the same curve and emits the convergence predictor's wider column set, including the
per-step columns the Spark job derives with window functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantum_pipeline.ml.schema import FIRST_ITERATION, RUN_KEY, default_molecules
from quantum_pipeline.ml.trajectory import BASIS_SETS, OPTIMIZERS, simulate_descent


def generate_synthetic_trajectories(
    n_runs: int = 300,
    max_iter: int = 150,
    seed: int = 42,
    molecules: list[dict] | None = None,
) -> pd.DataFrame:
    """Generate VQE-like trajectory data for the energy estimator's development and tests.

    Each run is one curve from `quantum_pipeline.ml.trajectory.simulate_descent`. The
    outcome is drawn first and the curve shaped to match it.

    A converged run decays toward an asymptote just above the molecule's `e_fci` (its
    exact energy from full configuration interaction, FCI), a stalled one toward
    `e_local` and goes flat part-way through the window (the numbers are in
    `simulate_descent`). Within one molecule, the gap between the two asymptotes is what
    the regressor has to predict.

    None of the metadata columns is an input to `final_energy`: it comes from `e_fci` or
    `e_local` plus one random offset, and the metadata only pick the molecule and the
    curve shape. `num_qubits` is still predictive, because it identifies the molecule and
    so the energy level the target sits at.

    The `e_fci` values span about 106.5 Hartree across the five fixtures
    (H2 -1.1, N2 -107.7), which dwarfs the 0.34-0.46 Ha gap between `e_local`
    and `e_fci`.

    Leave-one-molecule-out CV (cross-validation) (`fitting.lomo_folds`: each fold
    trains on every molecule but one and tests on that one) withholds the held-out
    molecule's `num_qubits`, and as the default fixtures all have distinct qubit
    counts, every fold tests on a value never seen in training (the H2 and N2 folds
    lie outside the training range, the other three fall between training values).

    Of the other metadata columns, `optimizer` reaches only the curve: COBYLA and
    Nelder-Mead, which never compute a gradient, get an uphill step with probability
    0.04 per iteration. `basis_set` and `ansatz_reps` touch nothing; a model that
    weights either on this data is fitting noise. `optimizer` and `basis_set` are drawn
    independently, so they are uncorrelated.

    Args:
        n_runs: Target total, split across `molecules` as
            `max(1, n_runs // len(molecules))` runs each, remainder dropped. The real
            total is therefore a multiple of `len(molecules)`, and `n_runs` below
            `len(molecules)` still gives one run per molecule.
        max_iter: Upper bound on steps per run, inclusive. The lower bound is 30; when
            `max_iter` is below 30 every run has exactly 30 steps.
        seed: Seed for `numpy.random.default_rng`; the same seed gives the same frame.
            Every draw comes from that one generator, so a different `max_iter` shifts
            the stream from the first run whose step-count draw lands differently
            (measured at seed 42: going from 150 to 151 leaves only the first run
            identical, over 50 or 300 runs; the step-count draw that sets its 83 steps
            is the same, and every later run starts from a shifted state). A different
            `n_runs` gives the same frame when `max(1, n_runs // len(molecules))` is
            unchanged; when it changes, only the first molecule's runs stay identical,
            because every later molecule starts from a shifted generator state.
        molecules: Molecule spec dicts (`name`, `num_qubits`, `e_fci`, `e_local`: the
            local-minimum energy a stalled run settles near).
            Defaults to `quantum_pipeline.ml.schema.default_molecules`.

    Returns:
        One row per iteration with columns: experiment_id, molecule_name, num_qubits,
        optimizer, basis_set, ansatz_reps, iteration_step (1-indexed, starting at
        `schema.FIRST_ITERATION`), energy, cumulative_min_energy, energy_delta (null
        on step 1), parameter_delta_norm (null on step 1), final_energy, converged (bool).

        `converged` here is which asymptote the curve was given. What the real column
        records (scipy's `OptimizeResult.success`, which says nothing about the energy
        being right) is explained in `convergence/synthetic.generate_synthetic_trajectories`.
        The energy estimator never reads it; the column is here so the frame matches
        the table.

        Fewer columns than the convergence generator emits: `energy/features.py`
        computes `energy_moving_std` and `steps_since_improvement` from `energy` itself,
        so the generator only has to supply the raw descent and the target.

    Raises:
        ValueError: If `molecules` is empty.
    """
    rng = np.random.default_rng(seed)

    if molecules is None:
        molecules = default_molecules()
    if not molecules:
        raise ValueError('molecules must not be empty')

    rows = []
    runs_per_mol = max(1, n_runs // len(molecules))

    for mol in molecules:
        for run_idx in range(runs_per_mol):
            experiment_id = f'syn_{mol["name"]}_{run_idx:04d}'

            # Independent draws: a round-robin over OPTIMIZERS (4) and BASIS_SETS (2)
            # would visit only 4 of the 8 pairs, since 2 divides 4, making `basis_set` a
            # function of `optimizer` and, after one-hot encoding, collinear with it.
            optimizer = OPTIMIZERS[rng.integers(len(OPTIMIZERS))]
            basis_set = BASIS_SETS[rng.integers(len(BASIS_SETS))]
            ansatz_reps = rng.integers(1, 4)

            # `max(max_iter + 1, 31)` keeps the range non-empty, as numpy's `integers`
            # raises when low >= high; a `max_iter` under 30 gives 30 steps.
            n_iter = rng.integers(30, max(max_iter + 1, 31))

            # Outcome first, curve second (see `simulate_descent`). A hard-coded 70%,
            # independent of the molecule, so `converged` is uncorrelated with `num_qubits`
            # here; the convergence generator scales it with qubit count instead.
            converged = rng.random() < 0.70

            descent = simulate_descent(rng, mol, int(n_iter), optimizer, converged)

            # The asymptote the curve decays toward, not energies[-1]; `Descent.e_final`.
            # The real `final_energy` is the lowest energy sampled, so it is defined
            # differently; the module docstring of energy/estimator.py sets the two
            # definitions side by side with what the
            # difference does to the error.
            e_final = descent.e_final

            prev_energy: float | None = None
            cummin = descent.energies[0]

            for step, e in enumerate(descent.energies, start=FIRST_ITERATION):
                cummin = min(cummin, e)

                # None on step 1, as `VQESolver.compute_energy` records it; the same line
                # in convergence/synthetic.py says why not 0.0.
                delta = None if prev_energy is None else e - prev_energy

                # None on step 1 as well: `VQESolver` has no previous parameter vector to
                # measure from, so the real table carries a null there.
                param_delta = (
                    None
                    if step == FIRST_ITERATION
                    else descent.param_deltas[step - FIRST_ITERATION]
                )

                rows.append(
                    {
                        RUN_KEY: experiment_id,
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
                    }
                )
                prev_energy = e

    return pd.DataFrame(rows)
