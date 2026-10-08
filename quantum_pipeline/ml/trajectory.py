"""The synthetic descent curve both generators draw from.

One caricature of VQE descent, not production code.

It was inlined in both generators' `generate_synthetic_trajectories` with every
constant slightly different - noise 0.001 vs 0.002, uphill probability 0.04 vs
0.05, and so on - so an experiment on one model's fake data did not transfer to
the other's.
The values here are the convergence module's, chosen only because they were the
lower-noise, fewer-uphill-steps pair.

This module only draws the curve. `convergence/synthetic.py` and `energy/synthetic.py`
call `simulate_descent` once per run and expand the returned `Descent` into rows shaped
like `ml_iteration_features`.

The convergence generator adds the `converged` label and the per-step columns
the Spark job derives; the energy generator adds the `final_energy` target
(and also carries `converged`) with only the raw per-step columns.

`OPTIMIZERS` and `BASIS_SETS` are the categorical fixtures they draw from, independently per run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator

# The two optimizers whose simulated curves get occasional uphill steps.
# --
# A modelling choice, not a measured fact about the real table. `VQESolver.compute_energy`
# (vqe_solver.py) is scipy's objective and records every function evaluation, and the
# `minimize` call passes no `jac` (analytic gradient).
GRADIENT_FREE_OPTIMIZERS = ('COBYLA', 'Nelder-Mead')

# Every name is a key of settings.SUPPORTED_OPTIMIZERS, so a real row can carry it.
OPTIMIZERS = ['COBYLA', 'L-BFGS-B', 'Nelder-Mead', 'SLSQP']

# The one-hot encoders (one 0/1 column per category) in convergence/models.py and
# preprocessing.py set handle_unknown='ignore', so a misspelling would not raise: a model
# fitted on other names would give every real row all-zero basis columns.
BASIS_SETS = ['sto3g', 'cc-pvdz']


class Descent(NamedTuple):
    """One run's simulated trajectory, `n_iter` samples long.

    The two lists must stay the same length: both generators index
    `param_deltas[step - FIRST_ITERATION]` while enumerating `energies`.

    Attributes:
        energies: Energy at each iteration, in the order the solver would emit them.
            Index 0 is `iteration_step` 1 (`schema.FIRST_ITERATION`).
        param_deltas: Stand-in for `parameter_delta_norm` (the L2 distance between
            consecutive parameter vectors, `vqe_solver.py`) at each iteration, shrinking
            as the curve flattens. Unlike the real table, it has a value at step 1;
            both generators emit None there instead, as `VQESolver.compute_energy` does
            (`vqe_solver.py`), there being no previous point to measure from. Synthetic
            rows therefore exercise the NaN path that `param_delta_norm_mean_k{k}` in
            `convergence/features.py` handles with nan-aware reductions.
        e_final: The asymptote the curve decays toward, before noise; becomes
            `final_energy` in `energy/synthetic.py`. Deliberately not `energies[-1]`:
            that carries its own noise draw and, for a converged run, still has about 1%
            of the drop left (see `k_rate` in `simulate_descent`). The real column is
            the lowest energy the solver sampled, not an asymptote; the module docstring
            of `energy/estimator.py` sets the two definitions side by side.
    """

    energies: list[float]
    param_deltas: list[float]
    e_final: float


def simulate_descent(
    rng: Generator,
    molecule: dict[str, Any],
    n_iter: int,
    optimizer: str,
    converged: bool,
) -> Descent:
    """Draw one exponential-decay descent: `E(t) = E_final + A * exp(-k * t) + noise`.

    The outcome is an input, decided by the caller before the curve is drawn - the
    reverse of a real run, where scipy sets `OptimizeResult.success` after the optimizer
    stops (or the solver forces it False when its own evaluation cap aborts the run) and
    the Spark job renames it.

    Args:
        rng: Source of every random draw; callers seed it, so a fixed seed reproduces
            the curve.
        molecule: Needs `e_fci` (the full configuration interaction energy: the exact
            ground-state energy in this basis, the floor no variational expectation value
            can pass, though noisy samples can) and `e_local` (a local minimum above it),
            both in Hartree.
            Other keys are ignored; see `schema.SYNTHETIC_MOLECULES`.
        n_iter: Number of samples. The generators set their own ranges; only
            `n_iter == 0` is special-cased here (empty lists, see `k_rate`).
        optimizer: Compared against `GRADIENT_FREE_OPTIMIZERS` and nothing else.
        converged: The label, fixed before the curve is drawn (see above). What the real
            `converged` column records is explained in
            `convergence/synthetic.generate_synthetic_trajectories`.

    Returns:
        `Descent` with `n_iter` energies and `n_iter` parameter deltas.

    A converged run decays to within 20 mHa above `e_fci` across the whole window with
    noise at 0.1% of |e_fci|. A stalled run decays toward `e_local`, reaches its
    asymptote 20-60% of the way through, and carries five times the noise.

    Within one molecule that 5x noise gap lets `energy_moving_std_5` (the rolling std
    over 5 steps) separate the classes on the last steps, once the stalled curve has
    flattened.

    It is not reliable everywhere: early in a converged H2 run the descent slope
    outweighs the 1 mHa noise and the std points the other way, and across molecules
    the gap is swamped because noise scales with |e_fci| (a converged N2 is noisier
    than a stalled H2).
    Either way the signal is a constant this function chose, which is the concrete
    reason a good score on synthetic data says nothing about whether the features
    are predictive: a model that scores well here has learned this function, not VQE.
    """
    e_fci = molecule['e_fci']
    e_local = molecule['e_local']

    if converged:
        # The asymptote sits above the exact energy (FCI), never below, since an exact
        # variational energy cannot pass it; the per-step noise is unbounded, so
        # individual samples can still dip under it.
        # Up to 20 mHa above it, which is about 12x chemical accuracy (1.6 mHa): a run can meet the
        # optimizer's stopping criterion well short of the right energy, which is all the
        # real `converged` records (convergence/synthetic.py).
        e_final = e_fci + rng.uniform(0.0, 0.02)

        # Decays across the whole window, but by the last steps the drift per step
        # (2-3e-4 Ha at n_iter=100) is under even H2's 1.1 mHa noise, so this tail looks
        # flat too. What separates the classes is where the flattening starts, the
        # noise level and the asymptote, not a tail that is visibly still moving.
        plateau_frac = 1.0
        noise_scale = abs(e_fci) * 0.001
    else:
        # Up to 0.10 Ha above e_local, so e_final can land above the `e_start` drawn
        # below (0.07-0.23 Ha above e_local); the amplitude is clamped there.
        e_final = e_local + rng.uniform(-0.05, 0.10)

        # Reaches 99% of its drop 20-60% of the way in, leaving the flat remainder the
        # plateau features are meant to detect.
        plateau_frac = float(rng.uniform(0.2, 0.6))

        # Noise scales with |e_fci| but the e_local-e_fci gap does not (0.34-0.46 Ha for
        # every fixture in schema.SYNTHETIC_MOLECULES), so the curve drowns as the
        # molecule grows: a stalled N2 run has noise std 0.54 Ha against a median drop
        # of 0.14 Ha, and the descent is invisible under it.
        noise_scale = abs(e_fci) * 0.005

    # Start above the local minimum by 20-50% of the e_local-e_fci gap (0.34-0.46 Ha per
    # fixture), so 0.07-0.23 Ha above e_local.
    e_start = e_local + rng.uniform(0.2, 0.5) * abs(e_local - e_fci)

    # Without the floor, e_final above e_start (about 3.5% of stalled H2 runs, 0.15% of N2
    # runs, whose wider gap pushes the start further up) makes the amplitude negative and
    # the curve climbs to its asymptote instead of descending. The 0.01 Ha floor keeps the
    # start above the asymptote and touches only those runs: no extra random draw.
    amplitude = max(e_start - e_final, 0.01)

    # Choose k so that 1% of the drop remains at t = n_iter * plateau_frac, i.e.
    # exp(-k * t) = 0.01. The 1e-9 only avoids dividing by zero for n_iter == 0, which
    # neither caller allows.
    k_rate = -np.log(0.01) / (n_iter * plateau_frac + 1e-9)

    energies: list[float] = []
    param_deltas: list[float] = []

    for t in range(n_iter):
        # Independent Gaussian noise at each step, uncorrelated from one step to the next.
        # It only makes the curve ragged; whether real jitter looks like this has not been
        # checked against `ml_iteration_features`.
        noise = float(rng.normal(0.0, noise_scale))
        energy = e_final + amplitude * np.exp(-k_rate * t) + noise

        if optimizer in GRADIENT_FREE_OPTIMIZERS and rng.random() < 0.04:
            # Adds 1.5x this step's |noise|, so the sample always ends above the clean
            # curve (by 0.5-2.5 |noise|, depending on the sign of the draw) but within a
            # few noise widths. It skews the noise upward rather than making a spike.
            energy += abs(noise) * 1.5

        energies.append(float(energy))
        # Step size: a random draw from an exponential distribution (mean 0.1, always
        # positive) times the same exp(-k_rate * t) decay as the energy, so steps shrink
        # as the curve flattens. `converged` reaches it only through k_rate: a stalled
        # run's plateau_frac is below 1, hence a larger k, so its steps shrink sooner.
        param_deltas.append(float(rng.exponential(0.1) * np.exp(-k_rate * t)))

    return Descent(energies=energies, param_deltas=param_deltas, e_final=float(e_final))
