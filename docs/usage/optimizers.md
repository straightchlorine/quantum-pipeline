---
title: "Optimizers: COBYLA, L-BFGS-B and More"
description: "Compare the classical optimizers available for VQE parameter tuning, such as COBYLA and L-BFGS-B, their settings, iteration limits and convergence control."
---

# Optimizers

Guide to the classical optimizers available for VQE parameter optimization.

## How It Works

The VQE optimization loop:

1. The optimizer proposes parameters $\theta$.
2. The simulator prepares state $|\psi(\theta)\rangle$ using the ansatz circuit.
3. The energy expectation $\langle \psi(\theta) | \hat{H} | \psi(\theta) \rangle$ is evaluated (exactly, or sampled with `--shots`).
4. The optimizer updates $\theta$ based on the energy.
5. Repeat until convergence or the iteration limit.

By the variational principle, the computed energy is always an upper bound on
the true ground state energy.

The pipeline offers the 10 optimizers listed in
[`settings.py`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/configs/settings.py).
Each one is configured in
[`optimizer_config.py`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/solvers/optimizer_config.py).

Seven optimizers use a gradient (L-BFGS-B, SLSQP, BFGS, CG, TNC, Newton-CG, trust-constr) and need exact expectation values, which any run without `--shots` provides (see [`--exact`](configuration.md#-exact)). With `--shots N` they are refused with exit code 2, so use COBYLA, Nelder-Mead or Powell for sampled runs.

```bash
quantum-pipeline -f data/molecules.json --optimizer L-BFGS-B
```

## Configured Optimizers

Each configured optimizer has a class that controls the `options` dict and
`tol` parameter passed to `scipy.optimize.minimize`, plus parameter validation.
The
[`OptimizerConfigFactory.create_config()`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/solvers/optimizer_config.py)
method picks the right class. It accepts `max_iterations` and
`convergence_threshold`, which are mutually exclusive. At the CLI,
`--convergence` takes precedence over `--max-iterations`.

### Specialized Configs

#### L-BFGS-B - Limited-memory BFGS with Bounds

Recommended default. A quasi-Newton method that approximates the inverse
Hessian using a limited number of past gradient evaluations. Memory-efficient
for high-dimensional problems.

- **Type**: Quasi-Newton (gradient-based)
- **Default maxiter**: 15,000
- **Config class**: [`LBFGSBConfig`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/solvers/optimizer_config.py)

| Mode | Triggered by | Behavior |
|------|-------------|----------|
| Strict iteration control | `--max-iterations N` | `maxfun=N`, `maxiter=N`, `ftol=1e-15`, `gtol=1e-15` |
| Convergence-based | `--convergence --threshold T` | `maxiter=15000`, `ftol=T`, `gtol=T` |
| Defaults | Neither flag | `maxiter=15000`, scipy default tolerances |

The tight tolerances in strict mode prevent scipy from stopping early. The
`compute_energy()` callback enforces the iteration limit via
`MaxFunctionEvalsReachedError`.

#### COBYLA - Constrained Optimization by Linear Approximations

Derivative-free, trust-region method with linear models. Handles noisy cost
functions well.

- **Type**: Derivative-free
- **Default maxiter**: 1,000
- **Config class**: [`COBYLAConfig`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/solvers/optimizer_config.py)

Uses the global `tol` parameter for convergence (not options-level). Warns if
`max_iterations < num_parameters + 2`, which is the minimum COBYLA needs for
an initial simplex.

#### SLSQP - Sequential Least Squares Programming

Gradient-based constrained optimizer that solves a sequence of quadratic
programming subproblems.

- **Type**: Sequential quadratic programming
- **Default maxiter**: 100
- **Config class**: [`SLSQPConfig`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/solvers/optimizer_config.py)

Sets `ftol` in options when convergence threshold is provided. Also passes the
threshold as global `tol`.

#### trust-constr - Trust-Region Constrained

Gradient-based trust-region method. It needs no Hessian and keeps its own
quasi-Newton approximation.

- **Type**: Trust-region
- **Default maxiter**: 1,000
- **Config class**: `TrustConstrConfig`

With `max_iterations` set, `xtol` and `gtol` are 1e-15. Scipy then stops later
than with its defaults, but it may still stop before `maxiter`. Otherwise the convergence threshold is passed as global `tol`.

### Generic Configs

The remaining 6 configured optimizers use
[`GenericConfig`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/solvers/optimizer_config.py),
which passes `maxiter` (or `maxfun` for TNC) into the options dict and returns
`convergence_threshold` as the global `tol`.

| Optimizer | Type | Default maxiter | Notes |
|-----------|------|-----------------|-------|
| `Nelder-Mead` | Derivative-free simplex | 5,000 | Slow, needs high budget |
| `Powell` | Derivative-free direction-set | 10,000 | Each iter is roughly n line searches |
| `BFGS` | Quasi-Newton | 1,000 | Fast convergence, full Hessian approximation |
| `CG` | Conjugate gradient | 2,000 | Moderate convergence, low memory |
| `TNC` | Truncated Newton (bounded) | 500 | Uses `maxfun` instead of `maxiter` |
| `Newton-CG` | Newton conjugate gradient | 1,000 | Needs the [analytic path](configuration.md#-exact): an exact run (no `--shots`) that is local and noiseless, without `--gpu`, with at most 16 qubits. Any other run exits with code 2. |

GenericConfig behavior:

- `max_iterations` set: uses it as `maxiter` (or `maxfun` for TNC)
- `convergence_threshold` set: uses optimizer's default `maxiter`, passes threshold as `tol`
- Neither set: uses optimizer's default `maxiter`

## Other Optimizers

Only the ten optimizers above are offered by the CLI. Other scipy methods
can be added with `OptimizerConfigFactory.register_optimizer()` and a custom
`OptimizerConfig` subclass or a lambda wrapping `GenericConfig`, plus an entry
in `settings.SUPPORTED_OPTIMIZERS`.

## Summary

| Optimizer | Type | Default maxiter |
|-----------|------|-----------------|
| `L-BFGS-B` | Quasi-Newton (gradient) | 15,000 |
| `COBYLA` | Derivative-free (trust-region) | 1,000 |
| `SLSQP` | Sequential quadratic programming | 100 |
| `BFGS` | Quasi-Newton (gradient) | 1,000 |
| `CG` | Conjugate gradient | 2,000 |
| `TNC` | Truncated Newton (bounded) | 500 |
| `Powell` | Derivative-free (direction-set) | 10,000 |
| `Nelder-Mead` | Derivative-free (simplex) | 5,000 |
| `Newton-CG` | Newton conjugate gradient (needs gradient) | 1,000 |
| `trust-constr` | Trust-region (gradient) | 1,000 |

In practice, the thesis used L-BFGS-B and COBYLA exclusively. The v2.0.0 verification tested COBYLA, L-BFGS-B, SLSQP, Nelder-Mead, Powell, and BFGS on H2 and
HeH+ (see [Benchmarking Results](../scientific/benchmarking.md#v200-verification-results)).
The main finding was that initialization strategy matters far more than
optimizer choice - with HF init, all tested gradient-based optimizers reached
similar energies.

## Iteration Control

When you pass `--max-iterations N`, the limit is enforced in two ways:

1. **Optimizer-level**: the config sets `maxiter` (and `maxfun` for L-BFGS-B)
   in the options dict.
2. **Callback-level**: `compute_energy()` in `VQESolver` tracks iteration count
   and raises `MaxFunctionEvalsReachedError` when exceeded.

If `--convergence` is set, `--max-iterations` is ignored (with a warning if it
was set to a non-default value); the threshold decides when the run stops, and
[`--max-evaluations`](configuration.md#-max-evaluations) (default 30000) is the
pipeline's own bound on evaluations. (`OptimizerConfig` itself
raises a `ValueError` if constructed directly with both values.)

## Performance Notes

On the analytic path (a local, exact, noiseless run with at most 16 qubits) a gradient optimizer gets the energy and its gradient from one statevector pass, so each iteration costs about one evaluation. On the Aer path (`--gpu`, `--noise`, or more than 16 qubits) it estimates the gradient by finite differences and needs about one extra evaluation per parameter, but faster convergence typically compensates. See [`--exact`](configuration.md#-exact).

Convergence mode (`--convergence`) pairs well with gradient-based optimizers
that converge monotonically:

```bash
quantum-pipeline -f data/molecules.json \
    --optimizer L-BFGS-B \
    --convergence \
    --threshold 1e-6
```
