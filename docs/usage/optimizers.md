# Optimizers

Guide to the classical optimizers available for VQE parameter optimization.

## How It Works

The VQE optimization loop:

1. The optimizer proposes parameters $\theta$.
2. The simulator prepares state $|\psi(\theta)\rangle$ using the ansatz circuit.
3. The energy expectation $\langle \psi(\theta) | \hat{H} | \psi(\theta) \rangle$ is measured.
4. The optimizer updates $\theta$ based on the measured energy.
5. Repeat until convergence or the iteration limit.

By the variational principle, the computed energy is always an upper bound on
the true ground state energy.

The pipeline accepts all 16 optimizers listed in
[`settings.py`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/configs/settings.py#L4),
but only 8 have dedicated configuration classes in
[`optimizer_config.py`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/solvers/optimizer_config.py).
The remaining 8 are accepted by the CLI but will raise a `ValueError` at
runtime because the factory does not know how to configure them.

```bash
quantum-pipeline -f molecules.json --optimizer L-BFGS-B
```

## Configured Optimizers

Each configured optimizer has a class that controls the `options` dict and
`tol` parameter passed to `scipy.optimize.minimize`, plus parameter validation.
The
[`OptimizerConfigFactory.create_config()`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/solvers/optimizer_config.py#L233)
method picks the right class. It accepts `max_iterations` and
`convergence_threshold`, which are mutually exclusive.

### Specialized Configs

#### L-BFGS-B - Limited-memory BFGS with Bounds

Recommended default. A quasi-Newton method that approximates the inverse
Hessian using a limited number of past gradient evaluations. Memory-efficient
for high-dimensional problems.

- **Type**: Quasi-Newton (gradient-based)
- **Default maxiter**: 15,000
- **Config class**: [`LBFGSBConfig`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/solvers/optimizer_config.py#L50)

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
- **Config class**: [`COBYLAConfig`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/solvers/optimizer_config.py#L94)

Uses the global `tol` parameter for convergence (not options-level). Warns if
`max_iterations < num_parameters + 2`, which is the minimum COBYLA needs for
an initial simplex.

#### SLSQP - Sequential Least Squares Programming

Gradient-based constrained optimizer that solves a sequence of quadratic
programming subproblems.

- **Type**: Sequential quadratic programming
- **Default maxiter**: 100
- **Config class**: [`SLSQPConfig`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/solvers/optimizer_config.py#L130)

Sets `ftol` in options when convergence threshold is provided. Also passes the
threshold as global `tol`.

### Generic Configs

The remaining 5 configured optimizers use
[`GenericConfig`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/solvers/optimizer_config.py#L163),
which passes `maxiter` (or `maxfun` for TNC) into the options dict and returns
`convergence_threshold` as the global `tol`.

| Optimizer | Type | Default maxiter | Notes |
|-----------|------|-----------------|-------|
| `Nelder-Mead` | Derivative-free simplex | 5,000 | Slow, needs high budget |
| `Powell` | Derivative-free direction-set | 10,000 | Each iter is roughly n line searches |
| `BFGS` | Quasi-Newton | 1,000 | Fast convergence, full Hessian approximation |
| `CG` | Conjugate gradient | 2,000 | Moderate convergence, low memory |
| `TNC` | Truncated Newton (bounded) | 500 | Uses `maxfun` instead of `maxiter` |

GenericConfig behavior:

- `max_iterations` set: uses it as `maxiter` (or `maxfun` for TNC)
- `convergence_threshold` set: uses optimizer's default `maxiter`, passes threshold as `tol`
- Neither set: uses optimizer's default `maxiter`

## Unconfigured Optimizers

These are listed in
[`settings.SUPPORTED_OPTIMIZERS`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/configs/settings.py#L4)
and accepted by the CLI, but have no entries in `OptimizerConfigFactory`.
Selecting one raises a `ValueError` at runtime:

`Newton-CG`, `trust-constr`, `trust-ncg`, `trust-exact`, `trust-krylov`,
`dogleg`, `COBYQA`, `custom`

To use one, register it via `OptimizerConfigFactory.register_optimizer()` with
a custom `OptimizerConfig` subclass or a lambda wrapping `GenericConfig`.

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

If both `max_iterations` and `convergence_threshold` are resolved at the solver
level, `max_iterations` takes priority. At the `OptimizerConfig` constructor
level, passing both raises a `ValueError`.

## Performance Notes

Gradient-based optimizers require $O(n)$ additional circuit evaluations per
iteration for $n$ parameters (parameter-shift rules), but the improved
convergence rate typically compensates.

For GPU-accelerated simulations the per-iteration overhead is low, making
gradient-based methods (especially L-BFGS-B) the most efficient choice in wall
time.

Convergence mode (`--convergence`) pairs well with gradient-based optimizers
that converge monotonically:

```bash
quantum-pipeline -f molecules.json \
    --optimizer L-BFGS-B \
    --convergence \
    --threshold 1e-6
```
