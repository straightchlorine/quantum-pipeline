---
title: Changelog
---

# Changelog

Release notes for Quantum Pipeline, starting from version 1.4.0.

For earlier versions, see the [GitHub releases](https://github.com/straightchlorine/quantum-pipeline/releases).

---

## 2.0.0 (preparation)

Complete rewrite of the simulation module, DAGs, tests, and
infrastructure. Adds batch generation, multiple ansatze and optimizers,
a full observability stack, and a storage migration from MinIO to Garage.
Removes legacy code paths and simplifies metrics.

### Removals

- `quantum-pipeline.py` root script removed. The package is now invoked
  via `quantum_pipeline.cli:main` entry point.
- Chemical accuracy metric removed from simulation output and tests.
- Scientific literature references replaced with simpler PySCF-derived
  metrics (HF energy, nuclear repulsion) throughout the codebase.

### Batch generation

A batch system for generating ML training data across many configurations.
Four tiers sweep combinations of basis set, optimizer, init strategy, seed,
and ansatz. Three hardware lanes (two GPU, one CPU) run concurrently via
`ThreadPoolExecutor`, and state is persisted to `gen/ml_batch_state.json`
for idempotent resume. Progress metrics (`qp_batch_*`) are pushed to the
PushGateway per tier and lane.

### Multiple ansatze

The `--ansatz` flag selects between `EfficientSU2` (default),
`RealAmplitudes`, and `ExcitationPreserving`.

### Extended optimizer support

Eight optimizers are used in batch generation: L-BFGS-B, COBYLA, SLSQP,
Nelder-Mead, Powell, BFGS, CG, and TNC. Additional scipy optimizers
(Newton-CG, COBYQA, trust-constr, dogleg, trust-ncg, trust-exact,
trust-krylov) are available but less tested.

### Observability

`PerformanceMonitor` collects system and experiment metrics and exports
them to the Prometheus PushGateway. Monitoring environment variables
renamed from `QUANTUM_PERFORMANCE_*` to shorter names:
`MONITORING_ENABLED`, `MONITORING_INTERVAL`, `PUSHGATEWAY_URL`,
`MONITORING_EXPORT_FORMAT`. Metrics use the `qp_` prefix (`qp_vqe_*` for
experiment metrics, `qp_sys_*` for system metrics, `qp_batch_*` for batch
progress).

Grafana dashboards added for VQE experiment tracking, batch progress,
and system resource usage (`monitoring/grafana/dashboards/`).

Full exporter stack in the Docker Compose environment: statsd-exporter,
postgres-exporter, redis-exporter, nvidia-gpu-exporter, and Garage S3
metrics. Prometheus scrape configuration covers all exporters and the
PushGateway.

### Storage migration

MinIO replaced by Garage (`dxflrs/garage`) as the S3-compatible object
store. Environment variables: `S3_ENDPOINT` (default `http://garage:3901`),
`S3_REGION`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`. Feature tables are synced
to Cloudflare R2 via the R2 sync DAG.

### Docker

- GPU image updated to CUDA 12.6.3, builds qiskit-aer from source with
  cuQuantum. `CUDA_ARCH` build argument controls the target GPU
  architecture (default `8.6`).
- `DOCKER_GID` build argument for Airflow container to access the Docker
  socket (default `970`).

### Simulation module

`Runner`, `Solver`, and `Mapper` base classes now use proper ABCs.
Hardcoded constants (HF fidelity threshold, per-optimizer default
iterations) were moved to `configs/constants.py`.

The L-BFGS-B optimizer config had a workaround that set `ftol` and
`gtol` to `1e-15` when `max_iterations` was active, to prevent early
convergence. This was redundant - `compute_energy()` already enforces
the iteration limit by raising `MaxFunctionEvalsReachedError`. The hack
also prevented natural convergence, which is arguably worse for data
collection. Removed.

### DAGs

Configuration that was duplicated across 4 DAG files and 2 Spark scripts
(S3 paths, default args, Spark session setup) now lives in
`docker/airflow/common/`. Retries use exponential backoff. Spark tasks
have SLA targets. A dedicated batch generation DAG orchestrates ML data
collection runs, including container image builds. Airflow logic
simplified by removing overly complex scheduling and retry paths.

### Tests

Solver tests were spread across 8 files with overlapping coverage (4
files testing `VQESolver`, 3 testing optimizer config). Consolidated to
4 files. Deleted 6 tests that were testing `scipy.optimize.minimize`
behavior rather than project itself.

Added `pytest-xdist` for parallel test execution and `@pytest.mark.slow`
for a fast inner loop (`pytest -m "not slow"` runs in ~38s).

### Verification

Ran the refactored simulation across a range of configurations. All
combinations completed correctly on CPU (Aer statevector, seed 42,
EfficientSU2 2 reps unless noted).

<!-- TODO: fill in remaining combinations as they are tested -->

| Optimizer | Basis | Init | Iters | H\(_2\) (Ha) | HeH\(^+\) (Ha) | Notes |
|-----------|-------|------|-------|---------|-----------|-------|
| COBYLA | sto-3g | random | 30 | -1.555 | -4.004 | Completed naturally |
| L-BFGS-B | sto-3g | random | 30 | -0.895 | -3.283 | Hard limit |
| SLSQP | sto-3g | random | 50 | -1.111 | -3.562 | RealAmplitudes 3r |
| Nelder-Mead | sto-3g | random | 50 | -0.553 | -3.200 | |
| Powell | sto-3g | random | 30 | -0.005 | -0.011 | ExcitationPreserving |
| COBYLA | sto-3g | random | 235/221 | -1.785 | -4.338 | Convergence 1e-4 |
| COBYLA | sto-3g | hf | 50 | -1.836 | -4.267 | |
| BFGS | sto-3g | hf | 40 | -1.838 | -4.224 | |
| L-BFGS-B | sto-3g | hf | 30 | -1.838 | -4.218 | |
| L-BFGS-B | 6-31g | random | 1029/1372 | +2.101 | -0.956 | H\(_2\) stuck in local min |
| L-BFGS-B | 6-31g | hf | 50 | -1.857 | -4.294 | HF avoids local min |
| | | | | | | |

The last two rows are a good example of why initialization matters.
L-BFGS-B with random init on 6-31g spent 1029 iterations to arrive at a
positive energy for H\(_2\) - a barren plateau. The same setup with HF init
reached -1.857 Ha in 50 iterations.

---

## [1.4.0](https://github.com/straightchlorine/quantum-pipeline/releases/tag/1.4.0)

### Hartree-Fock Initialization

The random parameter initialization used in prior versions (uniform
\([0, 2\pi)\)) was identified as a major contributor to poor convergence
(see [Experimental Observations](scientific/vqe-algorithm.md#experimental-observations)).

Version 1.4.0 adds `--init-strategy hf`, which attempts to start VQE
from the classical Hartree-Fock solution instead.

A naive implementation - prepending a HartreeFock circuit to EfficientSU2
and setting all parameters to zero - turned out not to work. The fixed CX
entangling gates in EfficientSU2 are not parameterized and always act,
regardless of rotation angles. At zero parameters the rotation gates become
identity, but the CX gates still scramble the HF state.

The current approach runs a short classical pre-optimization that finds
EfficientSU2 parameters which directly prepare the HF state through the
ansatz, by maximizing state fidelity via `Statevector` simulation. This
partially resolves the initialization problem - for H\(_2\) (reps=1,
16 parameters) the pre-optimization reliably achieves fidelity of 1.0,
while for H\(_2\)O (reps=2, 72 parameters) it reaches approximately 0.993.

Early results comparing v1.3.x (random init) against v1.4.0 (HF init)
under the same conditions (sto-3g, COBYLA, reps=2, statevector, GPU,
max 300 iterations):

**H\(_2\)** (4 qubits, 32 parameters, HF reference: -1.117 Ha):

| Version | Init | Best electronic | Total energy | Iterations | Time |
|---------|------|----------------|-------------|------------|------|
| v1.3.x | Random | -1.73 Ha | n/a | 300 | 23 s |
| v1.4.0 | HF | -1.837 Ha | -1.122 Ha | 187 | 15 s |

The random run oscillated around -1.68 to -1.73 Ha without settling, while
HF init converged in 187 iterations (below the 300 limit) and found energy
within 5 mHa of the HF reference. Pre-optimization fidelity: 1.000.

**H\(_2\)O** (12 qubits, 72 parameters, HF reference: -74.963 Ha):

| Version | Init | Best electronic | Total energy | Iterations | Time |
|---------|------|----------------|-------------|------------|------|
| v1.3.x | Random | ~-75.95 Ha | n/a | 300 | 965 s |
| v1.4.0 | HF | -84.209 Ha | -75.016 Ha | 300 | 660 s |

The random run showed large oscillations at the end (costs jumping between
-73 and -76 Ha between iterations), indicating the optimizer was struggling
with the landscape. The HF run produced a stable result 53 mHa below the
HF reference, capturing some correlation energy. Pre-optimization fidelity:
0.993.

!!! note
    v1.3.x did not account for nuclear repulsion energy, so total energy is
    not available for those runs. The electronic energies are not directly
    comparable between versions - the v1.4.0 total energy column is what
    should be compared against literature values.

These results are encouraging but there is a lot more work to be done -
particularly around scaling the pre-optimization to larger molecules and
higher ansatz reps, where fidelity drops and the approach becomes slower.
Alternative strategies such as ADAPT-VQE or layer-by-layer parameter
fitting have not been explored yet.

### Nuclear Repulsion Energy

VQE optimizes the electronic Hamiltonian, which does not include nuclear
repulsion energy. Previous versions reported only the raw VQE minimum,
making it difficult to compare results against literature values directly.

The pipeline now captures nuclear repulsion energy from PySCF and adds it
to the VQE result. `VQEResult` exposes a `total_energy` property
(electronic + nuclear repulsion), and both components are logged:

```
Best energy: -1.83805736 Ha
Total energy (electronic + nuclear repulsion): -1.12295302 Ha
  (nuclear repulsion: 0.71510434 Ha)
```

### Code Organization

- **`quantum_pipeline/circuits/` package** - `HFData` and
  `build_hf_initial_state` moved from `solvers/` into a dedicated module,
  since initial state construction is circuit-level work rather than solver
  logic.
- **Mapper wrapper** - `Mapper.get_qiskit_mapper()` provides access to
  the underlying qiskit-nature mapper, so that circuit construction does
  not depend on direct qiskit imports.

### Bug Fixes

- **Timer context manager** - accessing `Timer.elapsed` inside a `with`
  block before `__exit__` ran caused `ValueError` when
  `MaxFunctionEvalsReachedError` was raised. The exception is now handled
  after the timer block exits.
- **Convergence vs max-iterations** - the `--convergence` flag no longer
  conflicts with the default `max_iterations=100`. When convergence mode
  is enabled, `max_iterations` is cleared unless explicitly set.
- **Simulation method validation** - GPU-only methods now produce a clear
  error when `--gpu` is not enabled.
- **Ansatz reps parsing** - `--ansatz-reps` is correctly parsed as an
  integer.

### Known Limitations

The items below are either known to be incomplete or are being actively
worked on:

- **HF pre-optimization scaling** - for molecules with many qubits and
  high ansatz reps, the fidelity pre-optimization may not converge to a
  satisfactory starting state. This is a known weakness of the current
  approach.
- **EfficientSU2 ansatz** - the hardware-efficient ansatz still does not
  preserve particle number or spin symmetry. The anomalies documented in
  the thesis (e.g. HeH\(^+\) sub-FCI energy) remain possible.
