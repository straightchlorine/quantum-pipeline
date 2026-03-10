---
title: Changelog
---

# Changelog

Release notes for Quantum Pipeline, starting from version 1.4.0.

For earlier versions, see the [GitHub releases](https://github.com/straightchlorine/quantum-pipeline/releases).

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
