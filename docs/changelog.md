---
title: Changelog
---

# Changelog

Release notes for Quantum Pipeline, starting from version 1.4.0.

For earlier versions, see the [GitHub releases](https://github.com/straightchlorine/quantum-pipeline/releases).

---

## 1.4.0

### Hartree-Fock Initialization

The random parameter initialization used in prior versions (uniform
\([0, 2\pi)\)) was identified as a major contributor to poor convergence
(see [Experimental Observations](scientific/vqe-algorithm.md#experimental-observations)).
Version 1.4.0 adds `--init-strategy hf`, which attempts to start VQE
from the classical Hartree-Fock solution instead.

A naive implementation — prepending a HartreeFock circuit to EfficientSU2
and setting all parameters to zero — turned out not to work. The fixed CX
entangling gates in EfficientSU2 are not parameterized and always act,
regardless of rotation angles. At zero parameters the rotation gates become
identity, but the CX gates still scramble the HF state. For H\(_2\) with
reps=1, this produced a starting electronic energy of +0.208 Ha instead of
the expected -1.832 Ha.

The current approach runs a short classical pre-optimization that finds
EfficientSU2 parameters which directly prepare the HF state through the
ansatz, by maximizing state fidelity via `Statevector` simulation. This
partially resolves the initialization problem — for H\(_2\) (reps=1,
16 parameters) the pre-optimization reliably achieves fidelity of 1.0,
while for H\(_2\)O (reps=2, 72 parameters) it reaches approximately 0.993.

Early results for H\(_2\) (sto-3g, COBYLA, 100 iterations, statevector):

| Init strategy | Total energy | Error from HF |
|---------------|-------------|----------------|
| Random | -1.097 Ha | 19.6 mHa |
| HF (v1.4.0) | -1.123 Ha | 6.1 mHa |

These results are encouraging but there is clearly more work to be done —
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

- **`quantum_pipeline/circuits/` package** — `HFData` and
  `build_hf_initial_state` moved from `solvers/` into a dedicated module,
  since initial state construction is circuit-level work rather than solver
  logic.
- **Mapper wrapper** — `Mapper.get_qiskit_mapper()` provides access to
  the underlying qiskit-nature mapper, so that circuit construction does
  not depend on direct qiskit imports.

### Bug Fixes

- **Timer context manager** — accessing `Timer.elapsed` inside a `with`
  block before `__exit__` ran caused `ValueError` when
  `MaxFunctionEvalsReachedError` was raised. The exception is now handled
  after the timer block exits.
- **Convergence vs max-iterations** — the `--convergence` flag no longer
  conflicts with the default `max_iterations=100`. When convergence mode
  is enabled, `max_iterations` is cleared unless explicitly set.
- **Simulation method validation** — GPU-only methods now produce a clear
  error when `--gpu` is not enabled.
- **Ansatz reps parsing** — `--ansatz-reps` is correctly parsed as an
  integer.

### Known Limitations

The items below are either known to be incomplete or are being actively
worked on:

- **HF pre-optimization scaling** — for molecules with many qubits and
  high ansatz reps, the fidelity pre-optimization may not converge to a
  satisfactory starting state. This is a known weakness of the current
  approach.
- **EfficientSU2 ansatz** — the hardware-efficient ansatz still does not
  preserve particle number or spin symmetry. The anomalies documented in
  the thesis (e.g. HeH\(^+\) sub-FCI energy) remain possible.
- **Kubernetes deployment** — Terraform configurations in `terraform/`
  are in early stages and not yet functional.
