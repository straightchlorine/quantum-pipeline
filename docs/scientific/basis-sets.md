---
title: Basis Sets
description: Compare the STO-3G, 6-31G and cc-pVDZ basis sets by accuracy, qubit count and cost, with a guide to choosing one.
---

# Basis Sets

A basis set defines the mathematical functions used to approximate molecular orbitals in quantum chemistry calculations. For VQE simulations, the basis set governs the number of qubits required under the [Jordan-Wigner transformation](https://docs.quantum.ibm.com/api/qiskit-nature/qiskit_nature.second_q.mappers.JordanWignerMapper) - one qubit per spin orbital, so twice the number of basis functions - establishing a trade-off between accuracy and computational cost. <!-- TODO: once mathematical-foundations.md is reviewed, restore: [Mathematical Foundations](mathematical-foundations.md#from-basis-functions-to-qubits) covers that counting and what basis truncation costs in energy. -->This page describes the three basis sets supported by the project and provides guidance on their selection.

## Supported Basis Sets

The pipeline validates basis sets against a
[fixed list in settings.py](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/configs/settings.py)
before any simulation begins. Validation is handled by
[`validate_basis_set()`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/drivers/basis_sets.py),
which raises `ValueError` for unsupported basis sets.

The three supported basis sets are `sto3g`, `6-31g`, and `cc-pvdz`.

## STO-3G

Slater-Type Orbital - 3 Gaussian (STO-3G), introduced by Hehre, Stewart, and Pople (1969), is the simplest basis set. It approximates each Slater-type orbital with a linear combination of three Gaussian-type functions.

### Characteristics

| Property | Value |
|----------|-------|
| Construction | One basis function per atomic orbital, contracted from 3 Gaussian primitives |
| Typical energy error | 0.1 - 1.0 Ha relative to the complete basis set limit |
| Computational cost | Lowest among supported basis sets |

### When to Use STO-3G

- Rapid prototyping and algorithm testing
- Benchmarking optimizer performance across many configurations
- Simulations on real quantum hardware where qubit count is constrained
- Initial exploration of larger molecular systems

## 6-31G

The 6-31G basis set, developed by Ditchfield, Hehre, and Pople (1971), is a split-valence basis set that provides meaningful improvement in accuracy over STO-3G at moderate additional cost.

### Characteristics

| Property | Value |
|----------|-------|
| Type | Split-valence double-zeta |
| Construction | Core: 1 function from 6 Gaussians. Valence: split into inner (3 Gaussians) and outer (1 Gaussian) |
| Typical energy error | 0.01 - 0.1 Ha relative to the complete basis set limit |

### When to Use 6-31G

- Standard calculations on small-to-medium molecules where STO-3G accuracy is insufficient
- Studies requiring a reasonable balance between accuracy and feasibility
- Investigating basis set convergence trends without the full expense of correlation-consistent sets

### V2.0.0 Observations

The v2.0.0 verification runs included 6-31G tests on H\(_2\) and HeH\(^+\), providing the first 6-31G data from the pipeline. The results highlight the importance of initialization strategy at this basis set level:

| Optimizer | Init | Molecule | Iters | Energy (Ha) | Outcome |
|-----------|------|----------|-------|-------------|---------|
| L-BFGS-B | random | H\(_2\) | 1029 | +2.101 | Poor local minimum, physically meaningless |
| L-BFGS-B | random | HeH\(^+\) | 1372 | -0.956 | Poor convergence |
| L-BFGS-B | HF | H\(_2\) | 50 | -1.857 | Good result |
| L-BFGS-B | HF | HeH\(^+\) | 50 | -4.294 | Good result |

The larger parameter space of 6-31G (8 qubits for H\(_2\), vs. 4 with STO-3G) made random initialization fail in these two L-BFGS-B runs. HF initialization reached a sensible energy in both. These runs used EfficientSU2.

## CC-pVDZ

The correlation-consistent polarized Valence Double-Zeta (cc-pVDZ) basis set, introduced by Dunning (1989), is a step up in accuracy and computational demands. It is part of a systematic hierarchy (cc-pVDZ, cc-pVTZ, cc-pVQZ, ...) that converges toward the complete basis set limit.

### Characteristics

| Property | Value |
|----------|-------|
| Type | Correlation-consistent polarized double-zeta |
| Construction | Includes polarization functions for capturing electron correlation, optimized for systematic correlation energy recovery |
| Typical energy error | 0.001 - 0.01 Ha relative to the complete basis set limit |

### When to Use CC-pVDZ

- High-accuracy calculations
- Studies of electron correlation effects
- Validation of VQE results against classical methods (CCSD, FCI)

## Selection Guide

The appropriate basis set depends on the objectives and constraints of the simulation. The following recommendations incorporate experience from both the thesis experiments (STO-3G, cc-pVDZ) and v2.0.0 verification runs (STO-3G, 6-31G).

| Use Case | Recommended Basis Set | Rationale |
|----------|----------------------|-----------|
| Rapid prototyping and testing | STO-3G | Fastest execution, minimal resource requirements |
| Standard small-molecule calculations | 6-31G | Good accuracy-cost balance (use HF init) |
| High-accuracy results | cc-pVDZ | Systematic correlation recovery |
| VQE algorithm benchmarking | STO-3G | Fast, repeatable results across many configurations |
| Comparison with experimental data | cc-pVDZ or higher | Quantitative accuracy required |
| Real quantum hardware experiments | STO-3G | Minimizes qubit count and circuit depth |
| Large molecules (>10 atoms) | STO-3G or 6-31G | Computational feasibility constraints |

### Decision Flowchart

When selecting a basis set, consider the following progression:

1. **Start with STO-3G** to verify that the simulation pipeline functions correctly and to establish baseline results.
2. **Upgrade to 6-31G** if quantitative improvements are needed and computational resources permit. Use HF initialization.
3. **Use cc-pVDZ** only when high accuracy is essential and the molecular system is small enough to remain tractable.

## Implementation Details

The basis set is passed to the PySCF driver in
[`VQERunner.provide_hamiltonian()`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/runners/vqe_runner.py).
The same call extracts the Hartree-Fock reference data - particle count, spatial orbitals, HF energy, nuclear repulsion energy - which the pipeline uses in three places. The first is the report of the run's deviation from the Hartree-Fock reference. The second is the initial state that the `ExcitationPreserving` ansatz structurally requires. The third is the parameter pre-optimization under `--init-strategy hf`. See [Ansatz Construction](vqe-algorithm.md#ansatz-construction) for which ansatz needs which.

Because the reference is the Hartree-Fock energy in the same basis, the reported deviation cancels basis-set error and reflects ansatz and optimizer quality. It does not measure accuracy against the true ground state, so it cannot be read across basis sets as a quality comparison. <!-- TODO: restore ([Chemical accuracy](mathematical-foundations.md#chemical-accuracy)) -->

Additional basis sets supported by PySCF (e.g., cc-pVTZ, aug-cc-pVDZ, 6-311++G**) can be added by extending the
[`SUPPORTED_BASIS_SETS`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/configs/settings.py)
list without modifying the core simulation logic.

## References

1. Szabo, A. & Ostlund, N.S. *Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory.* Dover Publications (1996)
2. Dunning, T.H. *Gaussian basis sets for use in correlated molecular calculations. I. The atoms boron through neon and hydrogen.* J. Chem. Phys. 90, 1007-1023 (1989)
3. Hehre, W.J. et al. *Self-Consistent Molecular-Orbital Methods. XII. Further Extensions of Gaussian-Type Basis Sets for Use in Molecular Orbital Studies of Organic Molecules.* J. Chem. Phys. 56, 2257-2261 (1972)
4. [PySCF Documentation - Basis Sets](https://pyscf.org/user/gto.html)
