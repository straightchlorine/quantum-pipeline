---
title: Scientific Background Overview
description: Background on electronic structure and VQE, with entry points to the algorithm, basis set and benchmarking pages.
---

# Scientific Background

Quantum chemistry simulation is a promising near-term application of quantum
computing. The core challenge is the electronic structure problem: finding a
molecule's ground-state energy by computing the lowest eigenvalue of its
Hamiltonian. Classical exact methods scale factorially with system size, so
approximate methods trade accuracy for tractability.

The Variational Quantum Eigensolver (VQE) is a hybrid quantum-classical
algorithm. It uses parameterized quantum circuits to prepare trial states and
classical optimizers to refine them. Its shallow circuits make it a candidate
for the NISQ era, where devices have limited qubits and significant noise.

This section covers the algorithm, the supported basis sets, and benchmarking
results, drawing on the thesis experiments and the v2.0.0 verification runs.
Limitations are noted in
[Benchmarking: Limitations](benchmarking.md#limitations-and-future-work).

## Section Guide

<div class="grid cards" markdown>

-   :material-atom:{ .lg .middle } **VQE Algorithm**

    ---

    Variational principle, ansatz types, parameter initialization, and
    convergence behavior.

    [:octicons-arrow-right-24: VQE Algorithm](vqe-algorithm.md)

-   :material-molecule:{ .lg .middle } **Basis Sets**

    ---

    STO-3G, 6-31G, and cc-pVDZ: accuracy, cost, and qubit requirements, with
    selection guidance.

    [:octicons-arrow-right-24: Basis Sets](basis-sets.md)

-   :material-chart-line:{ .lg .middle } **Benchmarking Results**

    ---

    GPU acceleration, energy results, initialization comparisons, and accuracy
    against PySCF references.

    [:octicons-arrow-right-24: Benchmarking Results](benchmarking.md)

</div>

## Notation Conventions

| Symbol | Meaning |
|--------|---------|
| \(\lvert \psi \rangle\) | Quantum state (Dirac notation) |
| \(\hat{H}\) | Hamiltonian operator |
| \(\theta\) | Variational parameters |
| \(E_0\) | Ground-state energy |
| Ha | Hartree (atomic unit of energy) |
| \(n\) | Number of qubits |
| \(K\) | Number of basis functions (spatial orbitals) |
| \(N = 2K\) | Number of spin orbitals; equals \(n\) under Jordan-Wigner |
| \(N_e\) | Number of electrons |
| \(\mathbf{R}_A, Z_A\) | Position and charge of nucleus \(A\) |
| \(\hat{a}_p^\dagger, \hat{a}_p\) | Fermionic creation and annihilation operators for spin orbital \(p\) |
| \(h_{pq}, h_{pqrs}\) | One- and two-electron integrals |
| \(E_{NN}\) | Nuclear repulsion energy (additive constant) |
| \(E_{\text{HF}}\) | Hartree-Fock energy |
| \(E_{\text{corr}}\) | Correlation energy, \(E_{\text{exact}} - E_{\text{HF}}\) |
| \(U(\theta)\) | Parameterized ansatz circuit |
| \(m\) | Number of variational parameters |
| \(\hat{P}_j, c_j\) | Pauli string and its Hamiltonian coefficient |
| \(M\) | Number of measurement shots |
