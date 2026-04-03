# Scientific Background

Quantum chemistry simulation is one of the most promising near-term applications
of quantum computing. The core challenge is the electronic structure problem -
finding the ground-state energy of molecular systems by computing the lowest
eigenvalue of the molecular Hamiltonian. Classical exact methods (Full
Configuration Interaction) scale factorially with system size, making them
intractable beyond the smallest molecules. Approximate methods like
Hartree-Fock, DFT, and Coupled Cluster trade accuracy for tractability.

The Variational Quantum Eigensolver (VQE) takes a different approach. As a
hybrid quantum-classical algorithm, VQE uses parameterized quantum circuits to
prepare trial states and classical optimizers to refine them. Shallow circuits
and classical post-processing make VQE practical for the NISQ era, where
quantum devices have limited qubits and significant noise.

The pipeline supports three ansatz types, sixteen classical optimizers, two
parameter initialization strategies (random and Hartree-Fock), and three basis
sets. Accuracy is evaluated against PySCF-derived Hartree-Fock reference
energies. The scientific content in this section draws on thesis experiments
(random initialization, L-BFGS-B, consumer GPUs) and v2.0.0 verification runs
(multiple optimizers and init strategies). Limitations are documented in
[Benchmarking: Limitations](benchmarking.md#limitations-and-future-work).

## Section Guide

<div class="grid cards" markdown>

-   :material-atom:{ .lg .middle } **VQE Algorithm**

    ---

    Variational principle, ansatz construction (EfficientSU2, RealAmplitudes,
    ExcitationPreserving), parameter initialization strategies, and convergence
    behavior from thesis and v2.0.0 experiments.

    [:octicons-arrow-right-24: VQE Algorithm](vqe-algorithm.md)

-   :material-molecule:{ .lg .middle } **Basis Sets**

    ---

    STO-3G, 6-31G, and cc-pVDZ - trade-offs between accuracy, computational
    cost, and qubit requirements, with selection guidance.

    [:octicons-arrow-right-24: Basis Sets](basis-sets.md)

-   :material-chart-line:{ .lg .middle } **Benchmarking Results**

    ---

    GPU acceleration performance, energy results across molecules,
    initialization strategy comparisons, and accuracy against PySCF references.

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

## Further Reading

- Peruzzo, A. et al. (2014). "A variational eigenvalue solver on a photonic
  quantum processor." *Nature Communications*, 5, 4213.
- McClean, J. R. et al. (2016). "The theory of variational hybrid
  quantum-classical algorithms." *New Journal of Physics*, 18(2), 023023.
- Tilly, J. et al. (2022). "The Variational Quantum Eigensolver: a review of
  methods and best practices." *Physics Reports*, 986, 1-128.
- Preskill, J. (2018). "Quantum Computing in the NISQ era and beyond."
  *Quantum*, 2, 79.
