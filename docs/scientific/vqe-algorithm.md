---
title: Variational Quantum Eigensolver
---

# Variational Quantum Eigensolver

The Variational Quantum Eigensolver (VQE) is a hybrid quantum-classical algorithm
designed to approximate the ground-state energy of a quantum system. Originally
proposed by Peruzzo et al. (2014), VQE combines parameterized quantum circuits
with classical optimization to find the lowest eigenvalue of a molecular
Hamiltonian. Its shallow circuit depth and tolerance for noise make it one of the
most practical algorithms for near-term quantum devices operating in the NISQ
(Noisy Intermediate-Scale Quantum) regime.

For a thorough treatment of VQE theory, see Tilly et al. (2022) and the
[Qiskit VQE tutorial](https://learning.quantum.ibm.com/tutorial/variational-quantum-eigensolver).

---

## Algorithm Overview

The VQE algorithm operates as an iterative loop between a quantum processor (or
simulator) and a classical optimizer:

1. **Initialization** --- Select a molecular system, basis set, and ansatz.
   Initialize the variational parameters \(\theta\).

2. **State Preparation** --- Execute the parameterized quantum circuit (ansatz)
   to prepare the trial state \(\lvert \psi(\theta) \rangle\).

3. **Energy Measurement** --- Measure the expectation value
   \(\langle \psi(\theta) | \hat{H} | \psi(\theta) \rangle\) by decomposing
   the Hamiltonian into a sum of Pauli operators.

4. **Classical Optimization** --- Feed the measured energy back to a classical
   optimizer (e.g., L-BFGS-B, COBYLA), which proposes updated parameters
   \(\theta'\).

5. **Convergence Check** --- If the energy change between successive iterations
   falls below a specified threshold (e.g., \(10^{-6}\) Ha), terminate.
   Otherwise, return to step 2.

### Flowchart

The following diagram illustrates the VQE optimization loop as implemented in
the Quantum Pipeline:

```mermaid
flowchart TD
    A[Initialize parameters θ] --> B[Prepare state |ψ⟩ via ansatz]
    B --> C[Measure ⟨ψ|H|ψ⟩]
    C --> D[Return energy to classical optimizer]
    D --> E{Converged?}
    E -->|No| F[Update θ → θ']
    F --> B
    E -->|Yes| G[Report ground-state energy E₀]

    style A fill:#1a237e,color:#ffffff
    style G fill:#1b5e20,color:#ffffff
    style E fill:#e65100,color:#ffffff
```

---

## Ansatz Construction

The **ansatz** is the parameterized quantum circuit that prepares the trial
state. The choice of ansatz is critical to VQE performance: it must be
expressive enough to represent the ground state while remaining shallow enough
to execute on noisy hardware.

### UCCSD Ansatz

The Unitary Coupled Cluster Singles and Doubles (UCCSD) ansatz is a
chemistry-inspired construction that applies single and double excitation
operators to a Hartree-Fock reference state:

\[
\lvert \psi_{\text{UCCSD}} \rangle = e^{T(\theta) - T^\dagger(\theta)} \lvert \phi_0 \rangle
\]

While UCCSD provides strong theoretical guarantees, its circuit depth can be
prohibitive for NISQ devices, motivating hardware-efficient alternatives.

### EfficientSU2 Ansatz

The Quantum Pipeline employs the **EfficientSU2** ansatz from Qiskit as the
default circuit construction. EfficientSU2 is a hardware-efficient ansatz that
uses layers of single-qubit SU(2) rotations followed by entangling CNOT gates.
Its advantages include:

- **Shallow circuit depth** --- scales linearly with the number of qubits and
  layers, making it feasible for NISQ simulation.
- **Full SU(2) coverage** --- each qubit undergoes RY and RZ rotations,
  providing sufficient expressibility for many molecular systems.
- **Flexible entanglement** --- supports various entanglement patterns (linear,
  full, circular).

The trade-off is that hardware-efficient ansatze lack the physical intuition of
UCCSD and may encounter optimization difficulties such as barren plateaus for
large systems.

---

## Convergence Behavior

The convergence of VQE depends on molecular complexity, ansatz choice, optimizer,
and parameter initialization. The thesis experiments observed the following
patterns.

For small molecular systems (e.g., H\(_2\), HeH\(^+\) with 4 qubits), VQE
typically converges within 50--100 iterations. As complexity increases, larger
molecules (e.g., H\(_2\)O, NH\(_3\) with 12 qubits) may require 1500--2700
iterations, with growing risk of entrapment in local minima.

<figure>
  <img src="https://qp-docs.codextechnologies.org/mkdocs/convergence_HH.png"
       alt="Convergence plot for H2 molecule showing energy vs. iteration number">
  <figcaption>Figure 1. Convergence behavior of VQE optimization for the H<sub>2</sub> molecule. The energy functional decreases rapidly within the first few iterations, reaching near-optimal values with approximately 50--100 iterations. The smooth convergence profile reflects the relatively simple optimization landscape for this two-electron system.</figcaption>
</figure>

<figure>
  <img src="https://qp-docs.codextechnologies.org/mkdocs/convergence_LiH.png"
       alt="Convergence plot for LiH molecule showing energy vs. iteration number">
  <figcaption>Figure 2. Convergence behavior of VQE optimization for the LiH molecule. Compared to H<sub>2</sub>, LiH exhibits a more complex convergence trajectory with an extended tail, reflecting the larger parameter space (8 qubits) and more intricate optimization landscape.</figcaption>
</figure>

Key factors affecting convergence:

- **Parameter initialization** --- Hartree-Fock-informed initialization
  generally yields faster and more reliable convergence than random initialization.
- **Basis set complexity** --- Experiments with cc-pVDZ demonstrated
  significantly slower convergence compared to STO-3G.
- **Ansatz expressibility** --- Insufficient expressibility results in
  systematic bias; excessive expressibility can introduce barren plateaus.

---

## Implementation in Quantum Pipeline

Within the Quantum Pipeline framework, VQE simulations are executed through the
`vqe_runner` module, which orchestrates the interaction between Qiskit's quantum
circuit primitives and the classical optimization backend. Key implementation
details include:

- **Hamiltonian construction** via PySCF driver integration, supporting multiple
  basis sets and molecular geometries.
- **Ansatz selection** defaulting to EfficientSU2 with configurable depth and
  entanglement topology.
- **Optimizer** defaulting to L-BFGS-B, with support for COBYLA, SLSQP,
  Nelder-Mead, and SPSA. See the [Optimizers](../usage/optimizers.md) page for
  configuration details.
- **Statevector simulation** with optional GPU acceleration through NVIDIA
  cuQuantum, enabling significant speedups for medium-to-large molecular
  systems (see [GPU Acceleration](../deployment/gpu-acceleration.md)).
- **Streaming telemetry** --- iteration-level data (energy, parameters, timing)
  is published to Apache Kafka for real-time monitoring and post-hoc analysis.

For practical guidance on running VQE simulations, consult the
[Quick Start](../getting-started/quick-start.md) and
[Examples](../usage/examples.md) pages.

---

## References

1. Peruzzo, A. et al. *A variational eigenvalue solver on a photonic quantum processor.* Nature Communications 5, 4213 (2014).
2. McClean, J.R. et al. *The theory of variational hybrid quantum-classical algorithms.* New Journal of Physics 18, 023023 (2016).
3. Tilly, J. et al. *The Variational Quantum Eigensolver: A review of methods and best practices.* Physics Reports 986, 1--128 (2022).
