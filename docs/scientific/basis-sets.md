---
title: Basis Sets
---

# Basis Sets

A **basis set** defines the mathematical functions used to approximate molecular orbitals in quantum chemistry calculations. For VQE simulations, the basis set governs the number of qubits required under the Jordan-Wigner transformation, establishing a trade-off between accuracy and computational cost. This page describes the three basis sets supported by the Quantum Pipeline --- STO-3G, 6-31G, and cc-pVDZ --- and provides guidance on their selection.

---

## STO-3G

**Slater-Type Orbital --- 3 Gaussian** (STO-3G), introduced by Hehre, Stewart,
and Pople (1969), is the simplest and most widely used minimal basis set. It
approximates each Slater-type orbital (STO) with a linear combination of three
Gaussian-type functions (GTOs).

### Characteristics

- **Type:** Minimal basis set
- **Construction:** Each atomic orbital is represented by exactly one basis
  function, which is itself a contraction of three Gaussian primitives.
- **Accuracy:** Provides qualitatively correct results for molecular geometries
  and relative energies, but exhibits significant quantitative errors. Typical
  energy errors range from 0.1 to 1.0 Ha relative to the complete basis set
  limit.
- **Computational cost:** Lowest among all supported basis sets. The small
  number of basis functions translates directly to fewer qubits and shorter
  circuit depths.

### When to Use STO-3G

- Rapid prototyping and algorithm testing
- Benchmarking optimizer performance across many configurations
- Simulations on real quantum hardware where qubit count is severely constrained
- Initial exploration of large molecular systems where higher-accuracy basis
  sets are computationally prohibitive

---

## 6-31G

The **6-31G** basis set, developed by Ditchfield, Hehre, and Pople (1971), is a
split-valence basis set that provides a meaningful improvement in accuracy over
STO-3G while maintaining moderate computational cost.

### Characteristics

- **Type:** Split-valence double-zeta
- **Construction:** Core orbitals are represented by a single basis function
  contracted from six Gaussian primitives. Valence orbitals are split into two
  components: an inner part contracted from three Gaussians and an outer part
  consisting of a single Gaussian. This splitting allows the valence electron
  density to adjust its spatial extent during the self-consistent field
  procedure.
- **Accuracy:** Substantially better than STO-3G for energies, geometries, and
  other molecular properties. Typical energy errors range from 0.01 to 0.1 Ha.
- **Computational cost:** Moderate --- approximately 2 to 5 times slower than
  STO-3G, depending on the molecular system.

### When to Use 6-31G

- Standard calculations on small-to-medium molecules where STO-3G accuracy is
  insufficient
- Studies requiring a reasonable balance between accuracy and computational
  feasibility
- Investigations of basis set convergence trends without the full expense of
  correlation-consistent sets

---

## CC-pVDZ

The **correlation-consistent polarized Valence Double-Zeta** (cc-pVDZ) basis
set, introduced by Dunning (1989), represents a significant step up in accuracy
and computational demands. It is designed specifically for correlated
calculations and forms part of a systematic hierarchy (cc-pVDZ, cc-pVTZ,
cc-pVQZ, ...) that converges smoothly toward the complete basis set limit.

### Characteristics

- **Type:** Correlation-consistent polarized double-zeta
- **Construction:** Includes polarization functions (higher angular momentum
  functions beyond those occupied in the atomic ground state) that are essential
  for capturing electron correlation effects. The basis functions are optimized
  to recover correlation energy systematically.
- **Accuracy:** Significantly superior to both STO-3G and 6-31G. Typical energy
  errors range from 0.001 to 0.01 Ha. Well-suited for quantitative comparison
  with experimental results and high-level classical methods.
- **Computational cost:** High --- approximately 10 to 100 times slower than
  STO-3G. The increased number of basis functions leads to substantially more
  qubits under the Jordan-Wigner mapping.

### When to Use CC-pVDZ

- High-accuracy quantum chemistry calculations requiring quantitative agreement
  with experiment
- Studies of electron correlation effects
- Validation of VQE results against established classical methods (CCSD, FCI)
- Research applications where computational cost is secondary to accuracy

---

## Comparison

The following table summarizes the key properties of each supported basis set,
including qubit requirements for representative molecular systems as determined
by experiments conducted with the Quantum Pipeline framework.

| Property | STO-3G | 6-31G | cc-pVDZ |
|----------|--------|-------|---------|
| **Type** | Minimal | Split-valence | Correlation-consistent |
| **Accuracy** | Basic (\(\sim\)0.1--1 Ha error) | Moderate (\(\sim\)0.01--0.1 Ha error) | High (\(\sim\)0.001--0.01 Ha error) |
| **Qubits (H\(_2\))** | 4 | 8 | 10 |
| **Qubits (H\(_2\)O)** | 14 | 26 | 28 |
| **Relative Cost** | 1x (baseline) | 2--5x | 10--100x |
| **Memory Scaling** | Minimal | Moderate | High (often >10x STO-3G) |
| **Polarization Functions** | No | No | Yes |
| **Correlation Recovery** | Poor | Partial | Systematic |

!!! note "Qubit counts under Jordan-Wigner mapping"
    The qubit counts listed above correspond to the number of spin-orbitals
    produced by each basis set under the Jordan-Wigner transformation. Exact
    counts may vary depending on symmetry reduction, active space selection, and
    frozen-core approximations applied during Hamiltonian construction.

---

## Selection Guide

The appropriate basis set depends on the objectives and constraints of the
simulation. The following recommendations are drawn from experimental experience
with the Quantum Pipeline framework.

| Use Case | Recommended Basis Set | Rationale |
|----------|----------------------|-----------|
| Rapid prototyping and testing | STO-3G | Fastest execution, minimal resource requirements |
| Standard small-molecule calculations | 6-31G | Good accuracy-cost balance |
| High-accuracy quantum chemistry | cc-pVDZ | Systematic correlation recovery |
| VQE algorithm benchmarking | STO-3G | Fast, repeatable results across many configurations |
| Comparison with experimental data | cc-pVDZ or higher | Quantitative accuracy required |
| Real quantum hardware experiments | STO-3G | Minimizes qubit count and circuit depth |
| Large molecules (>10 atoms) | STO-3G or 6-31G | Computational feasibility constraints |

### Decision Flowchart

When selecting a basis set, consider the following progression:

1. **Start with STO-3G** to verify that the simulation pipeline functions
   correctly and to establish baseline results.
2. **Upgrade to 6-31G** if quantitative improvements are needed and
   computational resources permit.
3. **Use cc-pVDZ** only when high accuracy is essential and the molecular system
   is small enough to remain tractable.

---

## Impact on VQE Performance

The choice of basis set profoundly affects VQE performance across multiple
dimensions. Experimental results from the Quantum Pipeline benchmarking suite
illustrate these effects.

### Convergence Behavior

Larger basis sets increase the dimensionality of the parameter space, making the
optimization landscape more complex. Experiments with cc-pVDZ on the H\(_2\)
molecule demonstrated that convergence becomes substantially more difficult
compared to STO-3G:

- With STO-3G (4 qubits), H\(_2\) converges within 50--100 iterations.
- With cc-pVDZ (10 qubits), the same molecule failed to converge to the correct
  ground-state energy within the allocated iterations, yielding energies of
  approximately 24--26 Ha versus the expected value of approximately -1 Ha.

This divergence is attributed to the expanded parameter space and the increased
prevalence of local minima under random initialization.

<figure>
  <img src="https://qp-docs.codextechnologies.org/mkdocs/ccpvdz_convergence.png"
       alt="Convergence plot for H2 with cc-pVDZ basis set showing optimization difficulties">
  <figcaption>Figure 1. Convergence behavior of VQE for H<sub>2</sub> with the cc-pVDZ basis set. All hardware configurations exhibit difficulty reaching the correct ground-state energy, with optimization trajectories plateauing at elevated energy values (24--26 Ha). This illustrates the challenge of random parameter initialization in larger basis sets.</figcaption>
</figure>

### Iteration Timing

The computational cost per iteration scales significantly with basis set size.
For the H\(_2\) molecule, the average iteration time varies dramatically across
basis sets:

| Configuration | STO-3G (s/iter) | cc-pVDZ (s/iter) | Slowdown Factor |
|---------------|-----------------|-------------------|-----------------|
| CPU (i5-8500) | ~4.3 | 206.2 | ~48x |
| GPU GTX 1060 | ~2.4 | 50.6 | ~21x |
| GPU GTX 1050 Ti | ~2.5 | 58.4 | ~23x |

<figure>
  <img src="https://qp-docs.codextechnologies.org/mkdocs/ccpvdz_avg_iteration_time.png"
       alt="Bar chart comparing average iteration time across hardware configurations for cc-pVDZ">
  <figcaption>Figure 2. Average iteration time for H<sub>2</sub> with the cc-pVDZ basis set across hardware configurations. The CPU requires 206.19 seconds per iteration, while GPU GTX 1060 and GTX 1050 Ti achieve 50.55 and 58.42 seconds respectively --- a 3.5--4x improvement that underscores the value of GPU acceleration for computationally demanding basis sets.</figcaption>
</figure>

### GPU Acceleration Scaling

A particularly significant finding is that GPU acceleration benefits increase
with basis set complexity. For STO-3G, GPU speedup over CPU ranges from
1.74--1.81x. For cc-pVDZ, this speedup rises to 3.53--4.08x. This scaling
behavior occurs because larger basis sets involve more matrix operations that
can be efficiently parallelized on GPU architectures.

This observation has important implications: as simulations move toward more
accurate (and computationally expensive) basis sets, GPU acceleration becomes
not merely beneficial but essential for practical execution times.

---

## Implementation Details

Basis sets in the Quantum Pipeline are managed through integration with the
PySCF library. The supported basis sets are defined in the system configuration:

```python
SUPPORTED_BASIS_SETS = ['sto3g', '6-31g', 'cc-pvdz']
```

The basis set is passed to the PySCF driver during Hamiltonian construction:

```python
driver = PySCFDriver.from_molecule(molecule, basis=self.basis_set)
problem = driver.run()
hamiltonian = problem.second_q_ops()[0]
```

Additional basis sets supported by PySCF (e.g., cc-pVTZ, aug-cc-pVDZ,
6-311++G\*\*) can be added by extending the `SUPPORTED_BASIS_SETS` list without
modifying the core simulation logic.

---

## Summary

The choice of basis set is one of the most consequential configuration decisions
in VQE simulation. STO-3G provides a computationally efficient baseline suitable
for algorithm development and rapid prototyping. 6-31G offers a meaningful
accuracy improvement at moderate additional cost. cc-pVDZ delivers
correlation-consistent accuracy but demands substantially greater computational
resources, making GPU acceleration particularly valuable. The Quantum Pipeline
framework supports all three basis sets and is extensible to additional sets
through its PySCF integration.

---

## References

1. Szabo, A. & Ostlund, N.S. *Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory.* Dover Publications (1996)
2. Dunning, T.H. *Gaussian basis sets for use in correlated molecular calculations. I. The atoms boron through neon and hydrogen.* J. Chem. Phys. 90, 1007--1023 (1989)
3. Hehre, W.J. et al. *Self-Consistent Molecular-Orbital Methods. XII. Further Extensions of Gaussian-Type Basis Sets for Use in Molecular Orbital Studies of Organic Molecules.* J. Chem. Phys. 56, 2257--2261 (1972)
4. [PySCF Documentation -- Basis Sets](https://pyscf.org/user/gto.html)
