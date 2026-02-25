---
title: Benchmarking Results
---

# Benchmarking Results

This page presents the experimental results obtained through systematic
benchmarking of VQE simulations using the Quantum Pipeline framework. The
experiments evaluate the impact of GPU acceleration on simulation performance,
analyze convergence characteristics across molecular systems of varying
complexity, and assess the influence of basis set selection on both accuracy and
computational cost. All data presented here are drawn from thesis research
conducted with the Quantum Pipeline infrastructure.

---

## Experimental Setup

### Hardware Configurations

Three hardware configurations were employed to establish a comprehensive
performance baseline and evaluate GPU acceleration benefits.

| Configuration | Processor / GPU | Memory |
|---------------|----------------|--------|
| CPU Baseline | Intel Core i5-8500 (6 cores @ 3.00 GHz) | 16 GB RAM |
| GPU1 | NVIDIA GTX 1060 (1280 CUDA cores) | 6 GB VRAM |
| GPU2 | NVIDIA GTX 1050 Ti (768 CUDA cores) | 4 GB VRAM |

All configurations operated within a containerized Docker environment using the
`nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04` base image, ensuring consistency
of the software stack across all experimental runs.

### Simulation Parameters

The following parameters were held constant across all primary experiments:

| Parameter | Value |
|-----------|-------|
| Basis set | STO-3G |
| Optimizer | L-BFGS-B (gradient-based) |
| Convergence threshold (ftol, gtol) | \(10^{-6}\) |
| Simulation method | Statevector (CPU/GPU) |
| Ansatz | EfficientSU2 |
| Parameter initialization | Random uniform \([0, 2\pi]\) |

The L-BFGS-B optimizer was selected for its effective balance between
convergence speed and memory efficiency. As a limited-memory quasi-Newton method,
it approximates the inverse Hessian matrix and supports box constraints on
parameters --- properties well-suited to VQE optimization landscapes.

### Evaluation Metrics

The system was assessed on the following metrics:

- **Iteration time** --- wall-clock time for a single VQE optimization iteration
- **Total simulation time** --- elapsed time from initialization to convergence
- **Iterations to convergence** --- number of optimizer iterations required
- **Cost function value** --- final ground-state energy in Hartree (Ha)
- **Resource utilization** --- CPU/GPU load and memory consumption
- **Streaming throughput** --- Apache Kafka and Spark pipeline performance

---

## Molecules Tested

Six molecular systems of increasing complexity were selected to evaluate
system performance across a representative range of problem sizes. The molecules
span from the simplest two-electron system (H\(_2\)) to twelve-qubit systems
(H\(_2\)O, NH\(_3\)).

| Molecule | Formula | Qubits (STO-3G) | Electrons | Approx. Iterations | Complexity |
|----------|---------|-----------------|-----------|-------------------|------------|
| Hydrogen | H\(_2\) | 4 | 2 | 50--100 | Minimal |
| Helium hydride cation | HeH\(^+\) | 4 | 2 | 50--100 | Minimal |
| Lithium hydride | LiH | 8 | 4 | ~1,455 | Moderate |
| Beryllium dihydride | BeH\(_2\) | 10 | 6 | ~2,373 | Significant |
| Water | H\(_2\)O | 12 | 10 | ~2,185 | High |
| Ammonia | NH\(_3\) | 12 | 10 | ~2,410 | High |

The progression from 4-qubit to 12-qubit systems allows systematic evaluation of
how GPU acceleration scales with problem size and circuit complexity.

---

## Convergence Analysis

### Overall Convergence Behavior

Across all 24 experimental runs (6 CPU, 9 GPU1, 9 GPU2), consistent convergence
patterns were observed. All hardware configurations produced similar convergence
trajectories for each molecular system, confirming that GPU acceleration does not
introduce numerical artifacts or degrade optimization quality.

<figure>
  <img src="https://qp-docs.codextechnologies.org/mkdocs/fin_convergence_all.png"
       alt="Convergence comparison across all molecules and hardware configurations">
  <figcaption>Figure 1. Overall convergence comparison across all tested molecules and hardware configurations. The convergence trajectories are consistent between CPU and GPU runs, demonstrating that GPU acceleration preserves numerical fidelity while reducing wall-clock time per iteration.</figcaption>
</figure>

Key convergence observations:

- **Small molecules (H\(_2\), HeH\(^+\), 4 qubits):** Rapid convergence within
  50--100 iterations. The optimization landscape is relatively smooth with few
  local minima.
- **Medium molecules (LiH, 8 qubits):** Approximately 1,455 iterations to
  convergence. An extended optimization tail is visible as the algorithm
  navigates a more complex landscape.
- **Large molecules (BeH\(_2\), H\(_2\)O, NH\(_3\), 10--12 qubits):**
  1,500--2,700 iterations required. The risk of entrapment in local minima
  increases substantially.

### Molecule-Specific Convergence

<figure>
  <img src="https://qp-docs.codextechnologies.org/mkdocs/convergence_BeHH.png"
       alt="Convergence plot for BeH2 molecule">
  <figcaption>Figure 2. Convergence behavior for BeH<sub>2</sub> (10 qubits). The optimization trajectory exhibits multiple phases: an initial rapid descent, followed by a plateau region, and a subsequent refinement phase. This multi-phase behavior is characteristic of medium-complexity molecular systems.</figcaption>
</figure>

<figure>
  <img src="https://qp-docs.codextechnologies.org/mkdocs/convergence_OHH.png"
       alt="Convergence plot for H2O molecule">
  <figcaption>Figure 3. Convergence behavior for H<sub>2</sub>O (12 qubits). The water molecule presents one of the most demanding optimization landscapes in the test suite, requiring extensive iteration counts to approach convergence. The GPU configurations achieve comparable final energies to CPU while completing iterations significantly faster.</figcaption>
</figure>

### Convergence Quality

The final energy values achieved by each configuration provide insight into
optimization quality. GPU configurations often achieved lower (better) energy
values due to their ability to complete more iterations within the same
experimental time budget.

<figure>
  <img src="https://qp-docs.codextechnologies.org/mkdocs/fin_best_cost.png"
       alt="Best cost function values achieved by each hardware configuration">
  <figcaption>Figure 4. Best cost function values (lowest energy) achieved by each hardware configuration across all molecules. GPU configurations frequently attain lower energies, particularly for larger molecules, owing to the greater number of iterations completed within the experimental timeframe.</figcaption>
</figure>

#### Best Achieved Energies (Ha)

| Molecule | CPU | GPU GTX 1060 | GPU GTX 1050 Ti |
|----------|-----|-------------|-----------------|
| H\(_2\) | -0.79 | -1.05 | -1.07 |
| LiH | -5.32 | -5.23 | -6.05 |
| BeH\(_2\) | -12.98 | -12.60 | -12.46 |
| H\(_2\)O | -51.44 | **-59.40** | -57.32 |
| NH\(_3\) | -47.88 | -46.26 | -50.36 |

The GPU GTX 1060 achieved the overall best energy for H\(_2\)O (-59.40 Ha),
while the GPU GTX 1050 Ti produced the best results for H\(_2\) (-1.07 Ha),
LiH (-6.05 Ha), and NH\(_3\) (-50.36 Ha). These results reflect the stochastic
nature of VQE optimization under random initialization: more iterations yield
better landscape exploration, and GPU acceleration enables precisely this
advantage.

### Comparison with Reference Values

To assess the absolute quality of the computed energies, the best VQE results
were compared against established literature values obtained using various
classical quantum chemistry methods with the STO-3G basis set.

| Molecule | Best VQE (Ha) | Literature (Ha) | Method | Relative Error |
|----------|--------------|-----------------|--------|---------------|
| H\(_2\) | -1.07 | -1.117 | HF/STO-3G | 4.2% |
| HeH\(^+\) | -3.65 | -2.927 | Full CI/STO-3G | 24.7% |
| LiH | -6.05 | -7.882 | CCSD/STO-3G | 23.2% |
| BeH\(_2\) | -12.98 | -15.15 | FCI/CAS/STO-3G | 14.3% |
| H\(_2\)O | -59.40 | -74.963 | HF/STO-3G | 20.8% |
| NH\(_3\) | -50.36 | -55.454 | HF/STO-3G | 9.2% |

For H\(_2\), the VQE result is within 4.2% of the Hartree-Fock reference,
indicating good convergence. For larger molecules, the relative errors are
larger (9--25%), primarily attributable to random parameter initialization and
the potential for local minima entrapment with the EfficientSU2 ansatz. These
results are consistent with the known challenges of VQE optimization for complex
molecular systems and underscore the importance of informed initialization
strategies in production-quality simulations.

---

## CPU vs. GPU Performance

### Overall Speedup

The aggregate performance statistics across all experimental runs demonstrate
consistent GPU acceleration.

| Configuration | Avg. Time/Iteration (s) | Total Iterations | Speedup |
|---------------|------------------------|------------------|---------|
| CPU (Intel i5-8500) | 4.259 | 8,832 | 1.00x (baseline) |
| GPU GTX 1060 6GB | 2.357 | 12,057 | **1.81x** |
| GPU GTX 1050 Ti 4GB | 2.454 | 10,871 | **1.74x** |

<figure>
  <img src="https://qp-docs.codextechnologies.org/mkdocs/fin_speedup.png"
       alt="GPU speedup summary across all experiments">
  <figcaption>Figure 5. GPU speedup relative to CPU baseline across all experimental runs. The GTX 1060 achieves a 1.81x mean speedup, while the GTX 1050 Ti achieves 1.74x. Both GPUs provide consistent acceleration that translates directly into greater iteration throughput.</figcaption>
</figure>

The GPU GTX 1060, with its higher CUDA core count (1280 vs. 768) and greater
VRAM capacity (6 GB vs. 4 GB), consistently outperforms the GTX 1050 Ti.
However, both GPUs deliver meaningful acceleration over the CPU baseline.

Critically, the GPU configurations also completed a substantially greater total
number of iterations: GPU1 executed 36% more iterations than CPU (12,057 vs.
8,832), enabling more thorough exploration of the parameter space.

### Iteration Time Analysis

<figure>
  <img src="https://qp-docs.codextechnologies.org/mkdocs/fin_iteration_time.png"
       alt="Iteration time comparison across molecules and hardware configurations">
  <figcaption>Figure 6. Average iteration time by molecule and hardware configuration. The acceleration benefit varies with molecular complexity: small molecules (4 qubits) show minimal or no benefit due to GPU overhead, while medium-sized molecules (8--10 qubits) exhibit the most pronounced speedup.</figcaption>
</figure>

### Scaling with Molecular Complexity

The relationship between molecular size and GPU acceleration benefit follows a
characteristic pattern:

| Molecule | Qubits | Avg. Iterations | GPU Speedup Range |
|----------|--------|----------------|-------------------|
| H\(_2\) | 4 | 650 | 0.8--1.0x (GPU overhead dominates) |
| HeH\(^+\) | 4 | 630 | 0.9--1.1x (marginal benefit) |
| LiH | 8 | 1,455 | 1.5--1.6x (moderate) |
| BeH\(_2\) | 10 | 2,373 | 1.8--2.1x (significant) |
| H\(_2\)O | 12 | 2,185 | 1.3--1.4x (moderate) |
| NH\(_3\) | 12 | 2,410 | 1.3--1.4x (moderate) |

For small molecules (4 qubits), the overhead of data transfer between CPU and
GPU memory negates the computational advantage. The crossover point occurs at
approximately 8 qubits, beyond which GPU acceleration provides consistent
benefits. The peak speedup of 1.8--2.1x is observed for BeH\(_2\) (10 qubits),
after which the speedup stabilizes at 1.3--1.4x for 12-qubit systems.

<figure>
  <img src="https://qp-docs.codextechnologies.org/mkdocs/performance_heatmap.png"
       alt="Performance heatmap showing speedup across molecules and GPU configurations">
  <figcaption>Figure 7. Performance heatmap illustrating the GPU speedup factor across all molecule-configuration combinations. The color intensity reflects the magnitude of acceleration, with the strongest benefits concentrated in the medium-complexity regime (8--10 qubits).</figcaption>
</figure>

### Iteration Count Comparison

<figure>
  <img src="https://qp-docs.codextechnologies.org/mkdocs/total_iterations_comparison.png"
       alt="Total iterations completed by each hardware configuration">
  <figcaption>Figure 8. Total iterations completed by each hardware configuration across all molecular systems. GPU configurations complete substantially more iterations than CPU within the same experimental timeframe, with GPU GTX 1060 executing 36% more iterations than the CPU baseline.</figcaption>
</figure>

### Time vs. Iterations

<figure>
  <img src="https://qp-docs.codextechnologies.org/mkdocs/time_vs_iterations.png"
       alt="Scatter plot of total time versus iterations for all experimental runs">
  <figcaption>Figure 9. Relationship between total simulation time and iteration count across all experimental runs. GPU runs cluster in the lower-right region of the plot, indicating more iterations completed in less time. The slope of each cluster reflects the average iteration time for each configuration.</figcaption>
</figure>

### Speedup Summary

<figure>
  <img src="https://qp-docs.codextechnologies.org/mkdocs/overall_speedup_summary.png"
       alt="Overall GPU speedup summary bar chart">
  <figcaption>Figure 10. Overall GPU speedup summary showing mean acceleration factors across all molecules. The GTX 1060 provides 1.81x average speedup, while the GTX 1050 Ti achieves 1.74x. These figures represent conservative estimates, as the speedup increases with problem complexity.</figcaption>
</figure>

### Practical Implications

The measured speedups have direct practical significance. A VQE simulation
campaign requiring 7 days of CPU computation can be completed in approximately
4 days on a GTX 1060 GPU. For research workflows involving hundreds of
molecular configurations, this acceleration compounds to weeks of saved
computation time. Furthermore, the ability to complete more iterations per unit
time directly translates to better optimization outcomes, as demonstrated by the
superior energy values achieved by GPU configurations.

---

## Basis Set Impact

### STO-3G Performance (Primary Experiments)

All primary benchmarking experiments utilized the STO-3G basis set, which
provides a computationally efficient baseline with 4--12 qubits for the tested
molecular systems. The results presented in the preceding sections --- speedup
factors of 1.74--1.81x, convergence within hundreds to thousands of iterations
--- are representative of STO-3G performance.

### CC-pVDZ Performance (Extended Experiments)

Additional experiments were conducted with the cc-pVDZ basis set on the H\(_2\)
molecule to assess the impact of basis set complexity on system performance.
These experiments revealed dramatically different performance characteristics.

| Configuration | STO-3G Time/Iter (s) | cc-pVDZ Time/Iter (s) | Slowdown |
|---------------|---------------------|----------------------|----------|
| CPU (i5-8500) | ~4.3 | 206.19 | ~48x |
| GPU GTX 1060 | ~2.4 | 50.55 | ~21x |
| GPU GTX 1050 Ti | ~2.5 | 58.42 | ~23x |

The most striking finding is that GPU acceleration scales favorably with basis
set complexity:

<figure>
  <img src="https://qp-docs.codextechnologies.org/mkdocs/ccpvdz_speedup.png"
       alt="GPU speedup comparison for cc-pVDZ basis set">
  <figcaption>Figure 11. GPU speedup relative to CPU for H<sub>2</sub> with the cc-pVDZ basis set. The GTX 1060 achieves a 4.08x speedup and the GTX 1050 Ti achieves 3.53x --- more than double the speedup observed with STO-3G (1.81x and 1.74x respectively). This confirms that GPU acceleration becomes increasingly valuable as basis set complexity grows.</figcaption>
</figure>

| Metric | STO-3G | cc-pVDZ |
|--------|--------|---------|
| GPU1 speedup | 1.81x | **4.08x** |
| GPU2 speedup | 1.74x | **3.53x** |
| CPU iterations (same time) | 8,832 | 724 |
| GPU1 iterations (same time) | 12,057 | 2,957 |
| GPU2 iterations (same time) | 10,871 | 2,557 |

<figure>
  <img src="https://qp-docs.codextechnologies.org/mkdocs/ccpvdz_iterations_comparison.png"
       alt="Iteration count comparison for cc-pVDZ experiments">
  <figcaption>Figure 12. Iteration count comparison for H<sub>2</sub> with cc-pVDZ. In the same experimental timeframe, GPU GTX 1060 completed 2,957 iterations versus 724 for CPU --- a 4x throughput increase. This additional iteration capacity is critical for navigating the more complex optimization landscapes associated with larger basis sets.</figcaption>
</figure>

### Convergence Challenges with CC-pVDZ

Despite the significant speedup, the cc-pVDZ experiments revealed convergence
difficulties inherent to larger basis sets:

- None of the configurations converged to the correct ground-state energy for
  H\(_2\) (expected approximately -1 Ha; observed approximately 24--26 Ha).
- Random parameter initialization proved inadequate for the expanded parameter
  space.
- The results underscore the need for informed initialization strategies
  (e.g., Hartree-Fock starting points) when working with correlation-consistent
  basis sets.

These findings highlight a fundamental tension in VQE simulation: while larger
basis sets improve the theoretical accuracy ceiling, they simultaneously make the
optimization problem harder to solve. GPU acceleration mitigates the
computational cost dimension of this challenge but cannot resolve the
algorithmic difficulties of navigating complex optimization landscapes.

### Implications for Basis Set Selection

The basis set impact data suggest the following practical guidelines:

1. **Development and prototyping** --- Use STO-3G exclusively. The fast iteration
   times (2--4 seconds) permit rapid experimentation with different ansatz
   configurations, optimizer settings, and initialization strategies.

2. **Production simulations** --- Consider 6-31G as a middle ground when STO-3G
   accuracy is insufficient but cc-pVDZ cost is prohibitive. GPU acceleration
   is recommended for any basis set beyond STO-3G.

3. **High-accuracy research** --- cc-pVDZ and beyond require GPU acceleration as
   a practical necessity. CPU-only execution at 206 seconds per iteration is
   impractical for the thousands of iterations needed to explore the parameter
   space adequately.

4. **Initialization strategy** --- For basis sets larger than STO-3G, random
   parameter initialization should be replaced with Hartree-Fock-informed
   starting points or other classical pre-computation strategies to improve
   convergence reliability.

---

## Detailed Error Analysis

### Sources of Discrepancy

The relative errors observed between VQE results and literature reference values
arise from several identifiable sources, each contributing to the overall
discrepancy to varying degrees.

**Random parameter initialization.** The EfficientSU2 ansatz parameters were
initialized from a uniform random distribution over \([0, 2\pi]\). For small
molecules with smooth optimization landscapes, this initialization is
sufficient to reach near-optimal solutions. For larger molecules, however, the
high-dimensional parameter space contains numerous local minima and saddle
points. The optimizer may converge to a local minimum far from the global
optimum, producing energies that significantly overestimate the true
ground-state value.

**Ansatz expressibility limitations.** The EfficientSU2 ansatz, while
hardware-efficient and broadly applicable, does not incorporate domain-specific
knowledge of molecular electronic structure. Chemistry-inspired ansatze such as
UCCSD encode physical excitation operators and may provide more direct paths to
the ground state, albeit at greater circuit depth.

**Barren plateaus.** For systems with many qubits, the optimization landscape of
hardware-efficient ansatze can exhibit barren plateaus --- regions where the
gradient of the cost function vanishes exponentially with system size. This
phenomenon impedes gradient-based optimizers such as L-BFGS-B and can cause
premature convergence to non-optimal parameter values.

**Basis set truncation.** Even with perfect optimization, the STO-3G basis set
introduces systematic errors due to its minimal representation of the molecular
orbital space. The reference values from literature (obtained via HF, CCSD, or
FCI methods) are themselves computed within the STO-3G basis, so basis set
truncation error affects both the VQE and reference calculations similarly. The
primary source of discrepancy is therefore optimization quality rather than
basis set limitation.

### Error Distribution by Molecular Complexity

The relative error exhibits a clear dependence on molecular complexity:

| Complexity Class | Molecules | Qubit Range | Relative Error Range |
|-----------------|-----------|-------------|---------------------|
| Simple | H\(_2\) | 4 | 4.2% |
| Simple | HeH\(^+\) | 4 | 24.7% |
| Moderate | LiH | 8 | 23.2% |
| Complex | BeH\(_2\) | 10 | 14.3% |
| Complex | H\(_2\)O | 12 | 20.8% |
| Complex | NH\(_3\) | 12 | 9.2% |

The non-monotonic relationship between qubit count and error (e.g., NH\(_3\) at
9.2% vs. H\(_2\)O at 20.8%, both with 12 qubits) indicates that the
optimization landscape topology varies significantly between molecules of
similar size. This suggests that molecular-specific tuning of ansatz parameters
and initialization strategies may yield substantial improvements in
convergence quality.

---

## Reproducibility and Statistical Analysis

### Run-to-Run Variability

Several molecules (H\(_2\), HeH\(^+\), LiH) were processed multiple times
across different hardware configurations, enabling assessment of result
reproducibility.

**Timing stability.** Iteration times exhibited run-to-run variations of
\(\pm\)5--10%, attributable to system-level factors such as operating system
scheduling, memory cache effects, and Docker container overhead. This level of
variability is consistent with expectations for containerized workloads and does
not compromise the validity of the performance comparisons.

**Energy variability.** Final energy values varied between runs of the same
molecule due to the stochastic nature of random parameter initialization. Each
run begins from a different point in parameter space and may converge to a
different local minimum. This variability is an intrinsic property of the VQE
algorithm under random initialization and is not attributable to hardware or
software instability.

### Statistical Significance of GPU Speedup

The GPU speedup measurements are based on averaged iteration times across
multiple molecules and runs. The consistency of the speedup factor across
different molecular systems (standard deviation <0.2x for both GPUs) provides
confidence that the reported acceleration is robust and not an artifact of
specific molecular configurations.

---

## Key Findings

The benchmarking campaign yielded several principal findings with implications
for both the practical use of the Quantum Pipeline and the broader understanding
of GPU-accelerated VQE simulation.

### 1. Consistent GPU Acceleration

GPU acceleration provides a reliable 1.74--1.81x speedup for STO-3G simulations
and 3.53--4.08x for cc-pVDZ simulations. This acceleration is consistent across
molecular systems and does not degrade numerical accuracy.

### 2. Scaling with Complexity

The benefit of GPU acceleration increases with problem complexity --- both in
terms of qubit count and basis set size. For medium-to-large molecular systems
and advanced basis sets, GPU acceleration transitions from a convenience to a
practical necessity.

### 3. More Iterations, Better Results

GPU configurations complete significantly more iterations within the same time
budget (36% more for GTX 1060 vs. CPU with STO-3G). This additional
computational capacity directly translates to better optimization outcomes, as
more thorough exploration of the parameter space yields lower energy values.

### 4. Convergence Depends on Initialization

Random parameter initialization is adequate for small molecules with simple
basis sets but becomes a significant limitation for larger systems. The cc-pVDZ
experiments demonstrate that informed initialization strategies are essential
for achieving convergence with advanced basis sets.

### 5. GPU as a NISQ-Era Bridge

In the NISQ era, where access to fault-tolerant quantum hardware remains
limited, GPU-accelerated classical simulation provides a practical bridge for
algorithm development, parameter optimization, and benchmarking. The measured
speedups indicate that a simulation requiring 7 days on CPU can be completed in
approximately 4 days on GPU, with even greater savings for larger basis sets.

### 6. Reproducibility

Repeated runs of the same molecular system produced iteration time variations
of \(\pm\)5--10%, indicating stable and reproducible system behavior. Energy
values varied between runs due to the stochastic nature of random parameter
initialization, which is an expected property of the VQE algorithm rather than a
system artifact.

---

## Limitations and Future Work

### Current Limitations

The benchmarking results presented here are subject to several limitations that
should be considered when interpreting the findings:

1. **Single optimizer.** All experiments used the L-BFGS-B optimizer. Different
   optimizers (COBYLA, SPSA, Nelder-Mead) may exhibit different scaling
   characteristics with respect to GPU acceleration and molecular complexity.

2. **Random initialization only.** The use of uniform random initialization
   limits the achievable accuracy for complex molecules. Hartree-Fock-informed
   initialization would likely improve convergence quality but was not employed
   in these experiments.

3. **Consumer-grade GPUs.** The GTX 1060 and GTX 1050 Ti are consumer-grade
   GPUs with limited CUDA core counts and VRAM. Professional-grade GPUs (e.g.,
   NVIDIA A100, H100) would be expected to provide substantially greater
   speedups, particularly for larger molecular systems.

4. **Limited cc-pVDZ data.** The cc-pVDZ experiments were conducted only for
   H\(_2\) due to computational constraints. Extending these experiments to
   larger molecules would provide more comprehensive basis set scaling data.

5. **Simulator-only results.** All experiments were conducted using statevector
   simulation. Results on real quantum hardware would be subject to additional
   factors including gate noise, measurement error, and decoherence.

### Planned Extensions

Future benchmarking efforts will address these limitations through:

- Systematic comparison of multiple classical optimizers
- Integration of Hartree-Fock and other informed initialization strategies
- Benchmarking on professional-grade GPU hardware (A100, H100)
- Extension of basis set studies to 6-31G and cc-pVDZ for all molecules
- Integration with IBM Quantum hardware backends for real-device validation
- Investigation of adaptive ansatz techniques (Adapt-VQE) within the pipeline

---

## Summary

The benchmarking results validate the Quantum Pipeline as an effective
infrastructure for systematic VQE experimentation. GPU acceleration provides
meaningful and consistent performance improvements that scale favorably with
problem complexity. The framework's streaming data architecture enables
comprehensive capture of iteration-level telemetry, supporting detailed
post-hoc analysis of convergence behavior and performance characteristics.

The principal quantitative findings are:

- **STO-3G speedup:** 1.74--1.81x for consumer-grade GPUs across six molecules
- **CC-pVDZ speedup:** 3.53--4.08x, demonstrating favorable scaling with basis
  set complexity
- **Iteration throughput:** GPU configurations complete up to 36% more
  iterations in the same time budget
- **Best energy results:** GPU configurations frequently achieve lower
  (better) energy values due to superior iteration throughput
- **Convergence quality:** H\(_2\) achieves 4.2% error relative to
  Hartree-Fock reference; larger molecules exhibit 9--25% errors attributable
  primarily to random initialization

The experimental findings reinforce the importance of considering hardware
acceleration, basis set selection, and initialization strategy as interconnected
design decisions in VQE simulation campaigns. GPU acceleration is particularly
critical for advanced basis sets, where the computational cost per iteration
increases by one to two orders of magnitude relative to minimal basis sets.
Future work will extend these benchmarks to additional ansatz types, optimizer
configurations, and the integration of real quantum hardware backends.
