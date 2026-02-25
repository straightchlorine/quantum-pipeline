# Performance Metrics

The Quantum Pipeline collects a comprehensive set of metrics during VQE simulation execution. These metrics are organized into three categories: system metrics, VQE execution metrics, and scientific accuracy metrics.

---

## Metric Categories

### System Metrics

System metrics track hardware resource utilization for each simulation container. These are collected by a background monitoring thread and pushed to the Prometheus PushGateway at regular intervals.

| Metric | Prometheus Name | Type | Unit | Description |
|--------|----------------|------|------|-------------|
| CPU Usage | `quantum_system_cpu_percent` | Gauge | % | Current CPU utilization percentage |
| Memory Usage | `quantum_system_memory_percent` | Gauge | % | Current memory utilization percentage |
| Memory Used | `quantum_system_memory_used_bytes` | Gauge | bytes | Current memory consumption in bytes |
| CPU Load (1m) | `quantum_system_cpu_load_1m` | Gauge | load | 1-minute load average |
| Container Uptime | `quantum_system_uptime_seconds` | Gauge | seconds | Time since container start |

!!! note "GPU Metrics"
    GPU acceleration is supported via the `--gpu` flag, but GPU-specific Prometheus metrics (e.g., GPU utilization, GPU memory) are not currently exported by the performance monitor. GPU benefits are observed indirectly through reduced VQE execution times.

### VQE Execution Metrics

VQE execution metrics capture the performance characteristics of the variational quantum eigensolver algorithm. These include timing breakdowns for each phase of the simulation and iteration-level progress data.

| Metric | Prometheus Name | Type | Unit | Description |
|--------|----------------|------|------|-------------|
| Iteration Count | `vqe_iteration_count` | Gauge | count | Current optimizer iteration number |
| Current Energy | `vqe_current_energy` | Gauge | Hartree | Latest energy value from the optimizer |
| Execution Time | `vqe_execution_time_seconds` | Gauge | seconds | Cumulative execution time |
| Convergence Rate | `vqe_convergence_rate` | Gauge | Ha/iteration | Rate of energy decrease per iteration |

The Grafana dashboard tracks additional detailed timing metrics:

| Dashboard Metric | Prometheus Name | Type | Description |
|-----------------|----------------|------|-------------|
| Total Execution Time | `quantum_vqe_total_time` | Gauge | End-to-end simulation time |
| VQE Optimization Time | `quantum_vqe_vqe_time` | Gauge | Time spent in VQE optimization loop |
| Hamiltonian Build Time | `quantum_vqe_hamiltonian_time` | Gauge | Time to construct the molecular Hamiltonian |
| Qubit Mapping Time | `quantum_vqe_mapping_time` | Gauge | Time for fermionic-to-qubit operator mapping |
| Iterations Count | `quantum_vqe_iterations_count` | Gauge | Total iterations to convergence |
| Minimum Energy | `quantum_vqe_minimum_energy` | Gauge | Best energy found during optimization |

### Scientific Accuracy Metrics

Scientific metrics evaluate the quality of VQE results against known reference values. These metrics are critical for assessing whether simulations achieve chemical accuracy.

| Dashboard Metric | Prometheus Name | Type | Description |
|-----------------|----------------|------|-------------|
| Ground State Energy | `quantum_experiment_minimum_energy` | Gauge | Minimum energy from optimization |
| Energy Error | `quantum_vqe_energy_error_millihartree` | Gauge | Deviation from reference value in mHa |
| Chemical Accuracy | `quantum_vqe_within_chemical_accuracy` | Gauge | Binary indicator (1 = within 1 mHa) |
| Accuracy Score | `quantum_vqe_accuracy_score` | Gauge | Normalized accuracy score (0.0 to 1.0) |

!!! info "Chemical Accuracy Threshold"
    Chemical accuracy is defined as an energy error of 1 millihartree (mHa) or less relative to the reference value. This threshold is significant because errors below 1 mHa are generally considered negligible for most chemical predictions.

---

## Prometheus Metric Names

### Complete Metric Reference

All metrics exported by the Quantum Pipeline use consistent naming conventions and label sets.

#### Labels

Every metric includes the following labels where applicable:

| Label | Description | Example Values |
|-------|-------------|----------------|
| `container_type` | Identifies the simulation container configuration | `cpu`, `gpu1`, `gpu2` |
| `molecule_symbols` | Chemical formula of the molecule being simulated | `H2`, `LiH`, `BeH2`, `H2O`, `NH3` |
| `optimizer` | Optimization algorithm used | `L-BFGS-B`, `COBYLA`, `SLSQP` |
| `basis_set` | Basis set for the simulation | `sto-3g`, `cc-pvdz` |

#### Full Metric List

```promql
# System Resource Metrics
quantum_system_cpu_percent{container_type="cpu"}
quantum_system_memory_percent{container_type="cpu"}
quantum_system_uptime_seconds{container_type="cpu"}

# VQE Performance Metrics
quantum_vqe_total_time{container_type, molecule_symbols, optimizer}
quantum_vqe_vqe_time{container_type, molecule_symbols, optimizer}
quantum_vqe_hamiltonian_time{container_type, molecule_symbols, optimizer}
quantum_vqe_mapping_time{container_type, molecule_symbols, optimizer}
quantum_vqe_iterations_count{container_type, molecule_symbols, optimizer}
quantum_vqe_minimum_energy{container_type, molecule_symbols, optimizer}

# Scientific Accuracy Metrics
quantum_experiment_minimum_energy{container_type}
quantum_experiment_iterations_count{container_type}
quantum_vqe_energy_error_millihartree{container_type, molecule_symbols, optimizer}
quantum_vqe_within_chemical_accuracy{container_type, molecule_symbols, optimizer}
quantum_vqe_accuracy_score{container_type, molecule_symbols, optimizer}

# Derived Efficiency Metrics (via PushGateway)
quantum_vqe_iterations_per_second{container_type, molecule_symbols, optimizer}
quantum_vqe_time_per_iteration{container_type, molecule_symbols, optimizer}
quantum_vqe_overhead_ratio{container_type, molecule_symbols, optimizer}
quantum_vqe_efficiency{container_type, molecule_symbols, optimizer}
quantum_vqe_setup_ratio{container_type, molecule_symbols, optimizer}
```

---

## Collection Configuration

### Collection Interval

The background monitoring thread collects system metrics at a configurable interval:

```python
# Default collection interval: 30 seconds
--performance-interval 30

# Higher frequency for detailed analysis
--performance-interval 1

# Lower frequency to reduce overhead
--performance-interval 60
```

### Export Formats

Metrics can be exported in multiple formats:

=== "Prometheus (Default)"

    Metrics are pushed to the PushGateway in Prometheus exposition format:

    ```bash
    quantum-pipeline run \
        --enable-performance-monitoring \
        --performance-export-format prometheus \
        --performance-pushgateway http://pushgateway:9091
    ```

=== "JSON"

    Metrics are saved to a local JSON file after simulation completion:

    ```bash
    quantum-pipeline run \
        --enable-performance-monitoring \
        --performance-export-format json
    ```

=== "Both"

    Export to both Prometheus and JSON simultaneously:

    ```bash
    quantum-pipeline run \
        --enable-performance-monitoring \
        --performance-export-format both \
        --performance-pushgateway http://pushgateway:9091
    ```

### Environment Variables

The PushGateway URL can also be set via environment variable:

```bash
export QUANTUM_PERFORMANCE_PUSHGATEWAY_URL=http://pushgateway:9091
```

---

## Querying Metrics

### PromQL Examples

The following PromQL queries demonstrate common monitoring tasks. These can be used in Grafana panels or directly in the Prometheus UI.

#### Average Iteration Time by Container Type

```promql
avg by (container_type) (
    quantum_vqe_total_time{molecule_symbols=~".*"}
)
```

#### VQE Performance (Iterations per Second)

```promql
quantum_vqe_iterations_count{container_type=~"$container_type"}
    / quantum_vqe_total_time{container_type=~"$container_type"}
```

#### VQE Efficiency Ratio

Ratio of VQE optimization time to total execution time, indicating how much time is spent on actual computation versus overhead:

```promql
quantum_vqe_vqe_time{container_type=~"$container_type"}
    / quantum_vqe_total_time{container_type=~"$container_type"}
```

#### Overhead Ratio

Ratio of setup time (Hamiltonian construction plus qubit mapping) to VQE execution time:

```promql
(
    quantum_vqe_hamiltonian_time{container_type=~"$container_type"}
    + quantum_vqe_mapping_time{container_type=~"$container_type"}
)
    / quantum_vqe_vqe_time{container_type=~"$container_type"}
```

#### Energy Error Threshold Alert

Identify simulations that exceed the chemical accuracy threshold:

```promql
quantum_vqe_energy_error_millihartree{molecule_symbols=~".*"} > 1
```

#### Container Resource Saturation

Detect containers approaching resource limits:

```promql
quantum_system_cpu_percent{container_type=~".*"} > 90
quantum_system_memory_percent{container_type=~".*"} > 85
```

#### Active Container Count

```promql
count(quantum_system_uptime_seconds{container_type=~"$container_type"})
```

<figure>
  <img src="https://qp-docs.codextechnologies.org/mkdocs/performance_heatmap.png"
       alt="Performance heatmap showing VQE execution metrics across different container types and molecules">
  <figcaption>Figure 1. Performance heatmap visualization displaying execution time distributions across container configurations and molecular systems.</figcaption>
</figure>

<figure>
  <img src="https://qp-docs.codextechnologies.org/mkdocs/avg_iteration_time_by_molecule.png"
       alt="Bar chart comparing average iteration time across molecules for CPU and GPU configurations">
  <figcaption>Figure 2. Average iteration time by molecule, comparing CPU baseline against GPU-accelerated configurations (GTX 1060 and GTX 1050 Ti).</figcaption>
</figure>

---

## Performance Baselines

!!! note "Experimental Setup"
    The baselines below were collected from a specific thesis experimental setup and are not general benchmarks. Results will vary depending on hardware, system load, and molecule complexity.

Based on experimental results with the sto-3g basis set and L-BFGS-B optimizer, the following performance baselines have been established:

| Configuration | Avg Time per Iteration | Speedup | Total Iterations |
|---------------|----------------------|---------|-----------------|
| CPU (Intel i5-8500) | 4.259 s | 1.00x (baseline) | 8,832 |
| GPU (GTX 1060 6GB) | 2.357 s | 1.81x | 12,057 |
| GPU (GTX 1050 Ti 4GB) | 2.454 s | 1.74x | 10,871 |

GPU acceleration provides the greatest benefit for molecules of medium complexity (8--10 qubits), with speedups reaching up to 2.1x for BeH2. For smaller molecules (4 qubits), GPU overhead can negate the computational advantage. For the more complex cc-pVDZ basis set, speedups of 4.08x (GTX 1060) and 3.53x (GTX 1050 Ti) have been observed.

---

For instructions on visualizing these metrics in Grafana, see [Grafana Dashboards](grafana-dashboards.md).
