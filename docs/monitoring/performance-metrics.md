# Performance Metrics

The simulation module collects metrics during VQE simulation execution, organized into three categories: system metrics, VQE execution metrics, and batch progress metrics. All metrics are exported by [`performance_monitor.py`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/monitoring/performance_monitor.py).

## Metric Categories

### System Metrics (`qp_sys_*`)

System metrics track hardware resource utilization for each simulation container. A background monitoring thread collects them and pushes to the Prometheus PushGateway at regular intervals.

| Metric | Prometheus Name | Type | Unit | Description |
|--------|----------------|------|------|-------------|
| CPU Usage | `qp_sys_cpu_percent` | Gauge | % | Current CPU utilization percentage |
| Memory Usage | `qp_sys_memory_percent` | Gauge | % | Current memory utilization percentage |
| Memory Used | `qp_sys_memory_used_bytes` | Gauge | bytes | Current memory consumption in bytes |
| CPU Load (1m) | `qp_sys_cpu_load_1m` | Gauge | load | 1-minute load average |
| Container Uptime | `qp_sys_uptime_seconds` | Gauge | seconds | Time since container start |

!!! note "GPU Metrics"
    GPU-specific Prometheus metrics (utilization, memory, temperature) are not exported by `PerformanceMonitor` directly. GPU usage varies too rapidly to produce reliable statistics from within the simulation process. Instead, GPU metrics are collected by [`nvidia_gpu_exporter`](https://github.com/utkuozdemir/nvidia_gpu_exporter) on port `:9835`, which reads `nvidia-smi` at a 30s interval. The Grafana dashboard has a dedicated GPU row with 8 panels sourced from this exporter.

### VQE Execution Metrics (`qp_vqe_*`)

VQE execution metrics are exported after each simulation completes. These include timing breakdowns, result values, and derived efficiency metrics.

| Metric | Prometheus Name | Type | Description |
|--------|----------------|------|-------------|
| Total Execution Time | `qp_vqe_total_time` | Gauge | End-to-end simulation time |
| Hamiltonian Build Time | `qp_vqe_hamiltonian_time` | Gauge | Time to construct the molecular Hamiltonian |
| Qubit Mapping Time | `qp_vqe_mapping_time` | Gauge | Time for fermionic-to-qubit operator mapping |
| VQE Optimization Time | `qp_vqe_vqe_time` | Gauge | Time spent in VQE optimization loop |
| Minimum Energy | `qp_vqe_minimum_energy` | Gauge | Best energy found during optimization (Hartree) |
| Iterations Count | `qp_vqe_iterations_count` | Gauge | Total optimizer iterations to convergence |
| Optimal Parameters | `qp_vqe_optimal_parameters_count` | Gauge | Number of optimized variational parameters |

### Scientific Accuracy Metrics (`qp_vqe_*`)

!!! warning "Experimental Feature"
    Scientific accuracy tracking is still in testing. Most of the thesis research was conducted without it. The reference values are drawn from an already scarce bibliography - they should be treated as best-effort baselines rather than authoritative benchmarks.

Scientific metrics evaluate the quality of VQE results against known reference values. These are exported alongside VQE execution metrics.

| Dashboard Metric | Prometheus Name | Type | Description |
|-----------------|----------------|------|-------------|
| Reference Energy | `qp_vqe_reference_energy` | Gauge | Literature reference energy for the molecule |
| Energy Error (Ha) | `qp_vqe_energy_error_hartree` | Gauge | Deviation from reference value in Hartree |
| Energy Error (mHa) | `qp_vqe_energy_error_millihartree` | Gauge | Deviation from reference value in mHa |
| Accuracy Score | `qp_vqe_accuracy_score` | Gauge | Normalized accuracy score (0 to 100) |

### Derived Efficiency Metrics (`qp_vqe_*`)

These are calculated from the timing and iteration data and pushed alongside VQE metrics.

| Metric | Prometheus Name | Type | Description |
|--------|----------------|------|-------------|
| Iterations per Second | `qp_vqe_iterations_per_second` | Gauge | `iterations_count / vqe_time` |
| Time per Iteration | `qp_vqe_time_per_iteration` | Gauge | `vqe_time / iterations_count` |
| Overhead Ratio | `qp_vqe_overhead_ratio` | Gauge | `(total_time - vqe_time) / vqe_time` |
| VQE Efficiency | `qp_vqe_efficiency` | Gauge | `vqe_time / total_time` |
| Setup Ratio | `qp_vqe_setup_ratio` | Gauge | `(hamiltonian_time + mapping_time) / total_time` |

### Batch Progress Metrics (`qp_batch_*`)

Batch progress metrics are pushed by [`scripts/generate_ml_batch.py`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/scripts/generate_ml_batch.py) and track the state of ML data generation runs across tiers and lanes.

| Metric | Prometheus Name | Type | Labels | Description |
|--------|----------------|------|--------|-------------|
| Total Runs | `qp_batch_total` | Gauge | `tier` | Total number of runs in the batch |
| Completed | `qp_batch_done` | Gauge | `tier`, `lane` | Runs completed successfully |
| Failed | `qp_batch_failed` | Gauge | `tier`, `lane` | Runs that failed |
| Pending | `qp_batch_pending` | Gauge | `tier`, `lane` | Runs waiting to start |
| In Progress | `qp_batch_in_progress` | Gauge | `tier`, `lane` | Runs currently executing |
| Last Completion | `qp_batch_last_completion_ts` | Gauge | `tier`, `lane` | Unix timestamp of most recent completion |

## Labels

Every metric includes the following labels where applicable:

| Label | Description | Example Values |
|-------|-------------|----------------|
| `container_type` | Simulation container configuration | `cpu`, `gpu1`, `gpu2` |
| `molecule_symbols` | Chemical formula of the molecule | `H2`, `LiH`, `BeH2`, `H2O`, `NH3` |
| `optimizer` | Optimization algorithm | `L-BFGS-B`, `COBYLA`, `SLSQP` |
| `basis_set` | Basis set for the simulation | `sto-3g`, `cc-pvdz` |
| `molecule_id` | Unique identifier for the molecule instance | `0`, `1`, `2` |
| `backend_type` | Quantum simulation backend | `aer_simulator`, `statevector` |
| `tier` | Batch generation tier (batch metrics only) | `sto3g`, `ccpvdz` |
| `lane` | Batch generation lane (batch metrics only) | `cpu`, `gpu1`, `gpu2` |

## Full Metric List

```promql
# System Resource Metrics
qp_sys_cpu_percent{container_type}
qp_sys_memory_percent{container_type}
qp_sys_memory_used_bytes{container_type}
qp_sys_cpu_load_1m{container_type}
qp_sys_uptime_seconds{container_type}

# VQE Performance Metrics
qp_vqe_total_time{container_type, molecule_id, molecule_symbols, optimizer, backend_type, basis_set}
qp_vqe_vqe_time{container_type, molecule_id, molecule_symbols, optimizer, backend_type, basis_set}
qp_vqe_hamiltonian_time{container_type, molecule_id, molecule_symbols, optimizer, backend_type, basis_set}
qp_vqe_mapping_time{container_type, molecule_id, molecule_symbols, optimizer, backend_type, basis_set}
qp_vqe_iterations_count{container_type, molecule_id, molecule_symbols, optimizer, backend_type, basis_set}
qp_vqe_minimum_energy{container_type, molecule_id, molecule_symbols, optimizer, backend_type, basis_set}
qp_vqe_optimal_parameters_count{container_type, molecule_id, molecule_symbols, optimizer, backend_type, basis_set}

# Scientific Accuracy Metrics
qp_vqe_reference_energy{container_type, molecule_id, molecule_symbols, optimizer, backend_type, basis_set}
qp_vqe_energy_error_hartree{container_type, molecule_id, molecule_symbols, optimizer, backend_type, basis_set}
qp_vqe_energy_error_millihartree{container_type, molecule_id, molecule_symbols, optimizer, backend_type, basis_set}
qp_vqe_accuracy_score{container_type, molecule_id, molecule_symbols, optimizer, backend_type, basis_set}

# Derived Efficiency Metrics
qp_vqe_iterations_per_second{container_type, molecule_id, molecule_symbols, optimizer, backend_type, basis_set}
qp_vqe_time_per_iteration{container_type, molecule_id, molecule_symbols, optimizer, backend_type, basis_set}
qp_vqe_overhead_ratio{container_type, molecule_id, molecule_symbols, optimizer, backend_type, basis_set}
qp_vqe_efficiency{container_type, molecule_id, molecule_symbols, optimizer, backend_type, basis_set}
qp_vqe_setup_ratio{container_type, molecule_id, molecule_symbols, optimizer, backend_type, basis_set}

# Batch Progress Metrics
qp_batch_total{tier}
qp_batch_done{tier, lane}
qp_batch_failed{tier, lane}
qp_batch_pending{tier, lane}
qp_batch_in_progress{tier, lane}
qp_batch_last_completion_ts{tier, lane}
```

## Collection Configuration

### Environment Variables

Monitoring is configured via environment variables (or constructor parameters, which take priority):

| Variable | Default | Description |
|----------|---------|-------------|
| `MONITORING_ENABLED` | `false` | Enable monitoring |
| `MONITORING_INTERVAL` | `10` | Collection interval in seconds |
| `PUSHGATEWAY_URL` | `http://localhost:9091` | Prometheus PushGateway URL |
| `MONITORING_EXPORT_FORMAT` | `prometheus` | `json`, `prometheus`, or `both` |
| `CONTAINER_TYPE` | `unknown` | Label for the container (set automatically in Docker) |

Constructor parameters take priority over environment variables, which take priority over `settings.py` defaults. See the [`PerformanceMonitor.__init__`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/monitoring/performance_monitor.py#L46) method for details.

### Enabling Monitoring

Activate monitoring with environment variables:

```bash
export MONITORING_ENABLED=true
export PUSHGATEWAY_URL=http://pushgateway:9091
```

When enabled, the container will:

1. Collect CPU and memory metrics in a background thread
2. Push system metrics to the PushGateway at configurable intervals
3. Export VQE metrics (iterations, energy, timing) on completion

See [Environment Variables](../deployment/environment-variables.md) for the full list of `MONITORING_*` variables.

!!! info "Monitoring Overhead"
    The monitoring thread introduces minimal overhead and runs independently of VQE computation.

### Export Formats

Metrics can be exported in multiple formats:

=== "Prometheus (Default)"

    Metrics are pushed to the PushGateway in Prometheus exposition format. Scrape targets are configured in [`prometheus.yml`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/monitoring/prometheus.yml).

    ```bash
    export MONITORING_ENABLED=true
    export MONITORING_EXPORT_FORMAT=prometheus
    export PUSHGATEWAY_URL=http://pushgateway:9091
    ```

=== "JSON"

    Metrics are saved to a local JSON file after simulation completion:

    ```bash
    export MONITORING_ENABLED=true
    export MONITORING_EXPORT_FORMAT=json
    ```

=== "Both"

    Export to both Prometheus and JSON simultaneously:

    ```bash
    export MONITORING_ENABLED=true
    export MONITORING_EXPORT_FORMAT=both
    export PUSHGATEWAY_URL=http://pushgateway:9091
    ```

!!! note "PushGateway Hostname"
    The PushGateway hostname depends on where Prometheus is running. The default `http://localhost:9091` works for local development. In the Docker Compose stack, set it to match your PushGateway service name (e.g., `http://pushgateway:9091`).

### PushGateway Grouping Key

VQE metrics are pushed with a grouping key that prevents overwrites between concurrent simulation runs:

```
/metrics/job/qp-vqe/container_type/{type}/molecule/{symbol}/optimizer/{optimizer}
```

System metrics use a separate job name: `qp-sys-{container_type}`.

## Alerting

Alerting rules are defined at
[`monitoring/grafana/provisioning/alerting/rules.yml`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/monitoring/grafana/provisioning/alerting/rules.yml)
and cover batch generation stalls, accuracy degradation, resource saturation,
GPU overheating, and service availability. Alerting is present in the
configuration but has not been tested yet and is not covered in detail here.

For instructions on visualizing metrics in Grafana, see [Grafana Dashboards](grafana-dashboards.md).
