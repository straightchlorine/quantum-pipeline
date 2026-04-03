# Grafana Dashboards

Grafana provides the visualization layer for monitoring simulations. The repository includes a pre-built
dashboard covering batch generation progress, VQE performance comparison, scientific accuracy,
GPU utilization, system resources, data platform health, databases, and orchestration.

## Setup

The main Docker Compose file [`compose/docker-compose.ml.yaml`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/compose/docker-compose.ml.yaml) defines only exporters. It assumes available Prometheus and Grafana instances.

For a basic Prometheus and Grafana setup, see the [Grafana documentation](https://grafana.com/docs/grafana-cloud/send-data/metrics/metrics-prometheus/prometheus-config-examples/docker-compose-linux/).

## Dashboard Import

### Auto-provisioning

If Grafana is configured with the provisioning file at [`monitoring/grafana/provisioning/dashboards/dashboard.yml`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/monitoring/grafana/provisioning/dashboards/dashboard.yml), dashboards are loaded automatically from `/var/lib/grafana/dashboards` inside the container. Mount the dashboard JSON there and Grafana picks it up on startup.

### Manual Import

The primary dashboard is located at [`monitoring/grafana/dashboards/quantum-pipeline.json`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/monitoring/grafana/dashboards/quantum-pipeline.json).

To import manually:

1. In Grafana, navigate to **Dashboards** > **Import**.
2. Click **Upload JSON file** and select `quantum-pipeline.json` from the repository.
3. Alternatively, paste the JSON content directly into the **Import via panel json** text field.
4. In the **Prometheus** data source dropdown, select your configured Prometheus instance.
5. Click **Import**.

The dashboard will appear under the name **Quantum Pipeline - ML Stack**.

### Template Variables

The dashboard includes template variables for interactive filtering:

| Variable | Label | Description | Source |
|----------|-------|-------------|--------|
| `DS_PROMETHEUS` | Data Source | Prometheus data source selector | Data source query |
| `tier` | Tier | Batch generation tier | `label_values(qp_batch_total, tier)` |
| `lane` | Lane | Batch generation lane | `label_values(qp_batch_done, lane)` |
| `optimizer` | Optimizer | Filter by optimization algorithm | `label_values(qp_vqe_total_time, optimizer)` |
| `molecule` | Molecule | Filter by molecule symbols | `label_values(qp_vqe_total_time, molecule_symbols)` |
| `container_type` | Container Type | Filter by container configuration (cpu, gpu1, gpu2) | `label_values(qp_vqe_total_time, container_type)` |

All variables support multi-select and an "All" option for viewing data across all configurations simultaneously.

## Dashboard Layout

The dashboard has 73 panels organized into 9 rows. Six rows (VQE Performance, Batch Generation, System Resources, Data Platform, Databases, Orchestration) are collapsed by default; three (Quality Metrics, GPU, Block Storage) are expanded.

### Row 1: VQE Performance (7 panels)

Detailed performance analysis across container configurations.

| Panel | Type | PromQL |
|-------|------|--------|
| VQE Efficiency | Time Series | `qp_vqe_efficiency{...}` |
| Overhead Ratio | Time Series | `qp_vqe_overhead_ratio{...}` |
| Iterations per Second | Time Series | `qp_vqe_iterations_per_second{...}` |
| Setup Ratio | Time Series | `qp_vqe_setup_ratio{...}` |
| VQE Total Time | Time Series | `qp_vqe_total_time{...}` |
| Time per Iteration | Time Series | `qp_vqe_time_per_iteration{...}` |
| Iteration Count | Time Series | `qp_vqe_iterations_count{...}` |

### Row 2: Batch Generation (7 panels)

Tracks the state and throughput of ML data generation runs.

| Panel | Type | PromQL |
|-------|------|--------|
| Pending | Stat | `sum(qp_batch_pending{tier=~"$tier"})` |
| Progress | Gauge | `sum(qp_batch_done{tier=~"$tier"}) / sum(qp_batch_total{tier=~"$tier"}) * 100` |
| Throughput (runs/hour) | Time Series | `increase(qp_batch_done{tier=~"$tier",lane=~"$lane"}[10m]) * 6` |
| Failed Over Time | Time Series | `qp_batch_failed{tier=~"$tier",lane=~"$lane"}` |
| Done | Stat | `sum(qp_batch_done{tier=~"$tier"})` |
| Failed | Stat | `sum(qp_batch_failed{tier=~"$tier"})` |
| Per-Lane Progress | Time Series | `qp_batch_done{tier=~"$tier",lane=~"$lane"}` |

### Row 3: Quality Metrics (4 panels)

Monitors the scientific quality of simulation results.

| Panel | Type | PromQL |
|-------|------|--------|
| Accuracy Score | Time Series | `qp_vqe_accuracy_score{...}` |
| Reference Energy | Time Series | `qp_vqe_reference_energy{...}` |
| Ground State Energy | Time Series | `qp_vqe_minimum_energy{...}` |
| Energy Error (mHa) | Time Series | `qp_vqe_energy_error_millihartree{...}` |

### Row 4: GPU (8 panels)

GPU metrics sourced from `nvidia_gpu_exporter` on port `:9835`.

| Panel | Description |
|-------|-------------|
| GPU Utilization | GPU core utilization percentage |
| GPU Power Draw | Power consumption in watts |
| Fan Speed | Fan speed percentage |
| Power State | Current GPU power state |
| Graphics Clock (MHz) | Core clock frequency |
| Memory Free | Available GPU memory |
| GPU Temperature | Temperature in Celsius |
| GPU Memory Usage | Memory utilization percentage |

### Row 5: System Resources (4 panels)

Hardware utilization for each simulation container.

| Panel | PromQL |
|-------|--------|
| CPU Usage | `qp_sys_cpu_percent{container_type=~"$container_type"}` |
| Memory Usage | `qp_sys_memory_percent{container_type=~"$container_type"}` |
| Uptime | `qp_sys_uptime_seconds{container_type=~"$container_type"}` |
| Load Average (1m) | `qp_sys_cpu_load_1m{container_type=~"$container_type"}` |

### Row 6: Data Platform (8 panels)

Redpanda Connect pipeline metrics from the internal metrics endpoint on port `:4195`.

| Panel | Description |
|-------|-------------|
| Redpanda Input Latency (ms) | Input processing latency |
| Redpanda Output Latency (ms) | Output processing latency |
| Redpanda Batch Throughput | Message batch processing rate |
| Redpanda Errors/sec | Error rate from `output_error_total` |
| Redpanda Output Connection | Output connection status |
| Redpanda Input Connection | Input connection status |
| Redpanda Messages In/sec | Inbound message throughput |
| Redpanda Messages Out/sec | Outbound message throughput |

### Row 7: Databases (18 panels)

Metrics from `postgres-exporter` on port `:9187` and `redis-exporter` on port `:9121`.

Covers: exporter health, connections, cache hit ratio, transactions/sec, deadlocks, tuple operations, DB size, active queries, locks (Postgres); keys, evicted keys, network I/O, uptime, hit rate, ops/sec, memory, clients (Redis).

### Row 8: Block Storage (8 panels)

Garage metrics from the admin API on port `:3903`.

Covers: disk usage, stored blocks, cluster health, connected nodes, storage nodes, partitions OK, resync queue, admin request rate.

### Row 9: Orchestration (9 panels)

Airflow metrics from the StatsD exporter on port `:9102`.

| Panel | Description |
|-------|-------------|
| DAGBag Size | Number of DAGs loaded |
| Airflow Task Instances | Task instance counts by state |
| DAG Parse Time | Time to parse DAG files |
| Scheduler Executable Tasks | Tasks ready for scheduling |
| Triggerer Capacity | Triggerer slot utilization |
| Executor Tasks | Tasks in executor by state |
| Scheduler Heartbeat | Scheduler liveness |
| DAG Import Errors | DAG loading failures |
| Pool Slots (default_pool) | Pool slot utilization |


## Alerting

Alert rules are provisioned from
[`monitoring/grafana/provisioning/alerting/rules.yml`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/monitoring/grafana/provisioning/alerting/rules.yml).
They cover batch stalls, accuracy degradation, resource saturation, GPU
temperature, and service availability. Alerting is present in the configuration
but has not been tested yet.

For a complete reference of available metrics and their Prometheus names, see [Performance Metrics](performance-metrics.md).
