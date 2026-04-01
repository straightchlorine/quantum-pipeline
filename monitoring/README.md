# Performance Monitoring

Optional monitoring for system resources, VQE metrics, scientific
accuracy, and batch generation progress. Disabled by default -- enable
via environment variables or constructor parameters.

## What it collects

**System metrics** (background thread): CPU usage, memory, load average,
container uptime. Collected at a configurable interval (default 30s).

**VQE metrics** (per molecule): Total time, hamiltonian/mapping/VQE time
breakdown, minimum energy, iteration count, parameter count.

**Accuracy metrics** (per molecule): Comparison against reference
energies from the built-in database (H2, HeH+, LiH, BeH2, H2O, NH3).
Reports absolute error in Hartree and milli-Hartree, an accuracy score
(0-100), and whether the result is within chemical accuracy (1 mHa).

**Derived metrics**: Iterations per second, time per iteration, overhead
ratio, VQE efficiency (VQE time / total time), setup ratio.

**Batch progress metrics**: Total, done, failed, pending, in-progress
counts and last completion timestamp per tier/lane. Pushed by
`scripts/generate_ml_batch.py`.

## Configuration

Enable monitoring by setting environment variables:

```bash
export MONITORING_ENABLED=true
export PUSHGATEWAY_URL=http://pushgateway:9091
```

| Variable | Default | Description |
|----------|---------|-------------|
| `MONITORING_ENABLED` | `false` | Enable monitoring |
| `MONITORING_INTERVAL` | `30` | Collection interval in seconds |
| `PUSHGATEWAY_URL` | -- | Prometheus PushGateway URL |
| `MONITORING_EXPORT_FORMAT` | `json` | `json`, `prometheus`, or `both` |
| `CONTAINER_TYPE` | `unknown` | Label for the container (set automatically in Docker) |

Constructor parameters take priority over environment variables, which
take priority over `settings.py` defaults.

## Export

Metrics can be exported to:

- **JSON files** in the `metrics/` directory (useful for local debugging)
- **Prometheus PushGateway** (for Grafana dashboards and alerting)

VQE metrics use a grouping key
(`/job/qp-vqe/container_type/.../molecule/.../optimizer/...`) to prevent
overwrites between concurrent simulation runs.

In the Docker Compose stack, PushGateway is already running and Grafana
is pre-configured to scrape it.

## Observability stack

The full monitoring stack includes additional exporters beyond the
PushGateway:

| Exporter | Port | Source |
|----------|------|--------|
| StatsD Exporter | `:9102` | Airflow |
| Postgres Exporter | `:9187` | PostgreSQL |
| Redis Exporter | `:9121` | Redis |
| NVIDIA GPU Exporter | `:9835` | nvidia-smi |
| Redpanda Connect | `:4195` | Internal pipeline metrics |
| Garage Admin | `:3903` | Object storage metrics |

## Usage

Monitoring integrates automatically with `VQERunner` -- if enabled, it
collects snapshots before and after each molecule and exports VQE metrics
to Prometheus. No code changes needed.

For programmatic use:

```python
from quantum_pipeline.monitoring import PerformanceMonitor

with PerformanceMonitor(enabled=True) as monitor:
    monitor.set_experiment_context(molecule_id=0, molecule_symbols='H2')
    # run VQE ...
    monitor.export_vqe_metrics_immediate(vqe_data)
```

## Prometheus metric names

All metrics are prefixed with `qp_`. System metrics use `qp_sys_*`, VQE
metrics use `qp_vqe_*`, batch metrics use `qp_batch_*`. Labels include
`container_type`, `molecule_symbols`, `basis_set`, `optimizer`,
`backend_type`, `tier`, and `lane`.

See `quantum_pipeline/monitoring/performance_monitor.py` for the full
list and their types.

## Thread safety

The `experiment_context` dictionary is protected by `_context_lock`,
making it safe to update context from any thread while the background
monitoring loop is running.
