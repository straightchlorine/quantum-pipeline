# Performance Monitoring

Optional monitoring for system resources, VQE metrics, and scientific
accuracy. Disabled by default — enable via environment variables or
constructor parameters.

## What it collects

**System metrics** (background thread): CPU usage, memory, disk I/O,
network I/O, container uptime. Collected at a configurable interval
(default 30s).

**VQE metrics** (per molecule): Total time, hamiltonian/mapping/VQE time
breakdown, minimum energy, iteration count, parameter count.

**Accuracy metrics** (per molecule): Comparison against reference
energies from the built-in database. Reports absolute error in Hartree
and milli-Hartree, an accuracy score (0-100), and whether the result is
within chemical accuracy (1 kcal/mol).

**Derived metrics**: Iterations per second, time per iteration, overhead
ratio, VQE efficiency (VQE time / total time).

## Configuration

Enable monitoring by setting environment variables:

```bash
export QUANTUM_PERFORMANCE_ENABLED=true
export QUANTUM_PERFORMANCE_PUSHGATEWAY_URL=http://pushgateway:9091
```

| Variable | Default | Description |
|----------|---------|-------------|
| `QUANTUM_PERFORMANCE_ENABLED` | `false` | Enable monitoring |
| `QUANTUM_PERFORMANCE_COLLECTION_INTERVAL` | `30` | Collection interval in seconds |
| `QUANTUM_PERFORMANCE_PUSHGATEWAY_URL` | — | Prometheus PushGateway URL |
| `QUANTUM_PERFORMANCE_EXPORT_FORMAT` | `json` | `json`, `prometheus`, or `both` |
| `CONTAINER_TYPE` | `unknown` | Label for the container (set automatically in Docker) |

Constructor parameters take priority over environment variables, which
take priority over `settings.py` defaults.

## Export

Metrics can be exported to:

- **JSON files** in the `metrics/` directory (useful for local debugging)
- **Prometheus PushGateway** (for Grafana dashboards and alerting)

In the Docker Compose stack, PushGateway is already running and Grafana
is pre-configured to scrape it.

## Usage

Monitoring integrates automatically with `VQERunner` — if enabled, it
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

All metrics are prefixed with `quantum_`. System metrics use
`quantum_system_*`, VQE metrics use `quantum_vqe_*`. Labels include
`container_type`, `molecule_symbols`, `basis_set`, `optimizer`, and
`backend_type`.

See `quantum_pipeline/monitoring/performance_monitor.py` for the full
list and their types.
