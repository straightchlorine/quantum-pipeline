# Performance Monitoring Guide

The Quantum Pipeline includes comprehensive performance monitoring capabilities for tracking system resources, VQE optimization metrics, and scientific accuracy in real-time.

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Metrics Collected](#metrics-collected)
- [Prometheus Integration](#prometheus-integration)
- [Grafana Dashboards](#grafana-dashboards)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)

---

## Overview

The performance monitoring system provides:

- **System Metrics**: CPU, memory, I/O, network usage
- **VQE Metrics**: Energy convergence, iteration counts, timing breakdowns
- **Scientific Metrics**: Reference energy comparisons, chemical accuracy tracking
- **Container Metrics**: Docker container resource usage
- **Real-time Export**: Metrics exported to Prometheus PushGateway and JSON files

All monitoring features can be enabled/disabled globally via configuration without code changes.

---

## Architecture

```
┌─────────────────────┐
│  Quantum Pipeline   │
│  (VQE Execution)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ PerformanceMonitor  │
│ - System metrics    │
│ - VQE metrics       │
│ - Background thread │
└──────────┬──────────┘
           │
           ├─────────────────┐
           ▼                 ▼
┌─────────────────┐  ┌──────────────┐
│ Prometheus      │  │ JSON Files   │
│ PushGateway     │  │ (metrics/)   │
└────────┬────────┘  └──────────────┘
         │
         ▼
┌─────────────────┐
│ Grafana         │
│ Dashboards      │
└─────────────────┘
```

---

## Configuration

### Configuration Priority

The monitoring system uses the following configuration priority (highest to lowest):

1. **Constructor parameters** (when initializing `PerformanceMonitor` directly)
2. **Environment variables** (prefixed with `QUANTUM_PERFORMANCE_`)
3. **Settings file** (`quantum_pipeline/configs/settings.py`)

### Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `QUANTUM_PERFORMANCE_ENABLED` | bool | `false` | Enable/disable monitoring globally |
| `QUANTUM_PERFORMANCE_COLLECTION_INTERVAL` | int | `30` | Metrics collection interval (seconds) |
| `QUANTUM_PERFORMANCE_PUSHGATEWAY_URL` | string | - | Prometheus PushGateway URL (e.g., `http://localhost:9091`) |
| `QUANTUM_PERFORMANCE_EXPORT_FORMAT` | list | `json` | Export formats: `json`, `prometheus`, or `both` (comma-separated) |
| `CONTAINER_TYPE` | string | `unknown` | Container type label (auto-set in Docker) |

### Settings File Configuration

Edit `quantum_pipeline/configs/settings.py`:

```python
# Performance Monitoring Configuration
PERFORMANCE_MONITORING_ENABLED = True
PERFORMANCE_COLLECTION_INTERVAL = 30  # seconds
PERFORMANCE_PUSHGATEWAY_URL = 'http://localhost:9091'
PERFORMANCE_EXPORT_FORMAT = ['prometheus', 'json']
PERFORMANCE_METRICS_DIR = Path('./metrics')
```

### Quick Enable/Disable

**Enable via environment**:
```bash
export QUANTUM_PERFORMANCE_ENABLED=true
export QUANTUM_PERFORMANCE_PUSHGATEWAY_URL=http://localhost:9091
```

**Disable temporarily**:
```bash
export QUANTUM_PERFORMANCE_ENABLED=false
```

---

## Metrics Collected

### System Metrics (Background Collection)

Collected continuously in a background thread:

| Metric | Description | Type |
|--------|-------------|------|
| `quantum_system_cpu_percent` | CPU usage percentage | Gauge |
| `quantum_system_cpu_load_1m` | 1-minute load average | Gauge |
| `quantum_system_memory_percent` | Memory usage percentage | Gauge |
| `quantum_system_memory_used_bytes` | Memory used in bytes | Gauge |
| `quantum_system_uptime_seconds` | Container uptime | Counter |

**Labels**: `container_type`

### VQE Experiment Metrics (Event-Driven)

Collected after each molecule completes:

| Metric | Description | Type |
|--------|-------------|------|
| `quantum_vqe_total_time` | Total experiment time (seconds) | Gauge |
| `quantum_vqe_hamiltonian_time` | Hamiltonian generation time | Gauge |
| `quantum_vqe_mapping_time` | Qubit mapping time | Gauge |
| `quantum_vqe_vqe_time` | VQE optimization time | Gauge |
| `quantum_vqe_minimum_energy` | Minimum energy found (Hartree) | Gauge |
| `quantum_vqe_iterations_count` | Total VQE iterations | Gauge |
| `quantum_vqe_optimal_parameters_count` | Number of optimized parameters | Gauge |

**Labels**: `container_type`, `molecule_id`, `molecule_symbols`, `basis_set`, `optimizer`, `backend_type`

### Scientific Accuracy Metrics

| Metric | Description | Type |
|--------|-------------|------|
| `quantum_vqe_reference_energy` | Reference energy from database (Hartree) | Gauge |
| `quantum_vqe_energy_error_hartree` | Absolute error in Hartree | Gauge |
| `quantum_vqe_energy_error_millihartree` | Absolute error in milli-Hartree | Gauge |
| `quantum_vqe_accuracy_score` | Accuracy percentage (0-100) | Gauge |
| `quantum_vqe_within_chemical_accuracy` | 1 if within 1 kcal/mol, 0 otherwise | Gauge |

**Labels**: Same as VQE metrics

### Derived Efficiency Metrics

Automatically calculated from VQE metrics:

| Metric | Description | Formula |
|--------|-------------|---------|
| `quantum_vqe_iterations_per_second` | Optimization speed | `iterations_count / vqe_time` |
| `quantum_vqe_time_per_iteration` | Average iteration time | `vqe_time / iterations_count` |
| `quantum_vqe_overhead_ratio` | Setup overhead vs computation | `(total_time - vqe_time) / vqe_time` |
| `quantum_vqe_efficiency` | VQE time fraction of total | `vqe_time / total_time` |
| `quantum_vqe_setup_ratio` | Setup time fraction | `(hamiltonian_time + mapping_time) / total_time` |

---

## Prometheus Integration

### Setting Up Prometheus PushGateway

**Docker Compose**:
```yaml
pushgateway:
  image: prom/pushgateway:latest
  container_name: pushgateway
  ports:
    - "9091:9091"
  networks:
    - quantum-net
```

**Standalone**:
```bash
docker run -d -p 9091:9091 prom/pushgateway
```

### Configuring Prometheus to Scrape PushGateway

Add to `prometheus.yml`:
```yaml
scrape_configs:
  - job_name: 'quantum-pipeline'
    honor_labels: true
    static_configs:
      - targets: ['pushgateway:9091']
```

### Verifying Metrics

Check PushGateway status:
```bash
curl http://localhost:9091/metrics
```

Check specific job metrics:
```bash
curl http://localhost:9091/metrics/job/quantum-vqe-quantum_pipeline
```

---

## Grafana Dashboards

### Quick Setup

1. **Add Prometheus data source** in Grafana:
   - URL: `http://prometheus:9090`
   - Access: Server (default)

2. **Import dashboard** from `static/grafana-dashboard.json` (if available)

3. **Or create custom panels** using the metrics above

### Recommended Dashboard Panels

**System Overview**:
- CPU Usage: `quantum_system_cpu_percent`
- Memory Usage: `quantum_system_memory_percent`
- Uptime: `quantum_system_uptime_seconds`

**VQE Performance**:
- Convergence: `quantum_vqe_minimum_energy` over time
- Iteration Speed: `quantum_vqe_iterations_per_second`
- Efficiency: `quantum_vqe_efficiency`

**Scientific Accuracy**:
- Energy Error: `quantum_vqe_energy_error_millihartree`
- Accuracy Score: `quantum_vqe_accuracy_score`
- Chemical Accuracy: `quantum_vqe_within_chemical_accuracy`

### Example Queries

**Average VQE time by molecule**:
```promql
avg by (molecule_symbols) (quantum_vqe_vqe_time)
```

**Energy error distribution**:
```promql
histogram_quantile(0.95, sum(rate(quantum_vqe_energy_error_millihartree[5m])) by (le))
```

**Optimizer comparison**:
```promql
avg by (optimizer) (quantum_vqe_iterations_count)
```

---

## Usage Examples

### Programmatic Usage

```python
from quantum_pipeline.monitoring import PerformanceMonitor

# Initialize with custom settings
monitor = PerformanceMonitor(
    enabled=True,
    collection_interval=60,
    pushgateway_url='http://localhost:9091',
    export_format=['prometheus', 'json']
)

# Set experiment context
monitor.set_experiment_context(
    molecule_id=1,
    molecule_symbols='H2',
    basis_set='sto-3g',
    optimizer='L-BFGS-B'
)

# Start background monitoring
monitor.start_monitoring()

# ... run VQE experiment ...

# Export VQE metrics immediately
monitor.export_vqe_metrics_immediate({
    'molecule_id': 1,
    'molecule_symbols': 'H2',
    'basis_set': 'sto-3g',
    'optimizer': 'L-BFGS-B',
    'backend_type': 'aer_simulator_statevector',
    'total_time': 120.5,
    'vqe_time': 100.2,
    'minimum_energy': -1.137,
    'iterations_count': 150
})

# Stop monitoring
monitor.stop_monitoring_thread()
```

### Context Manager Usage

```python
from quantum_pipeline.monitoring import PerformanceMonitor

with PerformanceMonitor(enabled=True) as monitor:
    monitor.set_experiment_context(molecule_id=1)
    # monitoring starts automatically
    # ... run experiments ...
    # monitoring stops automatically on exit
```

### Global Instance Usage

```python
from quantum_pipeline.monitoring import get_performance_monitor, set_experiment_context

# Get global monitor instance
monitor = get_performance_monitor()

# Use convenience functions
set_experiment_context(molecule_id=1, basis_set='sto-3g')
```

### Docker Usage

Enable monitoring in docker-compose:
```yaml
quantum-pipeline:
  environment:
    - QUANTUM_PERFORMANCE_ENABLED=true
    - QUANTUM_PERFORMANCE_PUSHGATEWAY_URL=http://pushgateway:9091
    - QUANTUM_PERFORMANCE_COLLECTION_INTERVAL=30
    - QUANTUM_PERFORMANCE_EXPORT_FORMAT=both
```

---

## Troubleshooting

### Issue: Metrics not appearing in Prometheus

**Check**:
1. PushGateway is running: `curl http://localhost:9091/metrics`
2. Monitoring is enabled: Check logs for "Performance monitoring initialized"
3. PushGateway URL is correct in configuration
4. Prometheus is scraping PushGateway

**Debug**:
```bash
# Check if metrics are being pushed
docker-compose logs quantum-pipeline | grep "metrics exported"

# Verify PushGateway has received metrics
curl http://localhost:9091/metrics | grep quantum_vqe
```

---

### Issue: Background monitoring thread not stopping

**Symptom**: Container takes long to stop
**Cause**: Monitoring thread timeout

**Solution**: Reduce collection interval or increase stop timeout:
```python
# In settings.py or via env var
PERFORMANCE_COLLECTION_INTERVAL = 10  # seconds (lower = faster shutdown)
```

---

### Issue: High memory usage from monitoring

**Cause**: JSON files accumulating in metrics directory

**Solution**: Regularly clean up metrics directory or use only Prometheus export:
```bash
# Clean old metrics files
find ./metrics -name "*.json" -mtime +7 -delete

# Or use Prometheus-only export
export QUANTUM_PERFORMANCE_EXPORT_FORMAT=prometheus
```

---

### Issue: Monitoring slowing down VQE calculations

**Check**:
1. Collection interval is reasonable (30-60 seconds recommended)
2. Export format is not both (choose one)
3. PushGateway is responsive

**Solution**: Increase interval or disable during critical runs:
```bash
export QUANTUM_PERFORMANCE_ENABLED=false  # Temporary disable
```

---

## Advanced Configuration

### Custom Metrics Directory

```python
from pathlib import Path
from quantum_pipeline.monitoring import PerformanceMonitor

monitor = PerformanceMonitor(
    enabled=True,
    metrics_dir=Path('/custom/metrics/path')
)
```

### Selective Monitoring

Monitor only specific experiments:
```python
monitor = PerformanceMonitor(enabled=False)  # Disabled by default

# Enable for specific molecules
if molecule_id in [1, 5, 10]:
    monitor.enabled = True
    monitor.start_monitoring()
```

### Integration with VQERunner

The `VQERunner` automatically integrates with monitoring:
```python
from quantum_pipeline.runners.vqe_runner import VQERunner

runner = VQERunner(
    filepath='data/molecules.json',
    basis_set='sto-3g',
    max_iterations=100
)
# Monitoring is automatically enabled if configured
runner.run(backend)
```

---

## Scientific References Database

The monitoring module includes integration with the scientific references database for accuracy comparisons:

```python
from quantum_pipeline.monitoring.scientific_references import get_reference_database

db = get_reference_database()
reference_energy = db.get_reference_energy('H2', 'sto-3g')
```

See `quantum_pipeline/monitoring/scientific_references.py` for details.

---

---

## External Monitoring Integration

For scenarios where you have a separate monitoring server (common in thesis/research setups), the quantum pipeline can push metrics to an external Prometheus PushGateway.

### External PushGateway Configuration

Update `.env` file:
```bash
# Point to external PushGateway
QUANTUM_PERFORMANCE_PUSHGATEWAY_URL=http://192.168.1.100:9091

# Example for thesis server setup
EXTERNAL_PUSHGATEWAY_URL=http://monitoring-server:9091
```

### Network Configuration

Ensure firewall allows outbound connections to PushGateway:
```bash
# Allow outbound to external PushGateway
sudo ufw allow out to MONITORING_SERVER_IP port 9091
```

### Testing External Connection

From quantum pipeline server:
```bash
# Test PushGateway connectivity
curl -X POST http://monitoring-server:9091/metrics/job/quantum-test

# Verify metrics are received on monitoring server
curl http://monitoring-server:9091/metrics | grep quantum_
```

### Docker Compose with External Monitoring

```yaml
services:
  quantum-pipeline:
    environment:
      - QUANTUM_PERFORMANCE_ENABLED=true
      - QUANTUM_PERFORMANCE_PUSHGATEWAY_URL=http://external-monitoring-server:9091
      - CONTAINER_TYPE=GPU  # or CPU
```

---

## Additional Resources

- [Main README](../README.md) - General project documentation
- [Docker Setup](../docker/README.md) - Docker configuration
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- Module implementation: `quantum_pipeline/monitoring/performance_monitor.py`
