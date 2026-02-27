# Configuration Reference

Complete reference for all command-line parameters and configuration options available in the Quantum Pipeline.

---

## Overview

The Quantum Pipeline uses a hierarchical configuration system:

```mermaid
graph LR
    A[CLI Arguments] -->|Override| B[Config File --load]
    B -->|Fallback| C[Defaults]

    style A fill:#1a73e8,color:#fff,stroke:#1557b0
    style B fill:#e8710a,color:#fff,stroke:#c45e08
    style C fill:#34a853,color:#fff,stroke:#2d8f47
```

!!! tip "Configuration Sources"
    - **CLI Arguments**: Direct command-line flags (highest priority)
    - **Configuration File**: Loaded via `--load config.json`
    - **Defaults**: [`quantum_pipeline/configs/defaults.py`](https://github.com/straightchlorine/quantum-pipeline/blob/master/quantum_pipeline/configs/defaults.py)

!!! note "Environment Variables"
    Some environment variables are supported for performance monitoring (`QUANTUM_PERFORMANCE_*`) and Docker deployments (`CONTAINER_TYPE`), but they do not override CLI arguments or config file values.

---

## Quick Reference Table

| Category | Parameter | Type | Default | Description |
|----------|-----------|------|---------|-------------|
| Required | `--file` | path | - | Molecule data file (JSON) |
| Simulation | `--basis` | choice | `sto3g` | Basis set selection |
| Simulation | `--ansatz-reps` | int | `2` | Ansatz repetitions |
| Simulation | `--ibm` | flag | `false` | Use IBM Quantum backend |
| Simulation | `--min-qubits` | int | `None` | Minimum qubit requirement |
| VQE | `--max-iterations` | int | `100` | Maximum iterations |
| VQE | `--convergence` | flag | `false` | Enable convergence mode |
| VQE | `--threshold` | float | `1e-6` | Convergence threshold |
| VQE | `--optimizer` | choice | `L-BFGS-B` | Optimization algorithm |
| Output | `--output-dir` | path | `./gen` | Output directory |
| Output | `--log-level` | choice | `INFO` | Logging verbosity |
| Backend | `--shots` | int | `1024` | Circuit execution shots |
| Backend | `--optimization-level` | choice | `3` | Circuit optimization (0-3) |
| Features | `--report` | flag | `false` | Generate PDF report |
| Features | `--dump` | flag | `false` | Save configuration to JSON |
| Features | `--load` | path | - | Load configuration from JSON |
| Features | `--gpu` | flag | `false` | Enable GPU acceleration |
| Features | `--simulation-method` | choice | `tensor_network` | Backend method |
| Features | `--noise` | string | `None` | Noise model backend |
| Monitoring | `--enable-performance-monitoring` | flag | `false` | Enable metrics collection |
| Monitoring | `--performance-interval` | int | `30` | Metrics interval (seconds) |
| Monitoring | `--performance-pushgateway` | string | - | Pushgateway URL |
| Monitoring | `--performance-export-format` | choice | `both` | Export format |
| Kafka | `--kafka` | flag | `false` | Enable Kafka streaming |
| Kafka | `--servers` | string | `localhost:9092` | Kafka bootstrap servers |
| Kafka | `--topic` | string | `vqe_decorated_result` | Kafka topic name |
| Kafka | `--retries` | int | `3` | Send retry attempts |
| Kafka | `--retry-delay` | int | `2` | Retry delay (seconds) |
| Kafka | `--internal-retries` | int | `0` | Kafka internal retries |
| Kafka | `--acks` | choice | `all` | Acknowledgment level |
| Kafka | `--timeout` | int | `10` | Request timeout (seconds) |
| Security | `--ssl` | flag | `false` | Enable SSL/TLS |
| Security | `--disable-ssl-check-hostname` | flag | `true` | Disable SSL hostname check |
| Security | `--sasl-ssl` | flag | `false` | Enable SASL_SSL |
| Security | `--ssl-password` | string | `None` | SSL password |
| Security | `--ssl-dir` | path | `./secrets/` | SSL certificates directory |
| Security | `--ssl-cafile` | path | `None` | CA certificate file |
| Security | `--ssl-certfile` | path | `None` | Client certificate file |
| Security | `--ssl-keyfile` | path | `None` | Private key file |
| Security | `--ssl-crlfile` | path | `None` | Certificate revocation list |
| Security | `--ssl-ciphers` | string | `None` | SSL cipher suite |
| Security | `--sasl-mechanism` | choice | - | SASL authentication method |
| Security | `--sasl-plain-username` | string | `None` | SASL username |
| Security | `--sasl-plain-password` | string | `None` | SASL password |
| Security | `--sasl-kerberos-service-name` | string | `kafka` | Kerberos service name |
| Security | `--sasl-kerberos-domain-name` | string | `None` | Kerberos domain |

---

## Required Arguments

### `--file` / `-f`

**Type**: `string` (path)
**Required**: Yes
**Default**: None

Path to the molecule data file in JSON format.

=== "Example Usage"

    ```bash
    python quantum_pipeline.py --file data/molecules.json
    ```

=== "JSON Format"

    ```json
    [
        {
            "symbols": ["H", "H"],
            "coords": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
            "multiplicity": 1,
            "charge": 0,
            "units": "angstrom",
            "masses": [1.008, 1.008]
        }
    ]
    ```

=== "Validation"

    - File must exist and be readable
    - Must contain valid JSON
    - Must follow molecule schema
    - Arrays must have matching lengths

!!! warning "File Not Found"
    If the file doesn't exist, the pipeline will raise an `ArgumentTypeError` and exit.

---

## Simulation Configuration

### `--basis` / `-b`

**Type**: `choice`
**Default**: `sto3g`
**Choices**: `sto3g`, `6-31g`, `cc-pvdz`

Basis set for quantum chemistry calculations.

=== "Usage"

    ```bash
    # Quick simulations
    python quantum_pipeline.py -f molecules.json --basis sto3g

    # Production quality
    python quantum_pipeline.py -f molecules.json --basis 6-31g

    # High accuracy
    python quantum_pipeline.py -f molecules.json --basis cc-pvdz
    ```

=== "Comparison"

    | Basis Set | Accuracy | Speed | Memory | Qubits (H₂O) |
    |-----------|----------|-------|--------|--------------|
    | `sto3g` | Low | Very High | Low | 14 |
    | `6-31g` | Medium | Medium | Medium | 26 |
    | `cc-pvdz` | Very High | Low | High | 58 |

!!! info "Basis Set Selection"
    - Use `sto3g` for prototyping and debugging
    - Use `6-31g` for balanced accuracy/performance
    - Use `cc-pvdz` for publication-quality results

### `--ansatz-reps` / `-ar`

**Type**: `int`
**Default**: `2`

Number of repetitions for the parameterized ansatz circuit.

```bash
# Minimal ansatz (fast, less expressive)
python quantum_pipeline.py -f molecules.json --ansatz-reps 1

# Default (balanced)
python quantum_pipeline.py -f molecules.json --ansatz-reps 2

# Deep ansatz (slower, more expressive)
python quantum_pipeline.py -f molecules.json --ansatz-reps 5
```

!!! tip "Choosing Ansatz Repetitions"
    - **Small molecules (H₂, HeH⁺)**: 1-2 reps sufficient
    - **Medium molecules (H₂O, NH₃)**: 2-3 reps recommended
    - **Large molecules (CO₂, benzene)**: 4-6 reps may be needed
    - More reps = more parameters = longer optimization time

### `--ibm`

**Type**: `flag` (action: `store_false`)
**Default**: `true` (local simulator)

Use IBM Quantum backend instead of local Aer simulator.

```bash
# Use IBM Quantum (requires account)
python quantum_pipeline.py -f molecules.json --ibm

# Use local simulator (default)
python quantum_pipeline.py -f molecules.json
```

!!! warning "IBM Quantum Requirements"
    - Requires IBM Quantum account and API token
    - Set environment variables: `IBM_RUNTIME_TOKEN`, `IBM_RUNTIME_INSTANCE`
    - Subject to queue times and resource limits
    - Cannot be used with `--gpu` flag

### `--min-qubits`

**Type**: `int`
**Default**: `None`

Minimum number of qubits required for the backend selection.

```bash
python quantum_pipeline.py -f molecules.json --ibm --min-qubits 20
```

!!! info "Validation Rule"
    - Only valid when `--ibm` is specified
    - Filters available IBM backends by qubit count
    - Raises error if used with local simulator

---

## VQE Parameters

### `--max-iterations`

**Type**: `int`
**Default**: `100`

Maximum number of VQE optimization iterations.

=== "Usage Examples"

    ```bash
    # Quick test (small molecules)
    python quantum_pipeline.py -f molecules.json --max-iterations 50

    # Standard simulation
    python quantum_pipeline.py -f molecules.json --max-iterations 100

    # High-accuracy (complex molecules)
    python quantum_pipeline.py -f molecules.json --max-iterations 500
    ```

=== "Recommendations"

    | Molecule Size | Basis Set | Recommended Iterations |
    |---------------|-----------|------------------------|
    | H₂, HeH⁺ | sto3g | 50-100 |
    | H₂O, NH₃ | sto3g | 100-200 |
    | CO₂, N₂ | sto3g | 200-500 |
    | Any | cc-pvdz | 300-1000 |

!!! danger "Mutual Exclusivity"
    **Cannot be used with `--convergence`**. These parameters are mutually exclusive and will raise a `ValueError` if both are specified.

### `--convergence`

**Type**: `flag` (action: `store_true`)
**Default**: `false`

Enable convergence-based optimization instead of fixed iterations.

```bash
# Enable convergence mode with default threshold
python quantum_pipeline.py -f molecules.json --convergence --threshold 1e-6

# High precision convergence
python quantum_pipeline.py -f molecules.json --convergence --threshold 1e-8
```

!!! info "Convergence Mode"
    - Stops when energy change falls below `--threshold`
    - Automatically determines optimal iteration count
    - Recommended for production runs
    - Requires `--threshold` to be set

!!! danger "Mutual Exclusivity"
    **Cannot be used with `--max-iterations`**. Choose one or the other.

### `--threshold`

**Type**: `float`
**Default**: `1e-6`

Convergence threshold for VQE optimization (Hartrees).

=== "Usage"

    ```bash
    # Standard convergence (1 μHa)
    python quantum_pipeline.py -f molecules.json --convergence --threshold 1e-6

    # High precision (10 nHa)
    python quantum_pipeline.py -f molecules.json --convergence --threshold 1e-8

    # Chemical accuracy (~1 kcal/mol = 1.6 mHa)
    python quantum_pipeline.py -f molecules.json --convergence --threshold 1.6e-3
    ```

=== "Threshold Guide"

    | Threshold | Energy Units | Use Case |
    |-----------|--------------|----------|
    | `1e-3` | 1 mHa (chemical accuracy) | Fast prototyping |
    | `1e-6` | 1 μHa | **Standard production** |
    | `1e-8` | 10 nHa | High-precision research |
    | `1e-10` | 0.1 nHa | Extreme precision |

!!! tip "Chemical Accuracy"
    Chemical accuracy is typically ~1 kcal/mol ≈ 1.6 mHa. Use `--threshold 1.6e-3` for this benchmark.

### `--optimizer`

**Type**: `choice`
**Default**: `L-BFGS-B`

Optimization algorithm for VQE parameter updates.

=== "Available Optimizers"

    **Gradient-Based (Recommended)**

    - `L-BFGS-B` - Limited-memory BFGS with bounds (default, best for GPU)
    - `BFGS` - Broyden-Fletcher-Goldfarb-Shanno
    - `CG` - Conjugate gradient method
    - `Newton-CG` - Newton conjugate gradient
    - `TNC` - Truncated Newton with bounds

    **Trust-Region Methods**

    - `trust-constr` - Trust-region constrained optimization
    - `trust-ncg` - Trust-region Newton conjugate gradient
    - `trust-exact` - Trust-region exact Hessian
    - `trust-krylov` - Trust-region Krylov method
    - `dogleg` - Dog-leg trust-region algorithm

    **Derivative-Free**

    - `COBYLA` - Constrained optimization by linear approximation
    - `COBYQA` - Constrained optimization by quadratic approximation
    - `Powell` - Powell's method
    - `Nelder-Mead` - Simplex algorithm

    **Sequential Methods**

    - `SLSQP` - Sequential least squares programming

    **Custom**

    - `custom` - User-defined optimization function

=== "Usage Examples"

    ```bash
    # Default (recommended)
    python quantum_pipeline.py -f molecules.json --optimizer L-BFGS-B

    # Derivative-free (noisy landscapes)
    python quantum_pipeline.py -f molecules.json --optimizer COBYLA

    # Sequential optimization
    python quantum_pipeline.py -f molecules.json --optimizer SLSQP
    ```

=== "Optimizer Comparison"

    | Optimizer | Type | Speed | Accuracy | Best For |
    |-----------|------|-------|----------|----------|
    | `L-BFGS-B` | Gradient | High | High | **Default choice** |
    | `BFGS` | Gradient | High | High | Smooth landscapes |
    | `COBYLA` | Derivative-free | Medium | Medium | Noisy cost functions |
    | `Powell` | Derivative-free | Low | Medium | Simple problems |
    | `SLSQP` | Sequential | High | Medium | Constrained optimization |

!!! tip "Optimizer Selection"
    - **Default**: Use `L-BFGS-B` for best balance of speed and accuracy
    - **GPU Acceleration**: `L-BFGS-B` performs exceptionally well with GPU
    - **Noisy Simulations**: Use `COBYLA` for better robustness
    - **Large Molecules**: Gradient-based optimizers converge faster

For detailed optimizer information, see [Optimizer Guide](optimizers.md).

---

## Output and Logging

### `--output-dir`

**Type**: `string` (path)
**Default**: `./gen`

Directory for storing output files, graphs, and reports.

```bash
python quantum_pipeline.py -f molecules.json --output-dir /path/to/output
```

!!! info "Auto-Created Subdirectories"
    The pipeline automatically creates:

    - `graphs/` - Visualization plots
    - `molecule_plots/` - 3D molecular structures
    - `operator_plots/` - Hamiltonian coefficients
    - `energy_plots/` - Convergence curves
    - `performance_metrics/` - Performance data (if monitoring enabled)

### `--log-level`

**Type**: `choice`
**Default**: `INFO`
**Choices**: `DEBUG`, `INFO`, `WARNING`, `ERROR`

Set the logging verbosity level.

=== "Usage"

    ```bash
    # Detailed debugging output
    python quantum_pipeline.py -f molecules.json --log-level DEBUG

    # Standard informational messages
    python quantum_pipeline.py -f molecules.json --log-level INFO

    # Only warnings and errors
    python quantum_pipeline.py -f molecules.json --log-level WARNING

    # Errors only
    python quantum_pipeline.py -f molecules.json --log-level ERROR
    ```

=== "Log Level Guide"

    | Level | When to Use | Output Volume |
    |-------|-------------|---------------|
    | `DEBUG` | Development, troubleshooting | Very High |
    | `INFO` | Production monitoring | **Medium (default)** |
    | `WARNING` | Production (quiet) | Low |
    | `ERROR` | Production (minimal) | Very Low |

!!! tip "Production Logging"
    Use `INFO` for production to balance observability with performance. Switch to `DEBUG` only when troubleshooting specific issues.

---

## Advanced Backend Options

### `--shots`

**Type**: `int` (positive)
**Default**: `1024`

Number of shots (measurement samples) for quantum circuit execution.

=== "Usage"

    ```bash
    # Fast simulation (less accurate)
    python quantum_pipeline.py -f molecules.json --shots 512

    # Standard (balanced)
    python quantum_pipeline.py -f molecules.json --shots 1024

    # High statistics (more accurate)
    python quantum_pipeline.py -f molecules.json --shots 4096
    ```

=== "Shot Count Guide"

    | Shots | Statistical Error | Runtime | Use Case |
    |-------|-------------------|---------|----------|
    | 512 | ~4.4% | Fast | Quick tests |
    | 1024 | ~3.1% | **Standard** | Production default |
    | 2048 | ~2.2% | Medium | Improved accuracy |
    | 4096 | ~1.6% | Slow | High precision |
    | 8192 | ~1.1% | Very Slow | Research quality |

!!! info "Statistical Error"
    Error scales as 1/√N where N is the number of shots. Doubling shots reduces error by ~30%.

!!! warning "Validation"
    Shots must be a positive integer. Values ≤ 0 will raise `ArgumentTypeError`.

### `--optimization-level`

**Type**: `choice`
**Default**: `3`
**Choices**: `0`, `1`, `2`, `3`

Qiskit circuit optimization level for transpilation.

=== "Optimization Levels"

    | Level | Description | Circuit Depth | Transpile Time | Use Case |
    |-------|-------------|---------------|----------------|----------|
    | `0` | No optimization | Highest | Fastest | Debugging only |
    | `1` | Light optimization | High | Fast | Quick tests |
    | `2` | Medium optimization | Medium | Medium | Balanced |
    | `3` | Heavy optimization | **Lowest** | **Slowest** | **Production** |

=== "Usage"

    ```bash
    # No optimization (debugging)
    python quantum_pipeline.py -f molecules.json --optimization-level 0

    # Heavy optimization (production)
    python quantum_pipeline.py -f molecules.json --optimization-level 3
    ```

!!! tip "Production Recommendation"
    Always use level `3` for production runs. The transpilation overhead is negligible compared to circuit execution time, and the gate count reduction significantly improves fidelity.

---

## Additional Features

### `--report`

**Type**: `flag` (action: `store_true`)
**Default**: `false`

Generate a PDF report after simulation completion.

```bash
python quantum_pipeline.py -f molecules.json --report
```

!!! info "Report Contents"
    - Molecular structure visualization
    - Hamiltonian operator coefficients
    - Energy convergence plots
    - VQE parameter evolution
    - Final results and statistics
    - Comparison with reference values (if available)

### `--dump`

**Type**: `flag` (action: `store_true`)
**Default**: `false`

Save the current configuration to a JSON file in `run_configs/` directory.

=== "Usage"

    ```bash
    # Generate configuration file
    python quantum_pipeline.py \
        -f molecules.json \
        --basis cc-pvdz \
        --max-iterations 200 \
        --optimizer L-BFGS-B \
        --dump
    ```

=== "Output Format"

    ```json
    {
        "file": "molecules.json",
        "basis": "cc-pvdz",
        "max_iterations": 200,
        "optimizer": "L-BFGS-B",
        "ansatz_reps": 2,
        "shots": 1024,
        "backend": {
            "local": true,
            "optimization_level": 3,
            "method": "tensor_network"
        }
    }
    ```

!!! tip "Reproducibility"
    Use `--dump` to save configurations for reproducible research. The generated JSON can be loaded later with `--load`.

!!! danger "Mutual Exclusivity"
    **Cannot be used with `--load`**. Choose one or the other.

### `--load`

**Type**: `string` (path)
**Default**: None

Load configuration from a previously saved JSON file.

```bash
# Load saved configuration
python quantum_pipeline.py --load run_configs/config_20250101.json
```

!!! warning "Validation"
    - File must exist and be readable
    - Must contain valid JSON matching configuration schema
    - Will raise `ArgumentTypeError` if file is invalid

!!! danger "Mutual Exclusivity"
    **Cannot be used with `--dump`**. Choose one or the other.

### `--gpu`

**Type**: `flag` (action: `store_true`)
**Default**: `false`

Enable GPU acceleration for quantum circuit simulation.

=== "Usage"

    ```bash
    # Enable GPU acceleration
    python quantum_pipeline.py -f molecules.json --gpu

    # GPU with optimized method
    python quantum_pipeline.py -f molecules.json \
        --gpu \
        --simulation-method statevector \
        --optimizer L-BFGS-B
    ```

=== "GPU Requirements"

    - CUDA-capable NVIDIA GPU
    - CUDA Toolkit (11.2+ recommended)
    - cuQuantum libraries (for Volta/Ampere GPUs)
    - qiskit-aer compiled with GPU support
    - Sufficient GPU memory (6GB minimum)

!!! tip "GPU Performance"
    - Best with `statevector` or `tensor_network` methods
    - `L-BFGS-B` optimizer performs exceptionally well on GPU
    - Can provide 10-100x speedup for large circuits
    - Monitor GPU memory with `nvidia-smi`

!!! warning "GPU Configuration"
    GPU settings in `defaults.py`:

    ```python
    'gpu_opts': {
        'device': 'GPU',
        'cuStateVec_enable': False,  # Set true for Volta/Ampere
        'blocking_enable': False,
        'batched_shots_gpu': True,
        'shot_branching_enable': True,
        'max_memory_mb': 5500,  # Adjust for your GPU
    }
    ```

### `--simulation-method`

**Type**: `choice`
**Default**: `tensor_network`

Backend simulation method for quantum circuit execution.

=== "Available Methods"

    | Method | Description | GPU | Use Case |
    |--------|-------------|-----|----------|
    | `automatic` | Auto-select based on circuit | Partial | Let Qiskit decide |
    | `statevector` | Dense statevector simulation | Yes | **Ideal circuits** |
    | `density_matrix` | Dense density matrix | No | Noisy simulations |
    | `stabilizer` | Clifford stabilizer states | No | Clifford circuits |
    | `extended_stabilizer` | Clifford + T approximate | No | Near-Clifford circuits |
    | `matrix_product_state` | Tensor network MPS | Partial | Low-entanglement |
    | `unitary` | Full unitary matrix | No | Small circuits only |
    | `superop` | Superoperator matrix | No | Noise channels |
    | `tensor_network` | cuTensorNet simulation | Yes | **GPU-accelerated** |

=== "Usage Examples"

    ```bash
    # GPU-accelerated (recommended for large circuits)
    python quantum_pipeline.py -f molecules.json \
        --simulation-method tensor_network --gpu

    # Ideal statevector simulation
    python quantum_pipeline.py -f molecules.json \
        --simulation-method statevector

    # Noisy simulation
    python quantum_pipeline.py -f molecules.json \
        --simulation-method density_matrix --noise ibmq_manila
    ```

=== "Performance Comparison"

    | Method | Speed (CPU) | Speed (GPU) | Memory | Accuracy |
    |--------|-------------|-------------|--------|----------|
    | `statevector` | Medium | Very High | High | Exact |
    | `tensor_network` | Low | Very High | Medium | Exact |
    | `density_matrix` | Very Low | Not supported | Very High | Exact |
    | `stabilizer` | Very High | Not supported | Low | Exact (Clifford) |
    | `mps` | High | Medium | Low | Approximate |

!!! tip "Method Selection"
    - **GPU available**: Use `tensor_network` or `statevector`
    - **CPU only**: Use `automatic` or `statevector`
    - **Low memory**: Use `matrix_product_state` or `stabilizer`
    - **Noisy circuits**: Use `density_matrix`

For detailed information, see [Simulation Methods Guide](simulation-methods.md).

### `--noise`

**Type**: `string`
**Default**: `None`

Specify a noise model to simulate realistic quantum hardware errors.

```bash
# Use IBM device noise model
python quantum_pipeline.py -f molecules.json --noise ibmq_manila

# Use generic noise model
python quantum_pipeline.py -f molecules.json --noise fake_backend
```

!!! info "Noise Models"
    - Can reference real IBM Quantum devices (e.g., `ibmq_manila`, `ibm_kyoto`)
    - Can use FakeBackend providers for testing
    - Automatically fetches noise parameters from IBM Quantum
    - Best used with `density_matrix` simulation method

---

## Performance Monitoring

### `--enable-performance-monitoring`

**Type**: `flag` (action: `store_true`)
**Default**: `false`

Enable performance and resource monitoring for VQE simulations.

```bash
python quantum_pipeline.py -f molecules.json \
    --enable-performance-monitoring \
    --performance-interval 30 \
    --performance-export-format both
```

!!! info "Monitored Metrics"
    - **System**: CPU%, GPU%, memory, disk I/O
    - **VQE**: Energy, iterations, convergence rate
    - **Scientific**: Comparison with reference database
    - **Efficiency**: Iterations/sec, overhead ratio

For complete monitoring documentation, see [Monitoring Guide](../monitoring/performance-metrics.md).

### `--performance-interval`

**Type**: `int`
**Default**: `30`

Performance metrics collection interval in seconds.

```bash
# High-frequency monitoring (10s intervals)
python quantum_pipeline.py -f molecules.json \
    --enable-performance-monitoring \
    --performance-interval 10

# Low-overhead monitoring (60s intervals)
python quantum_pipeline.py -f molecules.json \
    --enable-performance-monitoring \
    --performance-interval 60
```

!!! warning "Performance Impact"
    Lower intervals (< 10s) may introduce overhead. For production, use 30-60 second intervals.

### `--performance-pushgateway`

**Type**: `string` (URL)
**Default**: None

Prometheus PushGateway URL for metrics export.

```bash
python quantum_pipeline.py -f molecules.json \
    --enable-performance-monitoring \
    --performance-pushgateway http://localhost:9091
```

!!! tip "Integration"
    - Requires Prometheus PushGateway running
    - Default PushGateway port: 9091
    - Metrics available for Grafana dashboards
    - See [`docker-compose.thesis.yaml`](https://github.com/straightchlorine/quantum-pipeline/blob/master/docker-compose.thesis.yaml) for monitoring stack setup

### `--performance-export-format`

**Type**: `choice`
**Default**: `both`
**Choices**: `json`, `prometheus`, `both`

Format for exporting performance metrics.

=== "Export Formats"

    | Format | Description | Output |
    |--------|-------------|--------|
    | `json` | JSON file export | `gen/performance_metrics/*.json` |
    | `prometheus` | Prometheus PushGateway | Real-time metrics |
    | `both` | Both formats | **Recommended** |

=== "Usage"

    ```bash
    # JSON only (offline analysis)
    python quantum_pipeline.py -f molecules.json \
        --enable-performance-monitoring \
        --performance-export-format json

    # Prometheus only (real-time dashboards)
    python quantum_pipeline.py -f molecules.json \
        --enable-performance-monitoring \
        --performance-export-format prometheus \
        --performance-pushgateway http://localhost:9091
    ```

---

## Kafka Configuration

### `--kafka`

**Type**: `flag` (action: `store_true`)
**Default**: `false`

Enable streaming of VQE results to Apache Kafka for real-time data processing.

```bash
python quantum_pipeline.py -f molecules.json --kafka
```

!!! info "Kafka Integration"
    - Results serialized using **Avro** format
    - Automatic Schema Registry integration
    - Binary encoding for efficient transmission
    - Supports both CPU and GPU results
    - Compatible with Apache Spark consumers

!!! tip "Data Pipeline"
    Use with Airflow DAG for automated feature engineering:

    ```bash
    python quantum_pipeline.py -f molecules.json \
        --kafka \
        --servers kafka:9092 \
        --topic vqe_results
    ```

### `--servers`

**Type**: `string`
**Default**: `localhost:9092`

Kafka bootstrap servers (comma-separated for multiple brokers).

=== "Usage"

    ```bash
    # Single broker
    python quantum_pipeline.py -f molecules.json \
        --kafka --servers localhost:9092

    # Multiple brokers (HA setup)
    python quantum_pipeline.py -f molecules.json \
        --kafka --servers kafka1:9092,kafka2:9092,kafka3:9092

    # Docker Compose
    python quantum_pipeline.py -f molecules.json \
        --kafka --servers kafka:9092
    ```

=== "Connection Formats"

    | Format | Example | Use Case |
    |--------|---------|----------|
    | `host:port` | `localhost:9092` | Local development |
    | `service:port` | `kafka:9092` | Docker Compose |
    | `ip:port` | `192.168.1.10:9092` | Remote broker |
    | `host1:port,host2:port` | `kafka1:9092,kafka2:9092` | HA cluster |

### `--topic`

**Type**: `string`
**Default**: `vqe_decorated_result`

Kafka topic name for message categorization.

```bash
python quantum_pipeline.py -f molecules.json \
    --kafka \
    --topic vqe_experiment_results
```

!!! tip "Topic Naming"
    Use descriptive topic names that reflect data content:

    - `vqe_production_results` - Production runs
    - `vqe_test_results` - Testing/development
    - `vqe_h2_basis_sweep` - Specific experiments

### `--retries`

**Type**: `int`
**Default**: `3`

Number of attempts to send messages to Kafka before giving up.

```bash
# High reliability (more retries)
python quantum_pipeline.py -f molecules.json \
    --kafka --retries 5 --retry-delay 3
```

### `--retry-delay`

**Type**: `int`
**Default**: `2`

Delay in seconds between Kafka send retry attempts.

```bash
python quantum_pipeline.py -f molecules.json \
    --kafka --retry-delay 5
```

### `--internal-retries`

**Type**: `int`
**Default**: `0`

Number of automatic retries Kafka producer should attempt internally.

!!! warning "Duplicate Risk"
    Setting `internal-retries > 0` introduces risk of duplicate message delivery. Use with caution in production.

```bash
# Not recommended for production
python quantum_pipeline.py -f molecules.json \
    --kafka --internal-retries 2
```

### `--acks`

**Type**: `choice`
**Default**: `all`
**Choices**: `0`, `1`, `all`

Number of acknowledgments producer requires before considering a request complete.

=== "Acknowledgment Levels"

    | Level | Meaning | Durability | Performance | Use Case |
    |-------|---------|------------|-------------|----------|
    | `0` | No acks | None | Fastest | Testing only |
    | `1` | Leader ack | Low | Fast | Non-critical data |
    | `all` | All replicas | **Highest** | Slower | **Production** |

=== "Usage"

    ```bash
    # Maximum durability (production)
    python quantum_pipeline.py -f molecules.json \
        --kafka --acks all

    # Fast but risky (testing)
    python quantum_pipeline.py -f molecules.json \
        --kafka --acks 0
    ```

!!! danger "Data Loss Risk"
    Never use `acks=0` or `acks=1` for production VQE results. Always use `acks=all` to ensure data durability.

### `--timeout`

**Type**: `int`
**Default**: `10`

Number of seconds before producer considers a request as failed.

```bash
# Longer timeout for slow networks
python quantum_pipeline.py -f molecules.json \
    --kafka --timeout 30
```

---

## Security Settings

Security options for Apache Kafka SSL/TLS and SASL authentication.

### SSL/TLS Configuration

#### `--ssl`

**Type**: `flag` (action: `store_true`)
**Default**: `false`

Enable SSL/TLS encryption for Kafka connections.

```bash
python quantum_pipeline.py -f molecules.json \
    --kafka --ssl \
    --ssl-dir ./secrets/
```

!!! warning "Required with SSL"
    When `--ssl` is enabled, you must provide either:

    - `--ssl-dir` with certificate files, OR
    - `--ssl-cafile`, `--ssl-certfile`, and `--ssl-keyfile`

#### `--disable-ssl-check-hostname`

**Type**: `flag` (action: `store_false`)
**Default**: `true` (hostname checking enabled)

Disable SSL hostname verification. **ONLY USE FOR TESTING.**

```bash
# Testing only - NOT for production
python quantum_pipeline.py -f molecules.json \
    --kafka --ssl \
    --disable-ssl-check-hostname
```

!!! danger "Security Risk"
    Disabling hostname checking makes connections vulnerable to man-in-the-middle attacks. **Never use in production.**

#### `--ssl-password`

**Type**: `string`
**Default**: `None`

Password for encrypted SSL private key.

```bash
python quantum_pipeline.py -f molecules.json \
    --kafka --ssl \
    --ssl-dir ./secrets/ \
    --ssl-password "your_key_password"
```

#### `--ssl-dir`

**Type**: `string` (path)
**Default**: `./secrets/`

Directory containing SSL certificate files.

=== "Usage"

    ```bash
    python quantum_pipeline.py -f molecules.json \
        --kafka --ssl \
        --ssl-dir /path/to/certs/
    ```

=== "Expected Files"

    When using `--ssl-dir`, the directory must contain:

    ```
    secrets/
    ├── ca.crt          # CA certificate
    ├── client.crt      # Client certificate
    ├── client.key      # Private key
    └── client.crl      # CRL (optional)
    ```

!!! danger "Mutual Exclusivity"
    Cannot use `--ssl-dir` with individual file options (`--ssl-cafile`, etc.). Choose one approach.

#### `--ssl-cafile`

**Type**: `string` (path)
**Default**: `None`

Path to CA (Certificate Authority) certificate file.

```bash
python quantum_pipeline.py -f molecules.json \
    --kafka --ssl \
    --ssl-cafile /path/to/ca.crt \
    --ssl-certfile /path/to/client.crt \
    --ssl-keyfile /path/to/client.key
```

#### `--ssl-certfile`

**Type**: `string` (path)
**Default**: `None`

Path to client SSL certificate file.

#### `--ssl-keyfile`

**Type**: `string` (path)
**Default**: `None`

Path to client SSL private key file.

!!! info "Individual File Options"
    When not using `--ssl-dir`, you must provide:

    - `--ssl-cafile` (required)
    - `--ssl-certfile` (required)
    - `--ssl-keyfile` (required)
    - `--ssl-crlfile` (optional)

#### `--ssl-crlfile`

**Type**: `string` (path)
**Default**: `None`

Path to SSL certificate revocation list (CRL) file.

```bash
python quantum_pipeline.py -f molecules.json \
    --kafka --ssl \
    --ssl-cafile ca.crt \
    --ssl-certfile client.crt \
    --ssl-keyfile client.key \
    --ssl-crlfile revoked.crl
```

#### `--ssl-ciphers`

**Type**: `string`
**Default**: `None`

Specify SSL cipher suite to use for encryption.

```bash
python quantum_pipeline.py -f molecules.json \
    --kafka --ssl \
    --ssl-dir ./secrets/ \
    --ssl-ciphers "ECDHE-RSA-AES256-GCM-SHA384"
```

### SASL Authentication

#### `--sasl-ssl`

**Type**: `flag` (action: `store_true`)
**Default**: `false`

Enable SASL_SSL authentication for Kafka connections.

```bash
python quantum_pipeline.py -f molecules.json \
    --kafka --sasl-ssl \
    --sasl-mechanism PLAIN \
    --sasl-plain-username user \
    --sasl-plain-password pass
```

!!! warning "Required with SASL"
    When `--sasl-ssl` is enabled, you must also specify `--sasl-mechanism`.

#### `--sasl-mechanism`

**Type**: `choice`
**Choices**: `PLAIN`, `GSSAPI`, `SCRAM-SHA-256`, `SCRAM-SHA-512`

SASL authentication mechanism.

=== "Mechanisms"

    | Mechanism | Description | Required Parameters |
    |-----------|-------------|---------------------|
    | `PLAIN` | Simple username/password | `--sasl-plain-username`, `--sasl-plain-password` |
    | `SCRAM-SHA-256` | Salted challenge-response (SHA-256) | `--sasl-plain-username`, `--sasl-plain-password` |
    | `SCRAM-SHA-512` | Salted challenge-response (SHA-512) | `--sasl-plain-username`, `--sasl-plain-password` |
    | `GSSAPI` | Kerberos authentication | `--sasl-kerberos-service-name`, `--sasl-kerberos-domain-name` |

=== "Usage Examples"

    ```bash
    # PLAIN mechanism
    python quantum_pipeline.py -f molecules.json \
        --kafka --sasl-ssl \
        --sasl-mechanism PLAIN \
        --sasl-plain-username admin \
        --sasl-plain-password secret

    # SCRAM-SHA-256 (recommended)
    python quantum_pipeline.py -f molecules.json \
        --kafka --sasl-ssl \
        --sasl-mechanism SCRAM-SHA-256 \
        --sasl-plain-username admin \
        --sasl-plain-password secret

    # Kerberos (GSSAPI)
    python quantum_pipeline.py -f molecules.json \
        --kafka --sasl-ssl \
        --sasl-mechanism GSSAPI \
        --sasl-kerberos-service-name kafka \
        --sasl-kerberos-domain-name example.com
    ```

#### `--sasl-plain-username`

**Type**: `string`
**Default**: `None`

Username for SASL PLAIN and SCRAM authentication.

#### `--sasl-plain-password`

**Type**: `string`
**Default**: `None`

Password for SASL PLAIN and SCRAM authentication.

#### `--sasl-kerberos-service-name`

**Type**: `string`
**Default**: `kafka`

Kerberos service name for GSSAPI SASL mechanism.

```bash
python quantum_pipeline.py -f molecules.json \
    --kafka --sasl-ssl \
    --sasl-mechanism GSSAPI \
    --sasl-kerberos-service-name kafka
```

#### `--sasl-kerberos-domain-name`

**Type**: `string`
**Default**: `None`

Kerberos domain name for GSSAPI SASL mechanism.

```bash
python quantum_pipeline.py -f molecules.json \
    --kafka --sasl-ssl \
    --sasl-mechanism GSSAPI \
    --sasl-kerberos-service-name kafka \
    --sasl-kerberos-domain-name EXAMPLE.COM
```

---

## Configuration Files

### Dumping Configuration

Save current parameters to a JSON file for reproducibility:

```bash
python quantum_pipeline.py \
    -f molecules.json \
    --basis cc-pvdz \
    --max-iterations 200 \
    --optimizer L-BFGS-B \
    --ansatz-reps 2 \
    --dump
```

**Output** (in `run_configs/config_YYYYMMDD_HHMMSS.json`):

```json
{
    "file": "molecules.json",
    "basis": "cc-pvdz",
    "max_iterations": 200,
    "convergence_threshold_enable": false,
    "convergence_threshold": 1e-6,
    "optimizer": "L-BFGS-B",
    "ansatz_reps": 2,
    "shots": 1024,
    "backend": {
        "local": true,
        "min_qubits": null,
        "optimization_level": 3,
        "method": "tensor_network",
        "gpu": false,
        "noise_backend": null
    },
    "kafka": {
        "enabled": false,
        "servers": "localhost:9092",
        "topic": "vqe_decorated_result"
    }
}
```

### Loading Configuration

Load previously saved configuration:

```bash
python quantum_pipeline.py --load run_configs/config_20250122.json
```

!!! tip "Override Loaded Config"
    You can override specific parameters when loading:

    ```bash
    # Load config but change optimizer
    python quantum_pipeline.py \
        --load config.json \
        --optimizer COBYLA
    ```

---

## Environment Variables

### Performance Monitoring Variables

```bash
export QUANTUM_PERFORMANCE_ENABLED=true
export QUANTUM_PERFORMANCE_PUSHGATEWAY_URL=http://localhost:9091
export QUANTUM_PERFORMANCE_COLLECTION_INTERVAL=10
export QUANTUM_PERFORMANCE_EXPORT_FORMAT=json,prometheus
```

### IBM Quantum

```bash
# IBM Quantum Access
export IBM_RUNTIME_TOKEN=your_token_here
export IBM_RUNTIME_INSTANCE=crn:v1:bluemix:public:quantum-computing:...
```

!!! note "Docker Variables"
    In Docker deployments, additional variables like `KAFKA_SERVERS`, `CONTAINER_TYPE`, and `MAX_ITERATIONS` are set by the container entrypoint. These are not part of the general configuration hierarchy.

---

## Validation Rules

The argument parser enforces these validation rules:

### Mutual Exclusivity

| Rule | Error Message |
|------|---------------|
| `--dump` and `--load` | "cannot be used together" |
| `--max-iterations` and `--convergence` | "are mutually exclusive" |
| `--ssl-dir` and individual SSL files | "cannot specify both" |

### Conditional Requirements

| Condition | Requirement | Error Message |
|-----------|-------------|---------------|
| `--ibm` not set | Cannot use `--min-qubits` | "can only be used if --ibm is selected" |
| `--convergence` set | Must set `--threshold` | "must be set if --convergence is enabled" |
| Kafka params changed | Must set `--kafka` | "must be set for the options to take effect" |
| `--sasl-mechanism` set | Must set `--kafka` | "must be enabled when using SASL" |
| `--ssl` set | Must set `--kafka` | "must be enabled when using SSL" |

### SSL Validation

When `--ssl` is enabled:

- **Option 1**: Provide `--ssl-dir` (exclusive)
- **Option 2**: Provide `--ssl-cafile`, `--ssl-certfile`, `--ssl-keyfile` (all required)

### SASL Validation

When `--sasl-ssl` is enabled:

- Must specify `--sasl-mechanism`
- For `PLAIN`/`SCRAM-*`: Requires `--sasl-plain-username` and `--sasl-plain-password`
- For `GSSAPI`: Cannot use PLAIN/SCRAM credentials

---

## Best Practices

### Configuration Management

#### Recommended

- **Use `--dump` for reproducibility**: Save configurations for all production runs
- **Load configs in CI/CD**: Use `--load` in automated pipelines
- **Enable monitoring**: Always use `--enable-performance-monitoring` in production
- **Use convergence mode**: Prefer `--convergence` over fixed `--max-iterations`

#### Discouraged

- **Skip validation**: Always validate molecule files before running
- **Ignore warnings**: Address optimizer warnings about parameters
- **Use `acks=0`**: Never in production (risk of data loss)
- **Disable SSL checks**: Never use `--disable-ssl-check-hostname` in production

### Performance Optimization

#### Quick Iterations

```bash
python quantum_pipeline.py \
    -f molecules.json \
    --basis sto3g \
    --max-iterations 50 \
    --optimizer COBYLA \
    --shots 512
```

#### Production Quality

```bash
python quantum_pipeline.py \
    -f molecules.json \
    --basis sto3g \
    --convergence \
    --threshold 1e-6 \
    --optimizer L-BFGS-B \
    --shots 2048 \
    --report \
    --kafka
```

#### High-Accuracy Research

```bash
python quantum_pipeline.py \
    -f molecules.json \
    --basis cc-pvdz \
    --convergence \
    --threshold 1e-8 \
    --optimizer L-BFGS-B \
    --ansatz-reps 5 \
    --shots 4096 \
    --optimization-level 3 \
    --report \
    --enable-performance-monitoring
```

#### GPU-Accelerated

```bash
python quantum_pipeline.py \
    -f molecules.json \
    --gpu \
    --simulation-method tensor_network \
    --optimizer L-BFGS-B \
    --max-iterations 200 \
    --shots 2048
```

### Security Hardening

#### Kafka with SSL

```bash
python quantum_pipeline.py \
    -f molecules.json \
    --kafka \
    --ssl \
    --ssl-dir ./secrets/ \
    --acks all \
    --retries 5
```

#### Kafka with SASL

```bash
export KAFKA_PASSWORD="$SECRET_PASSWORD"

python quantum_pipeline.py \
    -f molecules.json \
    --kafka \
    --sasl-ssl \
    --sasl-mechanism SCRAM-SHA-256 \
    --sasl-plain-username admin \
    --sasl-plain-password "$KAFKA_PASSWORD" \
    --acks all
```

---

## Common Combinations

### Development

```bash
# Fast iteration for debugging
python quantum_pipeline.py \
    -f test_molecules.json \
    --basis sto3g \
    --max-iterations 20 \
    --optimizer COBYLA \
    --log-level DEBUG
```

### Testing

```bash
# Validate configuration before production
python quantum_pipeline.py \
    -f molecules.json \
    --basis sto3g \
    --max-iterations 100 \
    --optimizer L-BFGS-B \
    --report \
    --dump
```

### Production

```bash
# Full production run with data pipeline
python quantum_pipeline.py \
    -f molecules.json \
    --basis sto3g \
    --convergence \
    --threshold 1e-6 \
    --optimizer L-BFGS-B \
    --ansatz-reps 2 \
    --shots 2048 \
    --optimization-level 3 \
    --report \
    --kafka \
    --servers kafka:9092 \
    --acks all \
    --enable-performance-monitoring \
    --performance-pushgateway http://localhost:9091 \
    --performance-export-format both \
    --log-level INFO
```

### Research

```bash
# High-accuracy publication-quality run
python quantum_pipeline.py \
    -f molecules.json \
    --basis cc-pvdz \
    --convergence \
    --threshold 1e-8 \
    --optimizer L-BFGS-B \
    --ansatz-reps 5 \
    --shots 8192 \
    --optimization-level 3 \
    --report \
    --enable-performance-monitoring \
    --dump
```

### GPU Cluster

```bash
# Large-scale GPU-accelerated batch processing
python quantum_pipeline.py \
    -f large_molecules.json \
    --gpu \
    --simulation-method statevector \
    --basis 6-31g \
    --convergence \
    --threshold 1e-7 \
    --optimizer L-BFGS-B \
    --shots 4096 \
    --kafka \
    --enable-performance-monitoring
```

---

## Troubleshooting

### Configuration Issues

#### Problem: "Cannot use both --max-iterations and --convergence"

**Solution**: Choose one optimization stopping criterion:

```bash
# Use either this
python quantum_pipeline.py -f molecules.json --max-iterations 100

# OR this
python quantum_pipeline.py -f molecules.json --convergence --threshold 1e-6
```

#### Problem: "File not found" error

**Solution**: Ensure file path is correct and file exists:

```bash
# Check file exists
ls -l data/molecules.json

# Use absolute path
realpath data/molecules.json
python quantum_pipeline.py --file /absolute/path/to/molecules.json
```

#### Problem: Kafka parameters ignored

**Solution**: Enable `--kafka` flag:

```bash
# Wrong - kafka params ignored
python quantum_pipeline.py -f molecules.json --servers kafka:9092

# Correct
python quantum_pipeline.py -f molecules.json --kafka --servers kafka:9092
```

### Performance Issues

#### Problem: Simulation runs too slow

**Solutions**:

1. **Reduce basis set**: Use `sto3g` instead of `cc-pvdz`
2. **Enable GPU**: Add `--gpu --simulation-method statevector`
3. **Lower shots**: Reduce from 2048 to 1024
4. **Reduce ansatz reps**: Use 2 instead of 5

#### Problem: Out of memory

**Solutions**:

1. **Use tensor network**: `--simulation-method tensor_network`
2. **Enable GPU**: `--gpu` offloads to GPU memory
3. **Reduce shots**: Lower `--shots` value
4. **Use smaller basis**: Switch to `sto3g`

### Security Issues

#### Problem: SSL connection fails

**Solution**: Verify certificate paths and permissions:

```bash
# Check certificates exist
ls -l secrets/ca.crt secrets/client.crt secrets/client.key

# Check file permissions
chmod 600 secrets/client.key
```

#### Problem: SASL authentication fails

**Solution**: Verify credentials and mechanism:

```bash
# Test connection
kafka-console-consumer \
    --bootstrap-server kafka:9092 \
    --topic test \
    --consumer-property sasl.mechanism=SCRAM-SHA-256 \
    --consumer-property sasl.jaas.config='...'
```

---

## Next Steps

- **Learn about optimizers**: [Optimizer Guide](optimizers.md)
- **Understand simulation methods**: [Simulation Methods](simulation-methods.md)
- **See real examples**: [Usage Examples](examples.md)
- **Deploy to production**: [Deployment Guide](../deployment/docker-basics.md)
- **Monitor performance**: [Monitoring Guide](../monitoring/performance-metrics.md)
- **Integrate with data platform**: [Data Platform](../data-platform/kafka-streaming.md)
