# Usage Overview

This section covers all aspects of configuring and running VQE simulations with the Quantum Pipeline.

---

## Quick Navigation

<div class="grid cards" markdown>

-   **Configuration**

    ---

    Complete reference for all CLI parameters and settings

    [Configuration Guide →](configuration.md)

-   **Optimizers**

    ---

    16 optimization algorithms for VQE convergence

    [Optimizer Guide →](optimizers.md)

-   **Simulation Methods**

    ---

    Backend configuration and GPU acceleration options

    [Simulation Methods →](simulation-methods.md)

-   **Examples**

    ---

    Real-world usage examples and code recipes

    [Examples →](examples.md)

</div>

---

## Configuration Hierarchy

Quantum Pipeline uses a layered configuration system:

```mermaid
graph LR
    A[CLI Arguments] -->|Override| B[Config File --load]
    B -->|Fallback| C[Defaults]

    style A fill:#1a73e8,color:#fff,stroke:#1557b0
    style B fill:#e8710a,color:#fff,stroke:#c45e08
    style C fill:#34a853,color:#fff,stroke:#2d8f47
```

### Priority Order (Highest to Lowest)

1. **Command-Line Arguments** — Direct flags passed to `quantum_pipeline.py`
2. **Configuration File** — Loaded via `--load config.json`
3. **Defaults** — `quantum_pipeline/configs/defaults.py`

!!! note "Environment Variables"
    Some environment variables are supported for performance monitoring (`QUANTUM_PERFORMANCE_*`) and Docker deployments (`CONTAINER_TYPE`), but they do not participate in the general configuration hierarchy.

---

## Parameter Categories

### Core Parameters

Essential parameters for running VQE simulations:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--file` | path | required | Molecule data file (JSON) |
| `--basis` | string | `sto3g` | Basis set for simulation |
| `--max-iterations` | int | `100` | Maximum VQE iterations |
| `--optimizer` | string | `L-BFGS-B` | Optimization algorithm |
| `--ansatz-reps` | int | `2` | Ansatz circuit repetitions |

[Full parameter list →](configuration.md)

### Simulation Configuration

Backend and execution settings:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--simulation-method` | string | `tensor_network` | Backend simulation method |
| `--shots` | int | `1024` | Number of circuit shots |
| `--optimization-level` | int | `3` | Circuit optimization (0-3) |
| `--gpu` | flag | `false` | Enable GPU acceleration |

[Backend configuration →](simulation-methods.md)

### Data & Output

Result handling and storage:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--kafka` | flag | `false` | Stream to Apache Kafka |
| `--report` | flag | `false` | Generate PDF report |
| `--output-dir` | path | `./gen` | Output directory |
| `--log-level` | string | `INFO` | Logging verbosity |

---

## Common Workflows

### 1. Quick Prototyping

Fast iteration for testing:

```bash
python quantum_pipeline.py \
    --file molecules.json \
    --basis sto3g \
    --max-iterations 50 \
    --optimizer COBYLA
```

**Best for:**

- Testing molecule formats
- Rapid experimentation
- Debugging workflows

### 2. Production Simulations

Balanced performance and accuracy:

```bash
python quantum_pipeline.py \
    --file molecules.json \
    --basis sto3g \
    --max-iterations 150 \
    --optimizer L-BFGS-B \
    --ansatz-reps 2 \
    --report
```

**Best for:**

- Standard research workflows
- Reproducible results
- Moderate complexity molecules

### 3. High-Accuracy Research

Publication-quality results:

```bash
python quantum_pipeline.py \
    --file molecules.json \
    --basis cc-pvdz \
    --convergence \
    --threshold 1e-8 \
    --optimizer L-BFGS-B \
    --ansatz-reps 5 \
    --shots 4096 \
    --report
```

**Best for:**

- Academic publications
- Benchmarking studies
- Chemical accuracy requirements

!!! warning "Accuracy for publication purposes is still unstable"
    Mostly due to still randomly initiated ansatz - might cause algorithm to get stuck in local minima.
    This can cause long runs - without any fruitful results.

### 4. GPU-Accelerated Runs

Maximum performance:

```bash
python quantum_pipeline.py \
    --file molecules.json \
    --gpu \
    --simulation-method statevector \
    --max-iterations 200 \
    --optimizer L-BFGS-B
```

**Best for:**

- Large molecules
- Complex basis sets
- Time-critical simulations

### 5. Data Engineering Pipeline

Full platform integration:

```bash
python quantum_pipeline.py \
    --file molecules.json \
    --kafka \
    --enable-performance-monitoring \
    --max-iterations 150 \
    --report
```

**Best for:**

- Batch processing
- ML feature extraction
- Production deployments

---

## Choosing Parameters

### Basis Set Selection

Choose based on accuracy vs performance tradeoff:

| Basis Set | Accuracy | Speed | Memory | Use Case |
|-----------|----------|-------|--------|----------|
| `sto3g` | Low | Very High | Low | Quick tests, prototyping |
| `6-31g` | Medium | High | Medium | Standard simulations |
| `cc-pvdz` | Very High | Low | High | High-accuracy research |

[Learn more about basis sets →](../scientific/basis-sets.md)

### Optimizer Selection

Choose based on problem characteristics:

| Optimizer | Type | Speed | Accuracy | Best For |
|-----------|------|-------|----------|----------|
| `L-BFGS-B` | Gradient | High | High | **Recommended default** |
| `COBYLA` | Derivative-free | Medium | Medium | Noisy cost functions |
| `SLSQP` | Sequential | High | Medium | Constrained optimization |
| `Powell` | Derivative-free | Low | Medium | Smooth landscapes |

[Full optimizer comparison →](optimizers.md)

### Iteration Count

How many iterations do you need?

| Molecule Size | Basis Set | Recommended Iterations |
|---------------|-----------|------------------------|
| Small (H₂, HeH⁺) | sto3g | 50-100 |
| Medium (H₂O, NH₃) | sto3g | 100-200 |
| Large (CO₂, benzene) | sto3g | 200-500 |
| Any | cc-pvdz | 300-1000 |


---

## Environment Variables

Override settings without modifying code:

### Performance Monitoring Variables

```bash
# Performance monitoring
export QUANTUM_PERFORMANCE_ENABLED=true
export QUANTUM_PERFORMANCE_PUSHGATEWAY_URL=http://localhost:9091
export QUANTUM_PERFORMANCE_COLLECTION_INTERVAL=10
export QUANTUM_PERFORMANCE_EXPORT_FORMAT=json,prometheus
```

!!! note "Docker Environment Variables"
    In Docker deployments, additional variables like `KAFKA_SERVERS`, `CONTAINER_TYPE`, and `MAX_ITERATIONS` are used by the container entrypoint. See [Docker Compose Guide](../deployment/docker-compose.md).

---

## Configuration File Reference

### Default Settings Location

```
quantum_pipeline/configs/defaults.py
```

### Defaults Reference

All default values are defined in `quantum_pipeline/configs/defaults.py`. To change defaults permanently, edit that file directly. Supported optimizers, basis sets, and simulation methods are listed in `quantum_pipeline/configs/settings.py`.

---

## Best Practices

### Recommended

- Use `--convergence` for production runs
- Enable `--report` for reproducibility
- Start with `sto3g`, then upgrade basis
- Monitor with `--enable-performance-monitoring`
- Stream to Kafka for batch processing

### Discouraged

- Use both `--max-iterations` and `--convergence`
- Run GPU simulations without proper cooling
- Ignore optimizer recommendations
- Skip basis set validation
- Forget to enable logging in production

---

## Next Steps

Choose your path:

- **New to VQE?** Start with [Optimizer Guide](optimizers.md)
- **Performance tuning?** See [Simulation Methods](simulation-methods.md)
- **Real examples?** Check [Usage Examples](examples.md)
- **All parameters?** Read [Configuration Reference](configuration.md)
