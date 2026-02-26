# Basic Usage

This guide covers fundamental usage patterns and workflows for the Quantum Pipeline framework.

---

## Command Line Interface

The primary way to interact with Quantum Pipeline is through the command-line interface.

### Basic Syntax

```bash
python quantum_pipeline.py [OPTIONS]
```

### Required Options

Only one option is strictly required:

```bash
python quantum_pipeline.py --file <path-to-molecules.json>
```

All other parameters have sensible defaults defined in [`quantum_pipeline/configs/defaults.py`](https://github.com/straightchlorine/quantum-pipeline/blob/master/quantum_pipeline/configs/defaults.py).

### Common Workflows

=== "Quick Test Run"
    ```bash
    # Fast test with minimal iterations
    python quantum_pipeline.py \
        --file data/molecules.json \
        --max-iterations 10
    ```

=== "Production Run"
    ```bash
    # Full simulation with reporting
    python quantum_pipeline.py \
        --file data/molecules.json \
        --basis cc-pvdz \
        --max-iterations 200 \
        --optimizer L-BFGS-B \
        --report
    ```

=== "GPU-Accelerated"
    ```bash
    # Leverage GPU for faster computation
    python quantum_pipeline.py \
        --file data/molecules.json \
        --gpu \
        --simulation-method statevector \
        --max-iterations 150
    ```

=== "Data Streaming"
    ```bash
    # Stream results to Kafka
    python quantum_pipeline.py \
        --file data/molecules.json \
        --kafka \
        --max-iterations 100
    ```

---

## Python API

For programmatic control and integration into larger applications.

### Basic Example

```python
from quantum_pipeline.runners.vqe_runner import VQERunner

# Create VQE runner
runner = VQERunner(
    filepath='data/molecules.json',
    basis_set='sto3g',
    max_iterations=100,
    optimizer='COBYLA',
    ansatz_reps=2
)

# Execute simulation
runner.run()
```

### With GPU Acceleration

```python
from quantum_pipeline.runners.vqe_runner import VQERunner
from quantum_pipeline.configs.module.backend import BackendConfig

backend_config = BackendConfig(
    local=True,
    gpu=True,
    optimization_level=3,
    min_num_qubits=None,
    filters=None,
    simulation_method='statevector',
    gpu_opts=None,
    noise=None,
)

runner = VQERunner(
    filepath='data/molecules.json',
    basis_set='sto3g',
    max_iterations=100,
    optimizer='L-BFGS-B',
    backend_config=backend_config,
)

runner.run()
```

### Convergence-Based Optimization

```python
runner = VQERunner(
    filepath='data/molecules.json',
    basis_set='sto3g',
    convergence_threshold=1e-6,
    optimizer='L-BFGS-B'
)

runner.run()
```

!!! warning "Max Iterations vs Convergence"
    Never set both `max_iterations` and `convergence_threshold`. They are mutually exclusive.

---

## Configuration Files

For complex setups, use configuration files to manage parameters.

### Saving Configuration

Save current settings to a file:

```bash
python quantum_pipeline.py \
    --file molecules.json \
    --basis cc-pvdz \
    --max-iterations 200 \
    --optimizer L-BFGS-B \
    --dump my_config.json
```

### Loading Configuration

Load and run with saved configuration:

```bash
python quantum_pipeline.py --load my_config.json
```

### Example Configuration File

```json
{
    "file": "data/molecules.json",
    "basis": "sto3g",
    "max_iterations": 150,
    "optimizer": "L-BFGS-B",
    "ansatz_reps": 2,
    "simulation_method": "statevector",
    "shots": 1024,
    "optimization_level": 3,
    "gpu": true,
    "kafka": true,
    "report": true,
    "log_level": "INFO"
}
```

---

## Working with Molecule Files

### Single Molecule

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

### Multiple Molecules

```json
[
    {
        "symbols": ["H", "H"],
        "coords": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        "multiplicity": 1,
        "charge": 0,
        "units": "angstrom"
    },
    {
        "symbols": ["Li", "H"],
        "coords": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.5949]],
        "multiplicity": 1,
        "charge": 0,
        "units": "angstrom"
    }
]
```

### Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `symbols` | list[str] | Yes | Atomic symbols (e.g., "H", "O", "C") |
| `coords` | list[list[float]] | Yes | 3D coordinates for each atom |
| `multiplicity` | int | Yes | Spin multiplicity (1=singlet, 2=doublet, etc.) |
| `charge` | int | Yes | Total molecular charge |
| `units` | str | Yes | Coordinate units ("angstrom" or "bohr") |
| `masses` | list[float] | No | Atomic masses (auto-detected if omitted) |

---

## Output and Logging

### Log Levels

Control verbosity with `--log-level`:

```bash
# Minimal output
python quantum_pipeline.py --file molecules.json --log-level ERROR

# Standard output (default)
python quantum_pipeline.py --file molecules.json --log-level INFO

# Verbose output for debugging
python quantum_pipeline.py --file molecules.json --log-level DEBUG
```

### Output Directory

Specify where to save results:

```bash
python quantum_pipeline.py \
    --file molecules.json \
    --output-dir ./results \
    --report
```

Default output structure:

```
results/
├── reports/          # PDF reports
├── metrics/          # Performance metrics
└── visualizations/   # Plots and charts
```

---

## Performance Monitoring

Enable comprehensive performance tracking:

```bash
python quantum_pipeline.py \
    --file molecules.json \
    --enable-performance-monitoring \
    --performance-interval 30 \
    --performance-pushgateway http://localhost:9091 \
    --performance-export-format both
```

Or via environment variables:

```bash
export QUANTUM_PERFORMANCE_ENABLED=true
export QUANTUM_PERFORMANCE_PUSHGATEWAY_URL=http://localhost:9091
python quantum_pipeline.py --file molecules.json
```

See [Monitoring Guide](../monitoring/index.md) for details.

---

## Common Parameter Combinations

### Fast Prototyping

```bash
python quantum_pipeline.py \
    --file molecules.json \
    --basis sto3g \
    --max-iterations 50 \
    --optimizer COBYLA
```

- Fastest execution
- Lower accuracy
- Good for testing

### Balanced Performance

```bash
python quantum_pipeline.py \
    --file molecules.json \
    --basis sto3g \
    --max-iterations 150 \
    --optimizer L-BFGS-B \
    --ansatz-reps 2
```

- Moderate speed
- Good accuracy
- Recommended for most use cases

### High Accuracy

```bash
python quantum_pipeline.py \
    --file molecules.json \
    --basis cc-pvdz \
    --convergence \
    --threshold 1e-8 \
    --optimizer L-BFGS-B \
    --ansatz-reps 5 \
    --shots 4096
```

- Slower execution
- Highest accuracy
- For quality results

---

## Working with Results

### Accessing Results in Python

```python
from quantum_pipeline.runners.vqe_runner import VQERunner

runner = VQERunner(filepath='molecules.json', basis_set='sto3g', max_iterations=100)
runner.run()

# Access results
for result in runner.run_results:
    print(f"Molecule: {result.molecule.symbols}")
    print(f"Ground State Energy: {result.vqe_result.minimum} Hartree")
    print(f"Iterations: {len(result.vqe_result.iteration_list)}")
    print(f"Total Time: {result.total_time} seconds")
```

### Result Structure

Each `VQEDecoratedResult` contains:

- `vqe_result`: VQE optimization results
    - `minimum`: Optimized ground state energy
    - `optimal_parameters`: Final ansatz parameters
    - `iteration_list`: Full iteration history
- `molecule`: Molecular information
- `basis_set`: Basis set used
- `hamiltonian_time`: Time to build Hamiltonian
- `mapping_time`: Time to map to qubits
- `vqe_time`: VQE optimization time
- `total_time`: Total execution time

---

## Error Handling

### Common Errors

??? error "FileNotFoundError: Molecule file not found"
    **Cause**: Specified molecule file doesn't exist

    **Solution**:
    ```bash
    # Check file path
    ls -la data/molecules.json

    # Use absolute path
    realpath data/molecules.json
    python quantum_pipeline.py --file /full/path/to/data/molecules.json
    ```

??? error "ValueError: Cannot use both max_iterations and convergence"
    **Cause**: Both `--max-iterations` and `--convergence` specified

    **Solution**: Choose one:
    ```bash
    # Option 1: Fixed iterations
    python quantum_pipeline.py --file molecules.json --max-iterations 100

    # Option 2: Convergence threshold
    python quantum_pipeline.py --file molecules.json --convergence --threshold 1e-6
    ```

??? error "RuntimeError: GPU not available"
    **Cause**: GPU requested but CUDA not properly configured

    **Solution**:
    ```bash
    # Check CUDA availability
    nvidia-smi

    # Fall back to CPU
    python quantum_pipeline.py --file molecules.json  # Remove --gpu flag
    ```

---

## Next Steps

- **[Configuration Reference](../usage/configuration.md)** - All available parameters
- **[Optimizer Guide](../usage/optimizers.md)** - Choose the right optimizer
- **[Simulation Methods](../usage/simulation-methods.md)** - Backend configuration
- **[Examples](../usage/examples.md)** - Real-world use cases
