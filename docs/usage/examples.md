# Examples

Real-world usage examples demonstrating common Quantum Pipeline workflows, from simple simulations to full data pipeline integration.

---

## Example 1: Simple H2 Simulation

**Goal**: Run a minimal VQE simulation for a hydrogen molecule (H2) to verify the pipeline is working.

### Command

```bash
python quantum_pipeline.py \
    -f data/molecules.json \
    --basis sto3g \
    --max-iterations 50 \
    --optimizer L-BFGS-B
```

### Expected Output

```
QuantumPipeline - INFO - Loading molecule data from data/molecules.json
VQERunner - INFO - Processing molecule 1:

Molecule:
    Multiplicity: 1
    Charge: 0
    Unit: Angstrom
    Geometry:
        H   [0.0, 0.0, 0.0]
        H   [0.0, 0.0, 0.74]
    Masses:
        H   1.008
        H   1.008

VQERunner - INFO - Generating hamiltonian based on the molecule...
VQERunner - INFO - Hamiltonian generated in 0.088 seconds.
VQERunner - INFO - Mapping fermionic operator to qubits
VQERunner - INFO - Problem mapped to qubits in 0.023 seconds.
VQERunner - INFO - Running VQE procedure...
VQESolver - INFO - Initializing Aer simulator backend...
VQESolver - INFO - Aer simulator backend initialized.
VQESolver - INFO - Initializing the ansatz...
VQESolver - INFO - Ansatz initialized.
VQESolver - INFO - Optimizing ansatz and hamiltonian...
VQESolver - INFO - Ansatz and hamiltonian optimized.
VQESolver - INFO - Starting the minimization process with max iterations equal to 50.
VQESolver - INFO - Simulation via Aer completed in 5.23 seconds and 50 iterations.
VQERunner - INFO - VQE procedure completed in 5.32 seconds
VQERunner - INFO - Result provided in 5.43 seconds.
```

### Notes

- The `sto3g` basis set is the smallest and fastest, suitable for quick tests.
- H2 maps to 4 qubits with `sto3g`, making it an ideal test case.
- The expected ground state energy for H2 at 0.74 Angstrom bond length is approximately -1.137 Hartree.
- Use this pattern to verify your installation before running larger simulations.

---

## Example 2: Multi-Molecule Batch Processing

**Goal**: Process multiple molecules (H2, LiH, H2O) in a single run using the same simulation configuration.

### Molecule Data File

Create a file `data/molecules.json` containing multiple molecules:

```json
[
    {
        "symbols": ["H", "H"],
        "coords": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        "multiplicity": 1,
        "charge": 0,
        "units": "angstrom",
        "masses": [1.008, 1.008]
    },
    {
        "symbols": ["Li", "H"],
        "coords": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.6]],
        "multiplicity": 1,
        "charge": 0,
        "units": "angstrom",
        "masses": [6.941, 1.008]
    },
    {
        "symbols": ["O", "H", "H"],
        "coords": [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]],
        "multiplicity": 1,
        "charge": 0,
        "units": "angstrom",
        "masses": [15.999, 1.008, 1.008]
    }
]
```

### Command

```bash
python quantum_pipeline.py \
    -f data/molecules.json \
    --basis sto3g \
    --max-iterations 100 \
    --optimizer L-BFGS-B \
    --ansatz-reps 2 \
    --shots 1024 \
    --report
```

### Expected Behavior

- The pipeline processes each molecule sequentially.
- For each molecule, it generates the Hamiltonian, maps to qubits, and runs VQE.
- A PDF report is generated in `gen/` containing results for all molecules.
- Processing time varies by molecule complexity: H2 (seconds), LiH (tens of seconds), H2O (minutes).

### Notes

- Larger molecules require more qubits: H2 uses 4, LiH uses 12, H2O uses 14 (with `sto3g`).
- Use `--report` to generate a consolidated PDF with convergence plots for each molecule.
- For large batches, consider adding `--enable-performance-monitoring` to track resource usage.

---

## Example 3: GPU-Accelerated Research

**Goal**: Run a high-accuracy simulation using GPU acceleration with the `cc-pvdz` basis set for publication-quality results.

### Command

```bash
python quantum_pipeline.py \
    -f data/molecules.json \
    --basis cc-pvdz \
    --convergence \
    --threshold 1e-8 \
    --optimizer L-BFGS-B \
    --ansatz-reps 5 \
    --shots 4096 \
    --optimization-level 3 \
    --gpu \
    --simulation-method statevector \
    --report \
    --enable-performance-monitoring
```

### Expected Behavior

- GPU acceleration is enabled, offloading statevector operations to the NVIDIA GPU.
- The `cc-pvdz` basis set provides high accuracy but requires significantly more qubits (e.g., 58 qubits for H2O).
- Convergence mode runs until the energy change falls below $10^{-8}$ Hartree.
- Performance monitoring tracks CPU, GPU, and memory usage throughout the simulation.
- A detailed PDF report is generated with convergence curves and energy analysis.

### Notes

- GPU acceleration provides 10-100x speedup for large circuits compared to CPU-only execution.
- L-BFGS-B is the recommended optimizer for GPU workloads due to its efficient convergence.
- The `cc-pvdz` basis set generates many more qubits than `sto3g`. Ensure you have sufficient GPU memory (6+ GB recommended).
- Monitor GPU utilization with `nvidia-smi` in a separate terminal during the run.
- For circuits exceeding GPU memory, consider switching to `--simulation-method tensor_network`.

---

## Example 4: Full Data Pipeline

**Goal**: Run a VQE simulation with Kafka streaming enabled, demonstrating the complete data pipeline from simulation to storage.

### Prerequisites

Start the infrastructure stack using Docker Compose:

```bash
docker compose -f docker-compose.thesis.yaml up -d
```

This starts Kafka, Schema Registry, Kafka Connect, MinIO, Airflow, and Spark.

### Command

```bash
python quantum_pipeline.py \
    -f data/molecules.json \
    --basis sto3g \
    --max-iterations 150 \
    --optimizer L-BFGS-B \
    --gpu \
    --simulation-method statevector \
    --kafka \
    --servers kafka:9092 \
    --topic vqe_decorated_result \
    --acks all \
    --retries 3
```

### Expected Behavior

1. The VQE simulation runs with GPU acceleration.
2. Upon completion, results are serialized using Avro format.
3. The Avro schema is registered with the Schema Registry (auto-generated from result structure).
4. The serialized result is published to the Kafka topic.
5. Kafka Connect picks up the message and writes it to MinIO as an Avro file.
6. The result is stored in a path like:
   ```
   experiments/vqe_decorated_result_mol0_HH_it150_bs_sto3g_bk_aer_simulator_statevector_gpu/
       partition=0/
           vqe_decorated_result_mol0_HH_it150_bs_sto3g_bk_aer_simulator_statevector_gpu+0+0000000000.avro
   ```

### Topic Naming Convention

The Kafka topic is automatically suffixed with experiment metadata:

```
vqe_decorated_result_mol0_HH_it150_bs_sto3g_bk_aer_simulator_statevector_gpu
```

This encodes: molecule index (`mol0`), atoms (`HH`), iterations (`it150`), basis set (`sto3g`), and backend (`aer_simulator_statevector_gpu`).

### Notes

- The `--kafka` flag is required to enable streaming. Without it, other Kafka parameters are ignored.
- Use `--acks all` in production to ensure data durability across Kafka replicas.
- The Schema Registry caches schemas for repeated simulations with the same configuration.
- After Kafka Connect transfers data to MinIO, Apache Airflow can trigger Spark processing for feature engineering.

---

## Example 5: Convergence-Based Optimization

**Goal**: Use convergence-based stopping instead of a fixed iteration count, allowing the optimizer to run until the energy stabilizes.

### Command

```bash
python quantum_pipeline.py \
    -f data/molecules.json \
    --basis sto3g \
    --convergence \
    --threshold 1e-6 \
    --optimizer L-BFGS-B \
    --shots 2048
```

### Expected Behavior

- The optimizer runs until the energy change between consecutive iterations falls below $10^{-6}$ Hartree.
- The total number of iterations is determined automatically based on convergence.
- For well-behaved problems (H2 with L-BFGS-B), convergence typically occurs within 30-80 iterations.
- For more complex molecules, convergence may require hundreds of iterations.

### Threshold Guidelines

| Threshold | Precision Level | Typical Use Case |
|-----------|----------------|------------------|
| `1.6e-3` | Chemical accuracy (~1 kcal/mol) | Fast prototyping |
| `1e-6` | Standard (1 microHartree) | Production runs |
| `1e-8` | High precision (10 nanoHartree) | Publication-quality research |

### Notes

- Convergence mode (`--convergence`) and fixed iterations (`--max-iterations`) are mutually exclusive. Using both will raise a `ValueError`.
- Convergence mode is recommended for production runs because it adapts to the problem difficulty.
- Pair with `--report` to visualize the convergence curve in the generated PDF.
- The convergence threshold is passed to `scipy.optimize.minimize` as the `tol` parameter.

---

## Example 6: Configuration Save and Load

**Goal**: Save a simulation configuration for reproducibility and reload it later for repeated experiments.

### Step 1: Run and Save Configuration

```bash
python quantum_pipeline.py \
    -f data/molecules.json \
    --basis cc-pvdz \
    --convergence \
    --threshold 1e-6 \
    --optimizer L-BFGS-B \
    --ansatz-reps 2 \
    --shots 2048 \
    --gpu \
    --simulation-method statevector \
    --dump
```

This creates a configuration file in `run_configs/` with a timestamped filename:

```json
{
    "file": "data/molecules.json",
    "basis": "cc-pvdz",
    "convergence_threshold_enable": true,
    "convergence_threshold": 1e-6,
    "optimizer": "L-BFGS-B",
    "ansatz_reps": 2,
    "shots": 2048,
    "backend": {
        "local": true,
        "optimization_level": 3,
        "method": "statevector",
        "gpu": true,
        "noise_backend": null
    }
}
```

### Step 2: Reload Configuration

```bash
python quantum_pipeline.py --load run_configs/config_20250615.json
```

### Step 3: Reload with Overrides

```bash
# Load saved config but change the optimizer
python quantum_pipeline.py \
    --load run_configs/config_20250615.json \
    --optimizer COBYLA

# Load saved config but switch to a different basis set
python quantum_pipeline.py \
    --load run_configs/config_20250615.json \
    --basis sto3g
```

### Notes

- `--dump` and `--load` are mutually exclusive. You cannot save and load in the same run.
- CLI arguments override loaded configuration values, allowing selective parameter changes.
- Use `--dump` for every production run to maintain an audit trail of experiment configurations.
- Configuration files are stored in `run_configs/` with timestamps for easy identification.

---

## Example 7: Custom Python Script

**Goal**: Use the `VQERunner` API programmatically in a custom Python script for advanced workflows, automated parameter sweeps, or integration with external tools.

### Script

```python
from quantum_pipeline.runners.vqe_runner import VQERunner
from quantum_pipeline.configs.module.backend import BackendConfig

# Configure the backend
backend_config = BackendConfig(
    local=True,
    optimization_level=3,
    min_num_qubits=None,
    filters=None,
    simulation_method='statevector',
    gpu=False,
    gpu_opts=None,
    noise=None,
)

# Initialize the VQE runner
runner = VQERunner(
    filepath='data/molecules.json',
    basis_set='sto3g',
    max_iterations=100,
    optimizer='L-BFGS-B',
    ansatz_reps=2,
    default_shots=1024,
    report=False,
    kafka=False,
    backend_config=backend_config,
)

# Run the simulation
runner.run()

# Process results
for result in runner.run_results:
    print(f"Molecule: {result.molecule.molecule_data.symbols}")
    print(f"Ground state energy: {result.vqe_result.minimum:.6f} Ha")
    print(f"Iterations: {len(result.vqe_result.iteration_list)}")
    print(f"VQE time: {result.vqe_time:.2f} seconds")
    print(f"Total time: {result.total_time:.2f} seconds")
    print("---")
```

### Parameter Sweep Example

```python
from quantum_pipeline.runners.vqe_runner import VQERunner

optimizers = ['L-BFGS-B', 'COBYLA', 'BFGS', 'SLSQP']
results_summary = []

for opt in optimizers:
    runner = VQERunner(
        filepath='data/molecules.json',
        basis_set='sto3g',
        max_iterations=100,
        optimizer=opt,
        ansatz_reps=2,
        default_shots=1024,
    )

    runner.run()

    for result in runner.run_results:
        results_summary.append({
            'optimizer': opt,
            'energy': result.vqe_result.minimum,
            'iterations': len(result.vqe_result.iteration_list),
            'time': result.vqe_time,
        })

# Compare optimizer performance
for entry in results_summary:
    print(f"{entry['optimizer']:12s} | "
          f"Energy: {entry['energy']:12.6f} Ha | "
          f"Iterations: {entry['iterations']:4d} | "
          f"Time: {entry['time']:.2f}s")
```

### Notes

- The `VQERunner` class accepts the same parameters as the CLI but as constructor arguments.
- Use `BackendConfig`, `ProducerConfig`, and `SecurityConfig` for structured configuration.
- Programmatic usage is ideal for parameter sweeps, automated benchmarks, and integration with analysis notebooks.
- Results are returned as `VQEDecoratedResult` objects containing the full simulation data, including per-iteration energy values.

---

## Example 8: Thesis Experiment Reproduction

**Goal**: Reproduce the experiments from the thesis using the provided environment configuration file.

### Step 1: Set Up Environment

Copy the thesis environment example and customize it for your system:

```bash
cp .env.thesis.example .env
```

Review and update the key settings in `.env`:

```bash
# Quantum Pipeline Settings
MAX_ITERATIONS=50
LOG_LEVEL=INFO
SIMULATION_METHOD=statevector
BASIS_SET=sto3g

# Kafka Configuration
KAFKA_SERVERS=kafka:9092

# MinIO Configuration
MINIO_ROOT_USER=quantum-admin
MINIO_ROOT_PASSWORD=quantum-secret-key

# Performance Monitoring
QUANTUM_PERFORMANCE_ENABLED=true
QUANTUM_PERFORMANCE_COLLECTION_INTERVAL=10
QUANTUM_PERFORMANCE_PUSHGATEWAY_URL=http://monit:9091
QUANTUM_PERFORMANCE_EXPORT_FORMAT=json,prometheus
```

### Step 2: Start the Full Stack

```bash
docker compose -f docker-compose.thesis.yaml up -d
```

This starts the entire infrastructure:

- Quantum Pipeline container (with GPU support)
- Apache Kafka and Schema Registry
- Kafka Connect (S3 sink to MinIO)
- MinIO object storage
- Apache Airflow (DAG orchestration)
- Apache Spark (feature engineering)
- Prometheus and Grafana (monitoring)

### Step 3: Monitor the Experiment

The thesis configuration runs the pipeline with convergence-based optimization, GPU acceleration, and Kafka streaming enabled. Monitor progress through:

```bash
# View pipeline logs
docker logs -f quantum-pipeline

# Check GPU utilization
nvidia-smi -l 1

# View Kafka topics
docker exec -it kafka kafka-topics --list --bootstrap-server localhost:9092
```

### Step 4: Access Results

- **MinIO Console**: `http://localhost:9002` - browse stored Avro files
- **Airflow UI**: `http://localhost:8084` - view DAG execution status
- **Grafana**: Access dashboards for real-time performance metrics

### Resource Allocation Reference

The thesis experiments used the following resource distribution:

| Component | CPUs | RAM | GPU |
|-----------|------|-----|-----|
| CPU Pipeline | 3 | 16 GB | -- |
| GPU Pipeline | 3 | 16 GB | GTX 1060 |
| Infrastructure | 1 | 24 GB | -- |
| Monitoring | 0.5 | 8 GB | -- |

### Notes

- The `.env.thesis.example` file contains the exact settings used in the thesis experiments.
- The thesis compared CPU vs. GPU performance across multiple molecules and optimizers.
- Three separate pipeline containers were used: one CPU-only and two GPU-accelerated (each with a different GPU).
- Results were streamed via Kafka to MinIO, then processed by Spark through Airflow DAGs for feature engineering.
- All IBM Quantum credentials in the example file are placeholders and must be replaced with valid tokens if cloud backend comparison is desired.

---

## Quick Reference

| Example | Use Case | Key Flags |
|---------|----------|-----------|
| [1. Simple H2](#example-1-simple-h2-simulation) | Installation test | `--basis sto3g --max-iterations 50` |
| [2. Batch Processing](#example-2-multi-molecule-batch-processing) | Multi-molecule runs | `--report` |
| [3. GPU Research](#example-3-gpu-accelerated-research) | High-accuracy GPU | `--gpu --basis cc-pvdz --convergence` |
| [4. Data Pipeline](#example-4-full-data-pipeline) | Kafka streaming | `--kafka --servers kafka:9092` |
| [5. Convergence](#example-5-convergence-based-optimization) | Adaptive stopping | `--convergence --threshold 1e-6` |
| [6. Save/Load](#example-6-configuration-save-and-load) | Reproducibility | `--dump` / `--load` |
| [7. Python API](#example-7-custom-python-script) | Programmatic usage | `VQERunner` class |
| [8. Thesis](#example-8-thesis-experiment-reproduction) | Full reproduction | `.env.thesis.example` |

---

## Next Steps

- **Configure parameters in detail**: [Configuration Reference](configuration.md)
- **Understand optimizer choices**: [Optimizers](optimizers.md)
- **Choose a simulation method**: [Simulation Methods](simulation-methods.md)
- **Deploy with Docker**: [Docker Compose](../deployment/docker-compose.md)
