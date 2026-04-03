# Examples

Usage examples for common Quantum Pipeline workflows, from simple simulations to full data pipeline integration.

## Simple H\(_2\) Simulation

Run a minimal VQE simulation for a hydrogen molecule to verify the pipeline works.

```bash
quantum-pipeline \
    -f data/molecules.json \
    --basis sto3g \
    --max-iterations 50 \
    --optimizer L-BFGS-B
```

The `sto3g` basis set is the smallest and fastest, good for quick tests. H\(_2\) maps to 4 qubits with `sto3g`, making it a simple test case. The expected ground state energy for H\(_2\) at 0.74 Angstrom bond length is approximately -1.137 Hartree.

## Multi-Molecule Batch Processing

Process multiple molecules in a single run using the same simulation configuration. The pipeline processes each molecule sequentially - for each one it generates the Hamiltonian, maps to qubits, and runs VQE.

The molecule data file is a JSON array. See the included [`data/molecules.json`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/data/molecules.json) for the format.

| Field | Type | Description |
|-------|------|-------------|
| `symbols` | string array | Atom symbols (e.g. `["H", "H"]`) |
| `coords` | float array | 3D coordinates per atom |
| `multiplicity` | int | Spin multiplicity |
| `charge` | int | Net charge |
| `units` | string | Distance unit (e.g. `"angstrom"`) |
| `masses` | float array | Atomic masses |

```bash
quantum-pipeline \
    -f data/molecules.json \
    --basis sto3g \
    --max-iterations 100 \
    --optimizer L-BFGS-B \
    --ansatz-reps 2 \
    --shots 1024 \
    --report
```

Use `--report` to generate a [PDF report](https://qp-docs.codextechnologies.org/mkdocs/quantum_report.pdf) in `gen/` with molecular structure, convergence plots, and Hamiltonian coefficients. Ansatz circuit diagrams are saved separately in `gen/graphs/` ([example](https://qp-docs.codextechnologies.org/mkdocs/ansatz_H_H.png), [decomposed](https://qp-docs.codextechnologies.org/mkdocs/ansatz_decomposed_H_H.png)). Larger molecules need more qubits: H\(_2\) uses 4, LiH uses 12, H\(_2\)O uses 14 (with `sto3g`). Processing time varies accordingly.

## GPU-Accelerated Simulation

Run a higher-accuracy simulation using GPU acceleration with the `cc-pvdz` basis set.

```bash
quantum-pipeline \
    -f data/molecules.json \
    --basis cc-pvdz \
    --convergence \
    --threshold 1e-8 \
    --optimizer L-BFGS-B \
    --init-strategy hf \
    --ansatz-reps 5 \
    --shots 4096 \
    --optimization-level 3 \
    --gpu \
    --simulation-method statevector \
    --report
```

GPU acceleration offloads statevector operations to the NVIDIA GPU. The `cc-pvdz` basis set provides higher accuracy but needs significantly more qubits (e.g. 58 for H\(_2\)O). Convergence mode runs until the energy change falls below the threshold.

The `--init-strategy hf` flag pre-optimizes parameters using Hartree-Fock data, which helps avoid local minima that are common with `cc-pvdz`. This strategy only works with the `EfficientSU2` ansatz - using it with other ansatze falls back to random initialization.

!!! note "GPU performance"
    In thesis benchmarks (GTX 1060/1050 Ti, H\(_2\) molecule), GPU acceleration provided 1.7-4x speedup depending on basis set and circuit size. The `cc-pvdz` basis set generates many more qubits than `sto3g`, so ensure you have enough GPU memory (6+ GB recommended). For circuits exceeding GPU memory, consider `--simulation-method tensor_network`.

## Full Data Pipeline with Kafka

Run a VQE simulation with Kafka streaming enabled, demonstrating the complete data pipeline from simulation to storage.

Start the infrastructure stack first using Docker Compose. See the [`compose/`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/compose) directory and the [Docker Compose](../deployment/docker-compose.md) deployment guide for available configurations.

```bash
quantum-pipeline \
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

What happens:

1. The VQE simulation runs with GPU acceleration.
2. Results are serialized using Avro format and the schema is registered with the Schema Registry.
3. The serialized result is published to the Kafka topic.
4. Redpanda Connect picks up the message and writes it to Garage (S3-compatible storage) as an Avro file.

The `--kafka` flag is required to enable streaming. Without it, other Kafka parameters are ignored. After Redpanda Connect transfers data to Garage, Apache Airflow can trigger Spark processing for feature engineering.

## Convergence-Based Optimization

Use convergence-based stopping instead of a fixed iteration count, allowing the optimizer to run until the energy stabilizes.

```bash
quantum-pipeline \
    -f data/molecules.json \
    --basis sto3g \
    --convergence \
    --threshold 1e-6 \
    --optimizer L-BFGS-B \
    --shots 2048
```

The optimizer runs until the energy change between consecutive iterations falls below the threshold. For well-behaved problems (H\(_2\) with L-BFGS-B), convergence typically occurs within 30-80 iterations. For more complex molecules, it may require hundreds.

### Threshold guidelines

| Threshold | Precision level | Typical use case |
|-----------|----------------|------------------|
| `1.6e-3` | Chemical accuracy (~1 kcal/mol) | Fast prototyping |
| `1e-6` | Standard (1 microHartree) | General runs |
| `1e-8` | High precision (10 nanoHartree) | Publication-quality research |

Convergence mode (`--convergence`) and fixed iterations (`--max-iterations`) are mutually exclusive at the `OptimizerConfig` level. If `--convergence` is enabled and `--max-iterations` is left at the default, the entry point sets `max_iterations` to `None`. Passing both non-default values raises a `ValueError`.

For L-BFGS-B, the convergence threshold is passed as `ftol` and `gtol` in the options dict. For COBYLA and other optimizers, it is passed as the global `tol` parameter to `scipy.optimize.minimize`.

## Configuration Save and Load

Save a simulation configuration for reproducibility and reload it later.

### Save configuration

Adding `--dump` saves the run configuration as a JSON file in `run_configs/` with a timestamped filename.

```bash
quantum-pipeline \
    -f data/molecules.json \
    --basis cc-pvdz \
    --convergence \
    --threshold 1e-6 \
    --optimizer L-BFGS-B \
    --ansatz RealAmplitudes \
    --init-strategy hf \
    --ansatz-reps 2 \
    --shots 2048 \
    --gpu \
    --simulation-method statevector \
    --dump
```

### Load and override

```bash
# Reload a saved configuration
quantum-pipeline --load run_configs/config_20250615.json

# Load saved config but change the optimizer
quantum-pipeline \
    --load run_configs/config_20250615.json \
    --optimizer COBYLA

# Load saved config but switch basis set
quantum-pipeline \
    --load run_configs/config_20250615.json \
    --basis sto3g
```

`--dump` and `--load` are mutually exclusive. CLI arguments override loaded configuration values, allowing selective parameter changes.

## Programmatic Usage

The pipeline can also be used as a Python library through the [`VQERunner`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/runners/vqe_runner.py) class. It accepts the same parameters as the CLI but as constructor arguments, and returns [`VQEDecoratedResult`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/structures/vqe_observation.py) objects containing the full simulation data, including per-iteration energy values.

Backend, Kafka producer, and security settings can be passed as [`BackendConfig`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/configs/module/backend.py), [`ProducerConfig`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/configs/module/producer.py), and [`SecurityConfig`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/configs/module/security.py) objects respectively.

This is useful for parameter sweeps, automated benchmarks, and integration with analysis notebooks.

## Thesis Experiment Reproduction

The thesis experiments can be reproduced using the provided environment configuration. Copy the [`.env.thesis.example`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/.env.thesis.example) file and customize it for your system:

```bash
cp .env.thesis.example .env
```

Start the full stack using Docker Compose (see the [Docker Compose](../deployment/docker-compose.md) deployment guide), then monitor progress through container logs and `nvidia-smi`.

### Resource allocation reference

The thesis experiments used the following distribution:

| Component | CPUs | RAM | GPU |
|-----------|------|-----|-----|
| CPU Pipeline | 3 | 16 GB | - |
| GPU Pipeline | 3 | 16 GB | GTX 1060 |
| Infrastructure | 1 | 24 GB | - |
| Monitoring | 0.5 | 8 GB | - |

The thesis compared CPU vs. GPU performance across multiple molecules and optimizers. Three separate pipeline containers were used: one CPU-only and two GPU-accelerated (each with a different GPU). Results were streamed via Kafka to Garage (S3), then processed by Spark through Airflow DAGs for feature engineering.

## Ansatz and Init Strategy Comparison

Compare different ansatz types and initialization strategies to find the best combination for a given molecule.

```bash
# RealAmplitudes with random init
quantum-pipeline \
    -f data/molecules.json \
    --basis sto3g \
    --max-iterations 100 \
    --optimizer L-BFGS-B \
    --ansatz RealAmplitudes \
    --init-strategy random \
    --seed 42 \
    --report

# EfficientSU2 with Hartree-Fock init
quantum-pipeline \
    -f data/molecules.json \
    --basis cc-pvdz \
    --max-iterations 200 \
    --optimizer L-BFGS-B \
    --ansatz EfficientSU2 \
    --init-strategy hf \
    --seed 42 \
    --report

# ExcitationPreserving ansatz
quantum-pipeline \
    -f data/molecules.json \
    --basis sto3g \
    --max-iterations 100 \
    --optimizer L-BFGS-B \
    --ansatz ExcitationPreserving \
    --ansatz-reps 3 \
    --report
```

The `--seed` flag ensures reproducible results across runs with the same configuration. The `--init-strategy hf` flag only works with `EfficientSU2` - using it with other ansatze falls back to random initialization. Use `--report` to generate convergence plots for visual comparison.

The three supported ansatze are:

| Ansatz | Description |
|--------|-------------|
| `EfficientSU2` | Default. General-purpose with Ry and Rz rotation gates and CNOT entanglement. |
| `RealAmplitudes` | Uses only real-valued Ry rotation gates. Suitable for Hamiltonians with real coefficients. |
| `ExcitationPreserving` | Preserves particle number, which can be physically motivated for molecular systems. |

## Quick reference

| Example | Use case | Key flags |
|---------|----------|-----------|
| [Simple H\(_2\)](#simple-h2-simulation) | Installation test | `--basis sto3g --max-iterations 50` |
| [Batch Processing](#multi-molecule-batch-processing) | Multi-molecule runs | `--report` |
| [GPU Research](#gpu-accelerated-simulation) | High-accuracy GPU | `--gpu --basis cc-pvdz --init-strategy hf` |
| [Data Pipeline](#full-data-pipeline-with-kafka) | Kafka streaming | `--kafka --servers kafka:9092` |
| [Convergence](#convergence-based-optimization) | Adaptive stopping | `--convergence --threshold 1e-6` |
| [Save/Load](#configuration-save-and-load) | Reproducibility | `--dump` / `--load` |
| [Python API](#programmatic-usage) | Programmatic usage | `VQERunner` class |
| [Thesis](#thesis-experiment-reproduction) | Full reproduction | `.env.thesis.example` |
| [Ansatz/Init](#ansatz-and-init-strategy-comparison) | Ansatz comparison | `--ansatz --init-strategy` |

## Next steps

- [Configuration Reference](configuration.md) for detailed parameter documentation
- [Optimizers](optimizers.md) for optimizer choices
- [Simulation Methods](simulation-methods.md) for backend selection
- [Docker Compose](../deployment/docker-compose.md) for deployment
