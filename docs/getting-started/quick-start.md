# Quick Start

Run a VQE simulation for the H\(_2\) molecule in a few steps.

## Molecule data

The repository includes `data/molecules.json` with 9 molecules (H\(_2\), H\(_2\)O, He\(_2\), LiH, BeH, BH, CH\(_4\), NH\(_3\), N\(_2\)). You can use it directly.

To create your own file, use this JSON format:

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

The molecule format fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `symbols` | list of str | Yes | Atomic symbols (e.g. `"H"`, `"O"`, `"Li"`) |
| `coords` | list of [x, y, z] | Yes | 3D coordinates for each atom |
| `multiplicity` | int | Yes | Spin multiplicity (1 = singlet, 2 = doublet, etc.) |
| `charge` | int | Yes | Total molecular charge |
| `units` | str | Yes | `"angstrom"` or `"bohr"` |
| `masses` | list of float | No | Atomic masses (auto-detected if omitted) |

## Run a simulation

=== "CLI"
    ```bash
    quantum-pipeline \
        --file data/molecules.json \
        --molecule-index 0 \
        --basis sto3g \
        --max-iterations 100 \
        --optimizer L-BFGS-B \
        --report
    ```

=== "Docker"
    ```bash
    docker run --rm -v $(pwd)/data:/data \
        straightchlorine/quantum-pipeline:cpu \
        --file /data/molecules.json \
        --molecule-index 0 \
        --basis sto3g \
        --max-iterations 100 \
        --optimizer L-BFGS-B
    ```

`--molecule-index 0` selects the first molecule (H\(_2\)). Without it, all molecules in the file are processed.

The `--report` flag generates a PDF in `gen/` with molecular structure
visualization, energy convergence plot, and Hamiltonian operator coefficients.
See an [example report (H2, 6-31g, L-BFGS-B, HF init)](https://qp-docs.codextechnologies.org/mkdocs/quantum_report.pdf).

## Expected output

```
2025-06-15 11:33:36,762 - VQERunner - INFO - Processing molecule 1:

Molecule:
    Multiplicity: 1
    Charge: 0
    Unit: Angstrom
    Geometry:
        H   [0.0, 0.0, 0.0]
        H   [0.0, 0.0, 0.74]

2025-06-15 11:33:36,850 - VQERunner - INFO - Hamiltonian generated in 0.088 seconds.
2025-06-15 11:33:36,873 - VQERunner - INFO - Problem mapped to qubits in 0.023 seconds.
2025-06-15 11:33:36,873 - VQERunner - INFO - Running VQE procedure...
...
2025-06-15 11:33:47,439 - VQESolver - INFO - Simulation completed in 10.47 seconds and 100 iterations.

Final Energy: -1.1372 Hartree
```

For H\(_2\) with STO-3G, the ground state energy should be approximately -1.137 Hartree.

## Key parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `-f` / `--file` | Path to molecule JSON file (required) | - |
| `--molecule-index` | 0-based index of a single molecule to process | all |
| `-b` / `--basis` | Basis set: `sto3g`, `6-31g`, `cc-pvdz` | `sto3g` |
| `--max-iterations` | Maximum VQE iterations | `100` |
| `--optimizer` | Optimization algorithm (see below) | `L-BFGS-B` |
| `--ansatz` | Ansatz type: `EfficientSU2`, `RealAmplitudes`, `ExcitationPreserving` | `EfficientSU2` |
| `-ar` / `--ansatz-reps` | Ansatz repetition depth | `2` |
| `--init-strategy` | Parameter initialization: `random` or `hf` | `random` |
| `--seed` | Random seed for reproducibility | None |
| `--shots` | Number of shots per circuit execution | `1024` |
| `--simulation-method` | Simulator backend method | `statevector` |
| `--gpu` | Enable GPU acceleration | off |
| `--report` | Generate a PDF report | off |
| `--kafka` | Stream results to Kafka | off |
| `--convergence` | Enable convergence threshold (replaces fixed iterations) | off |
| `--threshold` | Convergence threshold value (requires `--convergence`) | `1e-6` |
| `--log-level` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | `INFO` |

See [Configuration](../usage/configuration.md) for the full parameter reference.

## Convergence-based optimization

Instead of a fixed iteration count, stop when the energy change is below a threshold:

```bash
quantum-pipeline \
    --file data/molecules.json \
    --molecule-index 0 \
    --basis sto3g \
    --convergence --threshold 1e-6 \
    --optimizer L-BFGS-B
```

Do not combine `--max-iterations` with `--convergence`. If you set `--convergence`, the default iteration limit is removed. If you explicitly set both, the iteration cap still applies but convergence is checked at each step.

## GPU acceleration

If you have an NVIDIA GPU:

```bash
quantum-pipeline \
    --file data/molecules.json \
    --basis sto3g \
    --max-iterations 100 \
    --gpu \
    --simulation-method statevector
```

Or with Docker:

```bash
docker run --rm --gpus all \
    straightchlorine/quantum-pipeline:gpu \
    --file /app/data/molecules.json \
    --gpu \
    --max-iterations 100
```

See [GPU Acceleration Guide](../deployment/gpu-acceleration.md) for setup details.

## Streaming to Kafka

Send VQE results to a Kafka topic in real time:

```bash
quantum-pipeline \
    --file data/molecules.json \
    --basis sto3g \
    --max-iterations 100 \
    --kafka
```

This requires Kafka and Schema Registry to be running. See [Kafka Streaming](../data-platform/kafka-streaming.md).

## Common issues

**Simulation takes too long?** Use fixed iterations instead of `--convergence`. Consider starting with `sto3g` basis set.

**Results do not match reference energies?** Random initialization can trap the optimizer in local minima, especially with `cc-pvdz`. Try `--init-strategy hf` for Hartree-Fock-based initialization, or increase `--max-iterations`.

**Memory errors with large molecules?** Use `--simulation-method matrix_product_state`, reduce `--ansatz-reps`, or use a smaller basis set.

See the [Troubleshooting Guide](../reference/troubleshooting.md) for more.

## Next steps

- [Basic Usage](basic-usage.md) - workflows, configuration files, output structure
- [Optimizer Guide](../usage/optimizers.md) - choosing the right optimizer
- [Architecture](../architecture/index.md) - how the system works
