---
title: "Quick Start: First VQE Run on H2"
description: "Run your first VQE simulation on the H2 molecule, read the output, then try convergence-based stopping, GPU acceleration and Kafka streaming of the results."
---

# Quick Start

Run a VQE simulation for the H\(_2\) molecule in a few steps.

## Molecule data

The repository includes `data/molecules.json` with 9 molecules (H\(_2\), H\(_2\)O, He\(_2\), LiH, BeH,
BH, CH\(_4\), NH\(_3\), N\(_2\)). You can use it directly.

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
| `multiplicity` | int | No | Spin multiplicity (1 = singlet, 2 = doublet, etc.). Defaults to 1 |
| `charge` | int | No | Total molecular charge. Defaults to 0 |
| `units` | str | No | `"angstrom"` or `"bohr"`. Defaults to `"angstrom"` |
| `masses` | list of float | No | Atomic masses (standard masses are used if omitted) |
| `name` | str | No | Label used in output and plots. Falls back to the joined symbols |

## Run a simulation

=== "CLI"
    ```bash
    quantum-pipeline \
        --file data/molecules.json \
        --molecule-index 0 \
        --basis sto3g \
        --max-iterations 100 \
        --optimizer L-BFGS-B \
        --ansatz-reps 3 \
        --init-strategy hf \
        --seed 42 \
        --report
    ```

=== "Docker"
    ```bash
    docker run --rm \
        -v "$(pwd)/gen:/usr/src/quantum_pipeline/gen" \
        straightchlorine/quantum-pipeline:cpu \
        --file data/molecules.json \
        --molecule-index 0 \
        --basis sto3g \
        --max-iterations 100 \
        --optimizer L-BFGS-B \
        --ansatz-reps 3 \
        --init-strategy hf \
        --seed 42 \
        --report
    ```

`--molecule-index 0` selects the first molecule (H\(_2\)). Otherwise, all molecules in the file are processed.
`--init-strategy hf`, `--ansatz-reps 3` and `--seed 42` are not required, but they make this first run reliable
and reproducible (see [Common issues](#common-issues)).

A run without `--shots` is exact (expectation values come from the statevector, no sampling),
so `--exact` is optional. With `hf` at the default 2 repetitions a gradient optimizer can
stop at the Hartree-Fock energy, which is why this run uses `--ansatz-reps 3`;
see [Hartree-Fock initialization](../scientific/vqe-algorithm.md#hartree-fock-initialization).

With Docker, the `-v` mount maps the container's `gen/` directory to a `gen/` folder in your current
directory so the report and plots survive the run.

The `--report` flag generates a PDF in `gen/` with molecular structure
visualization, energy convergence plot, and Hamiltonian operator coefficients.
See an [example report (H2, 6-31g, L-BFGS-B, HF init)](https://qp-docs.codextechnologies.org/mkdocs/quantum_report.pdf).

## Expected output

```
2026-06-13 22:30:29,961 - VQERunner - INFO - Processing molecule 1:

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

2026-06-13 22:30:30,082 - VQERunner - INFO - Hamiltonian generated in 0.120053 seconds.
2026-06-13 22:30:30,082 - VQERunner - INFO - HF reference energy: -1.116759 Ha
2026-06-13 22:30:30,100 - VQERunner - INFO - Problem mapped to qubits in 0.017953 seconds.
2026-06-13 22:30:30,100 - VQERunner - INFO - Running VQE procedure...
2026-06-13 22:30:30,105 - VQESolver - INFO - Analytic-gradient statevector path: 4 qubits, 32 parameters, optimizer L-BFGS-B.
...
2026-06-13 22:30:31,402 - VQESolver - INFO - VQE completed 54 iterations. Best energy: -1.85238817 Ha
2026-06-13 22:30:31,402 - VQESolver - INFO - Total energy (electronic + nuclear repulsion): -1.13728383 Ha (nuclear repulsion: 0.71510434 Ha)
...
2026-06-13 22:30:31,404 - VQERunner - INFO -   VQE Total Energy: -1.137284 Ha
2026-06-13 22:30:31,404 - VQERunner - INFO -   HF Reference:     -1.116759 Ha
2026-06-13 22:30:31,404 - VQERunner - INFO -   Deviation:        -20.525 mHa
2026-06-13 22:30:31,404 - VQERunner - INFO -   HF-deviation score: 73.3/100
```

The VQESolver line shows the analytic path: on a local, exact, noiseless run with a gradient
optimizer and at most 16 qubits, energies and gradients are computed directly on the statevector
(see [`--exact`](../usage/configuration.md#-exact)).

The energies are reported in the logs and in the PDF report, not as a single
"final energy" line. The literature ground-state energy for H\(_2\) with STO-3G
is about -1.137 Ha; this run reaches -1.13728 Ha, the exact (FCI) value for this basis.
The "Best energy" line is the electronic part only; the total energy is on the
"Total energy" line. Timings and the exact iteration count vary by machine.
Random initialization (the default) can settle in a higher local minimum, so the
exact value varies between runs (see [Common issues](#common-issues)).

## Key parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `-f` / `--file` | Path to molecule JSON file (required) | - |
| `--molecule-index` | 0-based index of a single molecule to process | all |
| `-b` / `--basis` | Basis set: `sto3g`, `6-31g`, `cc-pvdz` | `sto3g` |
| `--max-iterations` | Maximum VQE iterations | `100` |
| `--optimizer` | Optimization algorithm (see [Optimizer Guide](../usage/optimizers.md)) | `L-BFGS-B` |
| `--ansatz` | Ansatz type: `EfficientSU2`, `RealAmplitudes`, `ExcitationPreserving` | `EfficientSU2` |
| `-ar` / `--ansatz-reps` | Ansatz repetition depth | `2` |
| `--init-strategy` | Parameter initialization: `random` or `hf` (use `hf` with `--ansatz-reps 3`, see [Common issues](#common-issues)) | `random` |
| `--seed` | Random seed for reproducibility | None |
| `--shots` | Sample with this many shots per circuit execution; without it the run is exact | none (exact) |
| `--exact` | Exact expectation values. Already the default without `--shots`; overrides `--shots` | off |
| `--simulation-method` | Simulator backend method | `statevector` |
| `--gpu` | Enable GPU acceleration | off |
| `--report` | Generate a PDF report | off |
| `--kafka` | Stream results to Kafka | off |
| `--convergence` | Enable convergence threshold | off |
| `--threshold` | Convergence threshold value (requires `--convergence`) | `1e-6` |
| `--log-level` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | `INFO` |

See [Configuration](../usage/configuration.md) for the full parameter reference.

## Convergence-based optimization

When `--convergence` is enabled, the default iteration limit (100) is dropped and the optimizer
runs until the threshold is met.

If `--convergence` is set, `--max-iterations` is ignored, with a warning if it was
set to a non-default value.

```bash
quantum-pipeline \
    --file data/molecules.json \
    --molecule-index 0 \
    --basis sto3g \
    --convergence --threshold 1e-6 \
    --optimizer L-BFGS-B
```

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
    --file data/molecules.json \
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

This requires Kafka and Schema Registry to be running. If the broker is unreachable, the run exits with code 1 after spooling the results to `gen/undelivered/`. See [Kafka Streaming](../data-platform/kafka-streaming.md).

## Common issues

**Simulation takes too long?** Use fixed iterations instead of `--convergence`. Consider also starting with `sto3g` basis set for faster execution.

**Results do not match reference energies?** Random initialization can trap the optimizer in local minima.
Try `--init-strategy hf --ansatz-reps 3` for Hartree-Fock-based initialization, or increase `--max-iterations`.
With `hf` at the default 2 repetitions a gradient optimizer can stop at the Hartree-Fock energy; see [Hartree-Fock initialization](../scientific/vqe-algorithm.md#hartree-fock-initialization).

**Memory errors with large molecules?** Use `--simulation-method matrix_product_state`, reduce `--ansatz-reps`, or use a smaller basis set.

See the [Troubleshooting Guide](../reference/troubleshooting.md) for more.

## Next steps

- [Basic Usage](basic-usage.md) - workflows, configuration files, output structure
- [Optimizer Guide](../usage/optimizers.md) - choosing the right optimizer
- [Architecture](../architecture/index.md) - how the system works
