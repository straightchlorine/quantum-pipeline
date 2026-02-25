# Quick Start

Get started with Quantum Pipeline in under 5 minutes by running a simple VQE simulation for the H₂ molecule.

---

## Basic VQE Simulation

### Step 1: Prepare Molecule Data

Create a file `h2_molecule.json` with the following content:

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

!!! info "Molecule Format"
    - `symbols`: Atomic symbols
    - `coords`: 3D coordinates in specified units
    - `multiplicity`: Spin multiplicity (1 = singlet, 2 = doublet, etc.)
    - `charge`: Total molecular charge
    - `units`: Coordinate units (`angstrom` or `bohr`)
    - `masses`: Atomic masses (optional)

---

### Step 2: Run VQE Simulation

=== "Command Line"
    ```bash
    python quantum_pipeline.py \
        --file h2_molecule.json \
        --basis sto3g \
        --max-iterations 100 \
        --optimizer COBYLA \
        --report
    ```

=== "Python API"
    ```python
    from quantum_pipeline.runners.vqe_runner import VQERunner

    # Create runner
    runner = VQERunner(
        filepath='h2_molecule.json',
        basis_set='sto3g',
        max_iterations=100,
        optimizer='COBYLA',
        ansatz_reps=2
    )

    # Run simulation
    runner.run()
    ```

=== "Docker"
    ```bash
    docker run --rm -v $(pwd):/data \
        straightchlorine/quantum-pipeline:latest \
        --file /data/h2_molecule.json \
        --basis sto3g \
        --max-iterations 100 \
        --optimizer COBYLA
    ```

---

### Step 3: View Results

The simulation outputs:

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

!!! info "Expected Result"
    For H₂ with STO-3G basis, the ground state energy should be approximately **-1.137 Hartree**, which is within chemical accuracy of experimental values.

<figure>
  <img src="https://qp-docs.codextechnologies.org/mkdocs/convergence_HH.png"
       alt="Energy convergence plot for H₂ molecule VQE simulation">
  <figcaption>Figure 1. Energy convergence during VQE optimization of the H₂ molecule with the STO-3G basis set.</figcaption>
</figure>

---

## Understanding the Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--file` | Path to molecule data file | `h2_molecule.json` |
| `--basis` | Basis set for simulation | `sto3g`, `6-31g`, `cc-pvdz` |
| `--max-iterations` | Maximum VQE iterations | `100` |
| `--optimizer` | Optimization algorithm | `COBYLA`, `L-BFGS-B`, `SLSQP` |
| `--report` | Generate PDF report | Flag (no value) |

See [Configuration Guide](../usage/configuration.md) for all available parameters.

---

## Running Multiple Molecules

Create a file with multiple molecules:

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
        "symbols": ["O", "H", "H"],
        "coords": [[0.0, 0.0, 0.0], [0.0, 0.757, 0.586], [0.0, -0.757, 0.586]],
        "multiplicity": 1,
        "charge": 0,
        "units": "angstrom"
    }
]
```

Run the simulation:

```bash
python quantum_pipeline.py \
    --file molecules.json \
    --basis sto3g \
    --max-iterations 150 \
    --optimizer L-BFGS-B
```

---

## Convergence-Based Optimization

Instead of fixed iterations, use convergence threshold:

```bash
python quantum_pipeline.py \
    --file h2_molecule.json \
    --basis sto3g \
    --convergence --threshold 1e-6 \
    --optimizer L-BFGS-B
```

!!! warning "Mutually Exclusive"
    Do not use both `--max-iterations` and `--convergence` together.

---

## Enabling GPU Acceleration

If you have an NVIDIA GPU:

```bash
python quantum_pipeline.py \
    --file h2_molecule.json \
    --basis sto3g \
    --max-iterations 100 \
    --gpu \
    --simulation-method statevector
```

Or with Docker:

```bash
docker run --rm --gpus all \
    straightchlorine/quantum-pipeline:latest-gpu \
    --file /app/data/h2_molecule.json \
    --gpu \
    --max-iterations 100
```

See [GPU Acceleration Guide](../deployment/gpu-acceleration.md) for setup.

---

## Streaming Results to Kafka

Enable real-time data streaming:

```bash
python quantum_pipeline.py \
    --file molecules.json \
    --basis sto3g \
    --max-iterations 100 \
    --kafka
```

!!! info "Kafka Required"
    This requires Kafka and Schema Registry to be running. See [Kafka Streaming Guide](../data-platform/kafka-streaming.md).

---

## Generating Reports

Generate a PDF report with visualizations:

```bash
python quantum_pipeline.py \
    --file h2_molecule.json \
    --basis sto3g \
    --max-iterations 100 \
    --report
```

The report includes:

- Molecular structure visualization
- Energy convergence plot
- Hamiltonian coefficients
- Optimal parameters
- Timing breakdown

---

## Example: Advanced Configuration

A more advanced example with custom parameters:

```bash
python quantum_pipeline.py \
    --file molecules.json \
    --basis cc-pvdz \
    --max-iterations 200 \
    --optimizer L-BFGS-B \
    --ansatz-reps 5 \
    --shots 2048 \
    --optimization-level 3 \
    --simulation-method statevector \
    --report \
    --kafka \
    --log-level DEBUG
```

This configuration:

- Uses a larger basis set (`cc-pvdz`) for higher accuracy
- Allows more iterations for convergence
- Uses gradient-based optimizer (`L-BFGS-B`)
- Increases ansatz depth (`ansatz-reps=5`)
- Enables both reporting and Kafka streaming
- Sets verbose logging

---

## What's Next?

Now that you've run your first simulation:

1. **[Explore Configuration Options](../usage/configuration.md)** - Learn about all parameters
2. **[Choose the Right Optimizer](../usage/optimizers.md)** - 16 optimizers available
3. **[Understand System Architecture](../architecture/index.md)** - How it all works
4. **[Deploy Full Platform](../deployment/docker-compose.md)** - Run with data engineering
5. **[Monitor Performance](../monitoring/index.md)** - Track system metrics

---

## Common Issues

??? question "Simulation takes too long"
    - Try using fixed iterations instead of `--convergence`
    - Choose a faster optimizer like `L-BFGS-B`
    - Start with smaller basis sets like `sto3g`

??? question "Memory errors with large molecules"
    - Use `--simulation-method matrix_product_state` for memory efficiency
    - Reduce `--ansatz-reps` to decrease circuit depth
    - Consider using a smaller basis set
    - Enable GPU to offload memory to VRAM (if available)

??? question "Results don't match reference energies"
    - This can also be ansatz initialization issue - currently it's random, which may cause process to get stuck
      in local minima. This issue is something to be addressed in the future.
    - Increase `--max-iterations` (try 200-500)
    - Use tighter convergence threshold: `--convergence --threshold 1e-8`
    - Try different optimizers (L-BFGS-B recommended)
    - Use larger basis sets for better accuracy

See [Troubleshooting Guide](../reference/troubleshooting.md) for more help.
