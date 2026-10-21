---
title: "GPU Acceleration with Qiskit Aer CUDA"
description: "Set up NVIDIA GPU simulation with Qiskit Aer CUDA: prerequisites, the CUDA_ARCH build argument, Docker GPU access, and the speedups measured on consumer cards."
---

# GPU Acceleration

Quantum Pipeline uses NVIDIA GPUs through Qiskit Aer's CUDA backend to
accelerate quantum circuit simulation. This page covers the setup,
configuration, and performance characteristics of GPU-accelerated execution.

## Prerequisites

GPU acceleration requires an NVIDIA GPU (compute capability 6.1 or newer), NVIDIA
drivers 525 or newer (CUDA 12), NVIDIA Container Toolkit (1.14+), and Docker Engine (24.0+).
The GPU image is built on CUDA 12.6.3. For detailed installation steps, see the
[NVIDIA Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

!!! note "cuQuantum and Volta+ Features"
    NVIDIA cuQuantum (cuStateVec, cuTensorNet) provides additional acceleration
    for quantum simulations but requires Volta architecture or newer (compute
    capability 7.0+) and CUDA 11.2+. The `blocking_enable` option shares this
    requirement. The thesis and current experiments did not use these features
    because the test hardware is Pascal architecture. The image currently does 
    not build cuStateVec or the `tensor_network` method in.

## GPU Configuration

GPU behavior is controlled through the backend defaults in
[`defaults.py`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/configs/defaults.py).
The top-level `gpu` flag (default `False`) turns acceleration on, and the `gpu_opts`
dictionary holds the options passed to Qiskit Aer:

| Option | Default | Description |
|---|---|---|
| `device` | `GPU` | Target device (`GPU` or `CPU`). |
| `cuStateVec_enable` | `False` | NVIDIA cuStateVec (Volta+ only). |
| `blocking_enable` | `False` | Reduce synchronization overhead (Volta+ only). |
| `batched_shots_gpu` | `True` | Shot parallelization for better GPU utilization. |
| `shot_branching_enable` | `True` | Circuit branching optimization. |
| `max_memory_mb` | `5500` | Memory cap in MB. The default leaves a buffer on a 6 GB GTX 1060; adjust it for your GPU. |

When GPU mode is enabled, `gpu_opts` is unpacked into Qiskit Aer's `AerSimulator`
alongside the simulation method and any noise model
([`solver.py`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/solvers/solver.py)).

### Command-Line GPU Activation

Enable GPU acceleration from the command line with the `--gpu` flag:

```bash
quantum-pipeline \
  --file ./data/molecules.json \
  --gpu \
  --simulation-method statevector \
  --max-iterations 150
```

With `--gpu` the run uses the Aer exact path; the analytic path is CPU-only.

## CUDA_ARCH Build Argument

The GPU Docker image compiles Qiskit Aer from source. By default `CUDA_ARCH` is
`6.1 7.0 8.0 9.0+PTX`, which covers all of these (from 2.3.0 the published image
does too):

| Compute capability | Architecture | Example GPUs |
|---|---|---|
| `6.1` | Pascal | GTX 1060, GTX 1050 Ti |
| `7.0` / `7.5` | Volta / Turing | V100, RTX 2070, RTX 2080 |
| `8.0` / `8.6` | Ampere | A100, RTX 3060, RTX 3080 |
| `8.9` | Ada Lovelace | RTX 4070, RTX 4090 |
| `9.0` and newer | Hopper, Blackwell (via PTX) | H100 |

Pascal P100 (6.0) and older are not covered. For a faster local build (about 4x),
set `CUDA_ARCH` to your card's compute capability, from
`nvidia-smi --query-gpu=compute_cap --format=csv`:

```bash
# Pascal (GTX 10xx)
CUDA_ARCH=6.1 just docker-build gpu

# Turing (RTX 20xx)
CUDA_ARCH=7.5 just docker-build gpu

# Ada Lovelace (RTX 40xx)
CUDA_ARCH=8.9 just docker-build gpu
```

You can also set `CUDA_ARCH` in your `.env` file so it persists across builds.

## Docker GPU Setup

### Single Container

Run a GPU container with `--gpus all`:

```bash
docker run --rm --gpus all \
  quantum-pipeline:gpu \
  --file ./data/molecules.json \
  --gpu \
  --simulation-method statevector
```

To restrict to a specific GPU:

```bash
docker run --rm --gpus '"device=0"' \
  quantum-pipeline:gpu \
  --file ./data/molecules.json \
  --gpu
```

### Docker Compose

In Docker Compose, GPU access is configured through the `deploy.resources.reservations`
section:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ['0']   # Specific GPU by index
          capabilities: [gpu]
```

Use `count: all` instead of `device_ids` to grant access to all available GPUs:

```yaml
devices:
  - driver: nvidia
    count: all
    capabilities: [gpu]
```

The batch simulation containers in `compose/docker-compose.ml.yaml` use the `runtime: nvidia`
shorthand with `NVIDIA_VISIBLE_DEVICES` to assign specific GPUs to each container.

## Simulation Methods for GPU

Not all Qiskit Aer simulation methods support GPU acceleration:

| Method | GPU Support | Description |
|---|---|---|
| `statevector` | Yes | Full state vector simulation. Best for small-to-medium circuits. |
| `density_matrix` | Yes | Density matrix simulation. Supports noise models. |
| `tensor_network` | Yes (cuTensorNet) | Tensor network contraction. Requires cuQuantum. |

For the thesis experiments, `statevector` was used exclusively as it provides the
most direct benefit from GPU acceleration for VQE workloads. For the full comparison
table, GPU backend details, and memory scaling characteristics, see
[Simulation Methods](../usage/simulation-methods.md#gpu-compatible-methods).

## Performance Benchmarks

The thesis experiments measured GPU acceleration across six molecules with two
NVIDIA GPUs (GTX 1060, GTX 1050 Ti) against an Intel i5-8500 CPU baseline.
The key results: 1.74-1.81x speedup on STO-3G, 3.53-4.08x on cc-pVDZ. Speedup
scales with qubit count and basis set size, with the crossover from
overhead-dominated to acceleration-dominated between 12 and 14 qubits
(LiH to BeH2 in the thesis runs).

These figures compare GPU and CPU on the Aer path. An exact run (no `--shots`)
with a gradient optimizer that is local, noiseless, without `--gpu` and has at
most 16 qubits takes the analytic path instead: energies and gradients are
computed directly on the statevector. At these sizes that CPU run is faster than
a `--gpu` run, and adding `--gpu` turns the analytic path off. See
[Analytic Fast-Path Comparison](../scientific/benchmarking.md#analytic-fast-path-comparison).

For the full benchmark data, per-molecule breakdown, convergence analysis, and
basis set comparisons, see
[Benchmarking Results](../scientific/benchmarking.md#cpu-vs-gpu-performance).

## Troubleshooting

For general GPU and Docker troubleshooting (driver installation, container toolkit
configuration, runtime issues), see the
[NVIDIA Container Toolkit troubleshooting guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/troubleshooting.html).

**Out of Memory:** The state vector size doubles with each additional qubit. For a
4 GB GPU, the practical limit is approximately 28 qubits with `statevector`
simulation.

**Qiskit Aer GPU Build:** The default image covers Pascal 6.1 and newer. Only a custom
single-arch build for another card, or a GPU older than 6.1, fails at runtime with
`CUDA error: no kernel image is available`. See the
[CUDA_ARCH table](#cuda_arch-build-argument) above.

## References

- [NVIDIA Container Toolkit Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [NVIDIA cuQuantum Documentation](https://docs.nvidia.com/cuda/cuquantum/)
- [Qiskit Aer GPU Simulation](https://qiskit.github.io/qiskit-aer/howtos/running_gpu.html)
- Bayraktar, H. et al. *cuQuantum SDK: A High-Performance Library for Accelerating Quantum Science.* arXiv:2308.01999 (2023)
