---
title: "Docker Basics: CPU and GPU Images"
description: "The CPU, GPU, Spark and Airflow Docker images for Quantum Pipeline, how to build them with just or by hand, and how to run a single simulation container."
---

# Docker Basics

Quantum Pipeline provides Docker images for CPU and GPU workloads, as well
as images for Spark and Airflow services. This page covers the available
images, building from source, and running containers.

## Available Images

The CPU and GPU images are published to Docker Hub under the
[`straightchlorine/quantum-pipeline`](https://hub.docker.com/r/straightchlorine/quantum-pipeline)
repository and to
[GHCR](https://github.com/straightchlorine/quantum-pipeline/pkgs/container/quantum-pipeline)
as `ghcr.io/straightchlorine/quantum-pipeline`, with the same tags.


| Image Tag | Base Image | Purpose |
|---|---|---|
| [`straightchlorine/quantum-pipeline:cpu`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/docker/Dockerfile.cpu) | `python:3.12-slim-trixie` | CPU-only simulations |
| [`straightchlorine/quantum-pipeline:gpu`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/docker/Dockerfile.gpu) | `ubuntu:24.04` (Aer compiled on `nvidia/cuda:12.6.3-devel-ubuntu24.04`) | GPU-accelerated simulations |
| [`quantum-pipeline-spark`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/docker/Dockerfile.spark) | `apache/spark:4.0.2-python3` | Spark master and worker nodes |
| [`airflow` (custom)](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/docker/airflow/Dockerfile) | `apache/airflow:3.1.8` | Airflow API server, scheduler, workers, triggerer |

### CPU Image

The CPU image is a container based on Python 3.12 (Debian Trixie). It installs
the Quantum Pipeline package and its dependencies. The `data/` directory with
sample molecule files is included, so you can run simulations immediately with
`--file data/molecules.json`.

Both images run as root by default but work under any `--user`: the working
directory `/usr/src/quantum_pipeline`, where `gen/` and `run_configs/` are written,
is writable by every user. A bind-mounted `gen/` must be writable by that user too.

The CPU and GPU images install their Python dependencies from hashed requirements
files exported from the PDM lock files, so builds run the exact locked versions.
See the [Docker README](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/docker/README.md)
for more detailed overview of the image builds.

### GPU Image

The GPU image compiles Qiskit Aer from source with the CUDA Thrust backend
(CUDA 12.6.3) and ships it on Ubuntu 24.04: Aer carries its own CUDA runtime
and the NVIDIA Container Toolkit provides the driver, so no CUDA or cuDNN base is
needed. cuStateVec and the `tensor_network` simulation method are not built in.

The CUDA toolkit image is pinned by tag and digest in the GPU Dockerfile. To use a
different CUDA 12 toolkit, change both the tag and the digest on the base image line
and rebuild.

The CUDA toolkit is inside the image, so the CUDA toolkit installed on your machine
does not matter. The host needs only an NVIDIA driver (525.60.13 or newer, the
minimum NVIDIA lists for CUDA 12.x) and the NVIDIA Container Toolkit.

GPUs newer than the compiled architectures run through PTX, which needs a driver
at least as new as the image's CUDA - 560 or newer.

In order to utilise the GPU setup run containers with `--gpus`; without it the
pipeline stops with `No CUDA device available!` and does not fall back to the CPU.

From 2.3.0, the `straightchlorine/quantum-pipeline:gpu` image on Docker Hub is
built for compute capabilities `6.1 7.0 8.0 9.0+PTX`. Code built for X.y runs on
every X.z with z at least y, and `+PTX` lets the driver compile for newer GPUs.
One image therefore covers:

| Architecture | Example GPUs |
|---|---|
| Pascal (6.1) | GTX 1060, GTX 1050 Ti |
| Volta (7.0) | Titan V, V100 |
| Turing (7.5) | RTX 2070, RTX 2080 |
| Ampere (8.0, 8.6) | A100, RTX 3060, RTX 3080 |
| Ada Lovelace (8.9) | RTX 4070, RTX 4090 |
| Hopper (9.0) | H100 |
| Blackwell | via PTX |

Pascal P100 (6.0) and anything older is not covered. You do not need to rebuild
for your GPU. A local build for a single architecture is optional: it is significantly
faster, but only useful on your own machine.

### Spark Image

The Spark image extends the Apache Spark 4.0.2 base by replacing Python 3.10 with
3.12 so it matches the Airflow driver environment. The image
itself contains no application code or JARs - those are resolved at runtime via
`spark.jars.packages` in `spark-defaults.conf` and cached in a named Ivy volume.

### Airflow Image

The Airflow image adds Java 17 (required for Spark job submission), Docker CE CLI
with buildx and compose plugins (for batch generation DAGs that spawn simulation
containers), rclone (for R2 sync), and Python dependencies (`apache-airflow-providers-apache-spark`,
`pyspark`) to the Apache Airflow 3.1.8 base image.

The `DOCKER_GID` build argument sets the GID of the `docker` group inside the
container so the Airflow user can access the host Docker socket. Check your host
GID with `stat -c '%g' /var/run/docker.sock` and pass it at build time if it
differs from the default (`970`).

## Building from Source

### Using the justfile (recommended)

The simplest way to build images:

```bash
# Build CPU image
just docker-build cpu

# Build GPU image (default CUDA_ARCH covers Pascal to Hopper)
just docker-build gpu

# Faster local build for one GPU (use its compute capability)
CUDA_ARCH=6.1 just docker-build gpu

# Build both CPU and GPU images
just docker-build all
```

### Manual builds

Each image can be built directly with Docker:

```bash
# CPU
docker build -t quantum-pipeline:cpu -f docker/Dockerfile.cpu .

# GPU (default: Pascal to Hopper)
docker build -t quantum-pipeline:gpu -f docker/Dockerfile.gpu .

# GPU for a single architecture (faster, local use)
docker build -t quantum-pipeline:gpu -f docker/Dockerfile.gpu \
  --build-arg CUDA_ARCH="6.1" .

# Airflow (with custom Docker GID)
docker build -t quantum-pipeline-airflow -f docker/airflow/Dockerfile \
  --build-arg DOCKER_GID="$(stat -c '%g' /var/run/docker.sock)" \
  docker/airflow/
```

!!! warning "GPU Build Time"
    Building the GPU image compiles Qiskit Aer from source with CUDA Thrust backend
    support. This takes a while. A single-arch `CUDA_ARCH` build is significantly 
    faster; find your card's value with `nvidia-smi --query-gpu=compute_cap --format=csv`.

## Running Containers

### CPU Simulation

Run a basic CPU simulation with:

```bash
docker run --rm \
  -v $(pwd)/data:/usr/src/quantum_pipeline/data \
  -v $(pwd)/gen:/usr/src/quantum_pipeline/gen \
  quantum-pipeline:cpu \
  --file ./data/molecules.json \
  --simulation-method statevector \
  --convergence
```

### GPU Simulation

For GPU-accelerated simulation, pass the `--gpus` flag and add the `--gpu` argument:

```bash
docker run --rm --gpus all \
  -v $(pwd)/data:/usr/src/quantum_pipeline/data \
  -v $(pwd)/gen:/usr/src/quantum_pipeline/gen \
  quantum-pipeline:gpu \
  --file ./data/molecules.json \
  --gpu \
  --simulation-method statevector \
  --convergence
```

### Connecting to Kafka

To send results to a Kafka broker, add the `--kafka` flag and set the
`KAFKA_SERVERS` environment variable:

```bash
docker run --rm --gpus all \
  --network quantum-ml-network \
  -e KAFKA_SERVERS=kafka:9092 \
  -v $(pwd)/data:/usr/src/quantum_pipeline/data \
  quantum-pipeline:gpu \
  --file ./data/molecules.json \
  --gpu \
  --kafka \
  --simulation-method statevector \
  --convergence
```

### Common Options

| Flag | Description |
|---|---|
| `--file <path>` | Path to molecule definition file (JSON) |
| `--gpu` | Enable GPU acceleration |
| `--kafka` | Enable Kafka output |
| `--topic <name>` | Kafka topic name for results |
| `--simulation-method <method>` | Simulation method, e.g. `statevector` or `automatic` (full list in [Environment Variables](environment-variables.md); `tensor_network` requires `--gpu`) |
| `--max-iterations <n>` | Maximum VQE optimizer iterations |
| `--convergence` | Enable convergence threshold |
| `--threshold <value>` | Convergence threshold value (default: `1e-6`) |
| `--optimizer <name>` | Optimizer algorithm (e.g., `L-BFGS-B`, `COBYLA`) |
| `--basis <set>` | Basis set (e.g., `sto3g`, `cc-pvdz`) |
| `--ansatz <type>` | Ansatz type (`EfficientSU2`, `RealAmplitudes`, `ExcitationPreserving`) |
| `--seed <n>` | Random seed for reproducible parameter initialization |
| `--init-strategy <strategy>` | Parameter initialization (`random`, `hf`); use `hf` with `--ansatz-reps 3` (see [`--init-strategy`](../usage/configuration.md#-init-strategy)) |
| `--log-level <level>` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `--report` | Generate a PDF report after simulation |
| `--enable-performance-monitoring` | Enable resource monitoring |
