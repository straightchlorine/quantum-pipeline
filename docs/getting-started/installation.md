---
title: "Installation: PyPI, Source and Docker"
description: "Install Quantum Pipeline from PyPI, from source with PDM or pip, or in Docker, and check the Python 3.10 to 3.14 and optional NVIDIA GPU requirements before running."
---

# Installation

## Prerequisites

- **Python**: 3.10 to 3.14. Installing from source with PDM, and the Docker images, use Python 3.12 or newer; on 3.10 and 3.11 install with pip. The ml extra needs 3.11 or newer.
- **OS**: Linux (tested). macOS and Windows (WSL2) are untested; may work for CPU-only use.
- **Optional**: NVIDIA GPU with CUDA for GPU acceleration

## From PyPI

```bash
pip install quantum-pipeline
```

## From source

Clone the repository and install:

```bash
# codeberg mirror
git clone https://codeberg.org/piotrkrzysztof/quantum-pipeline.git

# or github
git clone https://github.com/straightchlorine/quantum-pipeline.git

cd quantum-pipeline

# Install with PDM
pdm install

# Or with pip
pip install -e .
```

## Docker

Pre-built images are available on Docker Hub. A run without `--shots` is exact:
expectation values come from the statevector, with no shot sampling (see [`--exact`](../usage/configuration.md#-exact)).

=== "CPU"
    ```bash
    docker pull straightchlorine/quantum-pipeline:cpu

    docker run --rm straightchlorine/quantum-pipeline:cpu \
        --file data/molecules.json \
        --basis sto3g \
        --max-iterations 100
    ```

=== "GPU"
    ```bash
    docker pull straightchlorine/quantum-pipeline:gpu

    docker run --rm --gpus all straightchlorine/quantum-pipeline:gpu \
        --file data/molecules.json \
        --basis sto3g \
        --gpu \
        --max-iterations 100
    ```

GPU images require the NVIDIA Container Toolkit and Docker configured with the nvidia runtime. See the [GPU Acceleration Guide](../deployment/gpu-acceleration.md) for setup.

The runs above leave nothing behind once the container exits. To keep generated
reports and plots, pass `--report` and mount the output directory. The container
writes artifacts to `gen/` under its working directory (`/usr/src/quantum_pipeline`):

```bash
docker run --rm \
    -v "$(pwd)/gen:/usr/src/quantum_pipeline/gen" \
    straightchlorine/quantum-pipeline:cpu \
    --file data/molecules.json \
    --molecule-index 0 \
    --basis sto3g \
    --max-iterations 100 \
    --report
```

To build images locally:

```bash
just docker-build cpu    # or: gpu, all

# in case of gpu:
# optional: set CUDA_ARCH to your GPU for a faster local build (default covers Pascal to Hopper)
CUDA_ARCH=7.5 just docker-build gpu   # Turing (RTX 20xx)
```

## Full platform with Docker Compose

This deploys the simulation module alongside Kafka, Spark, Airflow, Garage,
monitoring exporters and MLflow. The stack is defined in
[`compose/docker-compose.ml.yaml`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/compose/docker-compose.ml.yaml).

The fastest way to get started is the setup script. It generates `.env` with
random secrets, renders the Spark configuration, configures Garage (S3 storage),
creates buckets, and sets up access keys:

```bash
# git clone https://github.com/straightchlorine/quantum-pipeline.git
git clone https://codeberg.org/piotrkrzysztof/quantum-pipeline.git

cd quantum-pipeline

# Automated first-time setup (generates .env, configures Garage)
just setup
# bash scripts/ml-setup.sh

# Start the stack
just up
# docker compose --env-file .env -f compose/docker-compose.ml.yaml up -d

# Stop the stack
just down
```

If `.env` already exists, it skips what is already configured. You can also copy
`.env.ml.example` to `.env` and fill in the values yourself, but `just setup` (or
`bash scripts/ml-setup.sh`) is still required before `just up`: it renders the Spark
configuration the stack mounts. To use an external S3 service instead of the compose
Garage, set `S3_ENDPOINT` and its credentials in `.env` before running it; see
[External S3 storage](../deployment/docker-compose.md#external-s3-storage).

Services started:

| Service | URL |
|---------|-----|
| Airflow | `http://localhost:8084` |
| Spark Master | `http://localhost:8080` |
| Garage S3 API | `http://localhost:3901` |
| Schema Registry | `http://localhost:8081` |
| MLflow | `http://localhost:5000` |

See [Docker Compose Guide](../deployment/docker-compose.md) for details.

## Verify installation

```bash
quantum-pipeline --file data/molecules.json --molecule-index 0 --basis sto3g --max-iterations 10
```

If this completes without errors, the installation is working.

## Optional dependency groups

Install extra groups with PDM or pip:

| Group | Command | What it adds |
|-------|---------|--------------|
| `dev` | `pdm install -G dev` | pytest, ruff, mypy, debugpy, testcontainers, pyspark |
| `docs` | `pdm install -G docs` | mkdocs, mkdocs-material, pymdown-extensions |
| `ml` | `pdm install -G ml` | scikit-learn, xgboost, mlflow, jupyterlab, seaborn, pyspark |

With pip, use `pip install -e ".[dev]"`, `pip install -e ".[docs]"`, or `pip install -e ".[ml]"`.

After installing `docs`, run `mkdocs serve` and open `http://127.0.0.1:8000`.

## Troubleshooting

See the [Troubleshooting Guide](../reference/troubleshooting.md) for more.

## Next steps

- [Quick Start](quick-start.md) - run your first VQE simulation
- [Configuration](../usage/configuration.md) - all available parameters
- [Docker Compose](../deployment/docker-compose.md) - deploy the full platform
