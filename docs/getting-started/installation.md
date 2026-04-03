# Installation

## Prerequisites

- **Python**: 3.12 (the only supported version)
- **OS**: Linux (tested). macOS and Windows (via WSL2) are untested but may work for CPU-only use.
- **Optional**: NVIDIA GPU with CUDA for GPU acceleration

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

Pre-built images are available on Docker Hub:

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

    docker run --rm --gpus all \
        straightchlorine/quantum-pipeline:gpu \
        --file data/molecules.json \
        --basis sto3g \
        --gpu \
        --max-iterations 100
    ```

GPU images require the NVIDIA Container Toolkit and Docker configured with the nvidia runtime. See the [GPU Acceleration Guide](../deployment/gpu-acceleration.md) for setup.

To build images locally:

```bash
just docker-build cpu    # or: gpu, all
```

## Full platform with Docker Compose

This deploys the VQE runner alongside Kafka, Spark, Airflow, Garage, and
monitoring. The stack is defined in
[`compose/docker-compose.ml.yaml`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/compose/docker-compose.ml.yaml).

The fastest way to get started is the setup script, which generates `.env` with
random secrets, configures Garage (S3 storage), creates buckets, and sets up
access keys - all in one step:

```bash
# git clone https://github.com/straightchlorine/quantum-pipeline.git
git clone https://codeberg.org/piotrkrzysztof/quantum-pipeline.git

cd quantum-pipeline

# Automated first-time setup (generates .env, configures Garage)
just setup
# Or without just: bash scripts/ml-setup.sh

# Start the stack
just up
# Or: docker compose --env-file .env -f compose/docker-compose.ml.yaml up -d

# Stop the stack
just down
```

The
[`ml-setup.sh`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/scripts/ml-setup.sh)
script handles everything you would otherwise have to fill in manually:
Garage RPC and admin secrets, Airflow passwords, Fernet key, JWT secret,
webserver secret key, S3 access keys, and bucket creation. If `.env` already exists, it skips
what is already configured.

If you prefer manual setup (or have existing deployments you'd like to use),
copy `.env.ml.example` to `.env` and fill in the values yourself before
running `just up`.

Services started:

| Service | URL |
|---------|-----|
| Airflow | `http://localhost:8084` |
| Spark Master | `http://localhost:8080` |
| Grafana | `http://localhost:3000` |
| Garage S3 API | `http://localhost:3901` |
| Schema Registry | `http://localhost:8081` |
| MLflow | `http://localhost:5000` |

See [Docker Compose Guide](../deployment/docker-compose.md) for details.

## Verify installation

```bash
quantum-pipeline --file data/molecules.json --basis sto3g --max-iterations 10
```

If this completes without errors, the installation is working.

## Optional dependency groups

Install extra groups with PDM or pip:

| Group | Command | What it adds |
|-------|---------|--------------|
| `dev` | `pdm install -G dev` | pytest, ruff, mypy, debugpy, testcontainers |
| `docs` | `pdm install -G docs` | mkdocs, mkdocs-material, pymdown-extensions |
| `ml` | `pdm install -G ml` | scikit-learn, xgboost, mlflow, jupyterlab, seaborn |

With pip, use `pip install -e ".[dev]"`, `pip install -e ".[docs]"`, or `pip install -e ".[ml]"`.

After installing `docs`, run `mkdocs serve` and open `http://127.0.0.1:8000`.

## Troubleshooting

If you get import errors after installing, make sure you installed in editable mode (`pip install -e .`) or with PDM (`pdm install`). Setting `PYTHONPATH` manually is not necessary when installed properly.

For Docker issues, check that the daemon is running (`docker ps`) and inspect logs with `docker compose logs quantum-pipeline`.

See the [Troubleshooting Guide](../reference/troubleshooting.md) for more.

## Next steps

- [Quick Start](quick-start.md) - run your first VQE simulation
- [Configuration](../usage/configuration.md) - all available parameters
- [Docker Compose](../deployment/docker-compose.md) - deploy the full platform
