# Installation

This guide covers different installation methods for the Quantum Pipeline framework.

---

## Prerequisites

- **Python**: 3.10, 3.11, or 3.12
- **Operating System**: Linux (tested), macOS (untested, CPU only in theory), Windows (untested, via WSL2 in theory)
- **Memory**: Minimum 8 GB RAM (16 GB recommended)
- **Optional**: NVIDIA GPU with CUDA support for GPU acceleration

---

## Installation Methods

### Method 1: PyPI Package

Install the latest stable release from PyPI:

```bash
pip install quantum-pipeline
```

Verify the installation:

```bash
python -c "import quantum_pipeline; print(quantum_pipeline.__version__)"
```

---

### Method 2: From Source

Clone the repository and install in development mode:

```bash
# Clone the repository
git clone https://github.com/straightchlorine/quantum-pipeline.git
cd quantum-pipeline

# Install with PDM
pdm install

# Or install with pip
pip install -e .
```

---

### Method 3: Docker

Pull pre-built images from Docker Hub:

=== "CPU Version"
    ```bash
    # Latest stable release
    docker pull straightchlorine/quantum-pipeline:latest

    # Run a simple simulation
    docker run --rm straightchlorine/quantum-pipeline:latest \
        --file /app/data/molecules.json \
        --basis sto3g \
        --max-iterations 100
    ```

=== "GPU Version"
    ```bash
    # GPU-enabled version (requires NVIDIA Docker)
    docker pull straightchlorine/quantum-pipeline:latest-gpu

    # Run with GPU acceleration
    docker run --rm --gpus all \
        straightchlorine/quantum-pipeline:latest-gpu \
        --file /app/data/molecules.json \
        --basis sto3g \
        --gpu \
        --max-iterations 100
    ```

!!! warning "GPU Prerequisites"
    For GPU support, you need:

    - NVIDIA GPU with CUDA support
    - NVIDIA Container Toolkit installed
    - Docker configured with nvidia runtime

    See [GPU Acceleration Guide](../deployment/gpu-acceleration.md) for setup instructions.

---

### Method 4: Full Platform with Docker Compose

Deploy the complete data engineering platform:

```bash
# Clone repository
git clone https://github.com/straightchlorine/quantum-pipeline.git
cd quantum-pipeline

# Copy environment file
cp .env.thesis.example .env
# Edit .env with your configuration

# Start all services
docker compose up -d
```

This launches:

- Quantum Pipeline (CPU/GPU)
- Apache Kafka with Schema Registry
- Apache Spark cluster (master + workers)
- Apache Airflow (webserver, scheduler, triggerer)
- MinIO object storage
- PostgreSQL database
- Prometheus & Grafana monitoring (optional)

!!! info "Service URLs"
    After starting, access web interfaces at:

    - **Airflow**: http://localhost:8084
    - **Spark Master**: http://localhost:8080
    - **MinIO Console**: http://localhost:9001
    - **Grafana**: http://localhost:3000 (if monitoring enabled)

See [Docker Compose Guide](../deployment/docker-compose.md) for detailed configuration.

---

## Verify Installation

After installation, verify everything works:

```python
from quantum_pipeline.runners.vqe_runner import VQERunner

# Create runner instance
runner = VQERunner(
    filepath='data/molecules.json',
    basis_set='sto3g',
    max_iterations=10,
    optimizer='COBYLA'
)

print("Quantum Pipeline installed successfully.")
```

---

## Optional Dependencies

### Development Tools

For contributing or development:

```bash
pdm install -G dev

# Or with pip
pip install -e ".[dev]"
```

This includes:

- `pytest` - Testing framework
- `debugpy` - Python debugger
- `ruff` - Fast Python linter

### Documentation Tools

To build documentation locally:

```bash
pdm install -G docs

# Or with pip
pip install -e ".[docs]"

mkdocs serve
```

View documentation at http://127.0.0.1:8000

### Airflow Integration

For running data processing workflows:

```bash
pdm install -G airflow

# Or with pip
pip install -e ".[airflow]"
```

---

## Platform-Specific Notes

### Linux

Quantum Pipeline is primarily developed and tested on Linux. All features are fully supported.

### macOS

**Not tested.** Basic functionality should work in theory, but:

- GPU acceleration is not available (CUDA requires Linux/Windows)

### Windows

**Not tested.** Support via WSL2 should work in theory:

1. Install WSL2 with Ubuntu
2. Install Docker Desktop with WSL2 backend
3. Follow Linux installation instructions

---

## Troubleshooting

### Import Errors

If you encounter import errors:

```bash
# Ensure quantum_pipeline is in your Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/quantum-pipeline"

# Or install in development mode
pip install -e .
```

### Docker Issues

If Docker containers fail to start:

```bash
# Check Docker daemon is running
docker ps

# View logs for specific service
docker compose logs quantum-pipeline

# Restart all services
docker compose restart
```

See [Troubleshooting Guide](../reference/troubleshooting.md) for more solutions.

---

## Next Steps

- **[Quick Start Guide](quick-start.md)** - Run your first VQE simulation
- **[Configuration Options](../usage/configuration.md)** - Customize your setup
- **[Docker Deployment](../deployment/docker-compose.md)** - Deploy full platform
