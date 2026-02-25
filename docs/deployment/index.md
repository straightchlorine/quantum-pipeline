# Deployment

Quantum Pipeline supports multiple deployment strategies, ranging from a lightweight
PyPI installation for development to a full Docker Compose deployment that includes
the entire data platform. This section covers each approach in detail, along with
GPU acceleration configuration and environment variable reference.

## Deployment Options

| Deployment | Services | Use Case |
|---|---|---|
| **PyPI Package** | Quantum Pipeline only | Development, quick experiments |
| **Docker (single container)** | Quantum Pipeline (CPU or GPU) | Isolated simulations |
| **Docker Compose** | Full platform (all services) | Production, thesis experiments |

The **PyPI package** is the simplest option. Install with `pip install quantum-pipeline`
and run simulations directly. This approach does not include infrastructure services
such as Kafka, Spark, or Airflow.

A **single Docker container** provides an isolated environment with all Python
dependencies pre-installed. The GPU variant includes CUDA libraries and a custom-built
Qiskit Aer with GPU support. This is suitable for running simulations without the
full data pipeline.

The **Docker Compose deployment** brings up the complete platform: Quantum Pipeline
containers (CPU and GPU), Apache Kafka with Schema Registry, Spark cluster, Airflow
orchestrator, MinIO object storage, and monitoring services. This is the configuration
used for thesis experiments.

## Resource Requirements

| Component | CPU | RAM | GPU | Storage |
|---|---|---|---|---|
| CPU Pipeline | 2 cores | 10 GB | -- | 2 GB |
| GPU Pipeline | 2 cores | 10 GB | NVIDIA (4+ GB VRAM) | 8 GB |
| Infrastructure | 1 core | 8 GB | -- | 10 GB |
| Monitoring | 0.5 cores | 2 GB | -- | 1 GB |
| **Full Platform** | **6 cores** | **30 GB** | **1-2 GPUs** | **20 GB** |

The thesis experiments used a system with an Intel Core i5-8500 (6 cores), 56 GB RAM,
and two NVIDIA GPUs (GTX 1060 6GB + GTX 1050 Ti 4GB).

!!! note "Minimum Requirements"
    For development and testing, 4 CPU cores and 16 GB RAM are sufficient when running
    a subset of services. GPU acceleration requires an NVIDIA GPU with CUDA compute
    capability 6.0 or higher.

## Guides

<div class="grid cards" markdown>

-   **Docker Basics**

    ---

    Container images, building from source, and running individual containers.

    [:octicons-arrow-right-24: Docker Basics](docker-basics.md)

-   **Docker Compose Deployment**

    ---

    Full platform deployment with all services, networking, and health checks.

    [:octicons-arrow-right-24: Docker Compose](docker-compose.md)

-   **GPU Acceleration**

    ---

    NVIDIA driver setup, CUDA configuration, and performance benchmarks.

    [:octicons-arrow-right-24: GPU Acceleration](gpu-acceleration.md)

-   **Environment Variables**

    ---

    Complete reference for all configuration variables used across services.

    [:octicons-arrow-right-24: Environment Variables](environment-variables.md)

</div>
