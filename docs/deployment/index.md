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
and run simulations directly. This does however require the user to set up other services
as well as ensure connectivity.

A **single Docker container** provides an isolated environment with all Python
dependencies pre-installed. The GPU variant includes CUDA libraries and a custom-built
Qiskit Aer with GPU support. This is suitable for running simulations without the
full data pipeline.

The **Docker Compose deployment** brings up the complete platform: Quantum Pipeline
containers (CPU and GPU), Apache Kafka with Schema Registry, Spark cluster, Airflow
orchestrator, MinIO object storage, and monitoring services. This is the configuration
used for thesis experiments.

## Resource Allocation

The thesis experiments used a system with an Intel Core i5-8500 (6 cores), 56 GB RAM,
and two NVIDIA GPUs (GTX 1060 6GB + GTX 1050 Ti 4GB).

It was stable and functioned well for extended periods of time. I'd imagine minimal
requirements are much lower - naturally, while accepting longer times of iteration.

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
