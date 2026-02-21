# Quantum Pipeline

<p align="center">
  <a href="https://pypi.org/project/quantum-pipeline/">
    <img src="https://badge.fury.io/py/quantum-pipeline.svg" alt="PyPI version">
  </a>
  <a href="https://pypi.org/project/quantum-pipeline/">
    <img src="https://img.shields.io/pypi/dm/quantum-pipeline" alt="PyPI Downloads per Month">
  </a>
  <a href="https://pepy.tech/project/quantum-pipeline">
    <img src="https://static.pepy.tech/badge/quantum-pipeline" alt="PyPI Total Downloads">
  </a>
  <a href="https://pypi.org/project/quantum-pipeline/">
    <img src="https://img.shields.io/pypi/pyversions/quantum-pipeline.svg" alt="Python Versions">
  </a>
  <br>
  <a href="https://hub.docker.com/r/straightchlorine/quantum-pipeline">
    <img src="https://img.shields.io/docker/pulls/straightchlorine/quantum-pipeline.svg" alt="Docker Pulls">
  </a>
  <a href="https://hub.docker.com/r/straightchlorine/quantum-pipeline">
    <img src="https://img.shields.io/docker/image-size/straightchlorine/quantum-pipeline/latest" alt="Docker Image Size">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
  </a>
  <a href="https://github.com/straightchlorine/quantum-pipeline">
    <img src="https://img.shields.io/github/stars/straightchlorine/quantum-pipeline.svg" alt="GitHub Stars">
  </a>
</p>

## Overview

The Quantum Pipeline project is an **extensible framework** designed for exploring
quantum algorithms. Currently, only **Variational Quantum Eigensolver (VQE)** is
implemented. It combines quantum and classical computing to estimate the
ground-state energy of molecular systems with a comprehensive data engineering pipeline.

The framework provides modules to handle algorithm orchestration, parametrization,
monitoring, and data visualization. Data can be streamed via **Apache Kafka** for
real-time processing, transformed into ML features using **Apache Spark**,
and stored in **Apache Iceberg** tables for scalable analytics.

---

## Key Features

### Core Quantum Computing

- **Molecule Loading** - Load and validate molecular data from files
- **Hamiltonian Preparation** - Generate second-quantized Hamiltonians for molecular systems
- **Quantum Circuit Construction** - Create parameterized ansatz circuits with customizable repetitions
- **VQE Execution** - Solve Hamiltonians using the VQE algorithm with support for various optimizers
- **Advanced Backend Options** - Customize simulation parameters such as qubit count, shot count, and optimization levels

### Data Engineering Pipeline

- **Real-time Streaming** - Stream simulation results to Apache Kafka with Avro serialization
- **ML Feature Engineering** - Transform quantum experiment data into ML features using Apache Spark
- **Data Lake Storage** - Store processed data in Apache Iceberg tables with versioning and time-travel
- **Object Storage** - Persist data using MinIO S3-compatible storage with automated backup
- **Workflow Orchestration** - Automate data processing workflows using Apache Airflow

### Analytics and Visualization

- **Visualization Tools** - Plot molecular structures, energy convergence, and operator coefficients
- **Report Generation** - Automatically generate detailed reports for each processed molecule
- **Scientific Reference Validation** - Compare VQE results against experimentally verified ground state energies
- **Feature Tables** - Access structured data through 9 specialized ML feature tables
- **Processing Metadata** - Track data lineage and processing history

### Production Deployment

- **Containerized Execution** - Deploy as multi-service Docker containers with GPU support
- **CI/CD Pipeline** - Automated testing, building, and publishing of Docker images
- **Scalable Architecture** - Distributed processing with Spark clusters and horizontal scaling
- **Security** - Comprehensive secrets management and secure communication between services

---

## Quick Links

<div class="grid cards" markdown>

-   **Getting Started**

    ---

    Install Quantum Pipeline and run your first VQE simulation in minutes

    [Installation Guide →](getting-started/installation.md)

-   **Configuration**

    ---

    Learn about optimizers, simulation methods, and parameter tuning

    [Usage Guide →](usage/index.md)

-   **Architecture**

    ---

    Understand the system design, data flow, and Avro serialization

    [Architecture Docs →](architecture/index.md)

-   **Deployment**

    ---

    Deploy with Docker, enable GPU acceleration, configure environments

    [Deployment Guide →](deployment/index.md)

</div>

---

## System Architecture

<figure>
  <img src="https://qp-docs.codextechnologies.org/mkdocs/gpu_quantum_pipeline_service.png"
       alt="GPU-accelerated quantum pipeline service architecture">
  <figcaption>Figure 1. Overview of the GPU-accelerated quantum pipeline service architecture.</figcaption>
</figure>

### Thesis Experiment Architecture

The following diagram presents the architecture as deployed for the engineering thesis experiments.
Kafka Connect writes raw Avro files to MinIO, and Spark (triggered by Airflow) reads from MinIO to
produce ML features, while utilising Iceberg for incremental processing.

```mermaid
graph TB
    subgraph "Quantum Simulation"
        QP[Quantum Pipeline<br/>VQE Runner]
    end

    subgraph "Streaming Layer"
        KAFKA[Apache Kafka<br/>Message Broker]
        SR[Schema Registry<br/>Avro Schemas]
        KC[Kafka Connect<br/>S3 Sink]
    end

    subgraph "Storage Layer"
        MINIO[MinIO<br/>Object Storage]
        ICEBERG[Apache Iceberg<br/>Feature Tables]
    end

    subgraph "Processing Layer"
        AIRFLOW[Apache Airflow<br/>Orchestration]
        SPARK[Apache Spark<br/>Feature Engineering]
    end

    QP -->|Publish Results| KAFKA
    KAFKA <-->|Schema Validation| SR
    KAFKA -->|Consume Topics| KC
    KC -->|Write Avro Files| MINIO
    AIRFLOW -->|Trigger| SPARK
    SPARK -->|Read Raw Data| MINIO
    SPARK -->|Write Features| ICEBERG
    ICEBERG -->|Store Parquet| MINIO

    style QP fill:#c5cae9,color:#1a237e
    style KAFKA fill:#ffe082,color:#000
    style SR fill:#ffe082,color:#000
    style KC fill:#ffe082,color:#000
    style SPARK fill:#a5d6a7,color:#1b5e20
    style AIRFLOW fill:#90caf9,color:#0d47a1
    style ICEBERG fill:#b39ddb,color:#311b92
    style MINIO fill:#b39ddb,color:#311b92
```

### General Architecture

The project is configurable -- Kafka can stream directly to Spark consumers, bypassing the MinIO
intermediate storage. This is useful for real-time processing scenarios.

```mermaid
graph TB
    subgraph "Quantum Simulation"
        QP2[Quantum Pipeline<br/>VQE Runner]
    end

    subgraph "Streaming Layer"
        KAFKA2[Apache Kafka<br/>Message Broker]
        SR2[Schema Registry<br/>Avro Schemas]
    end

    subgraph "Processing Layer"
        SPARK2[Apache Spark<br/>Feature Engineering]
        AIRFLOW2[Apache Airflow<br/>Orchestration]
    end

    subgraph "Storage Layer"
        ICEBERG2[Apache Iceberg<br/>Data Lake]
        MINIO2[MinIO<br/>Object Storage]
    end

    QP2 -->|Stream Results| KAFKA2
    KAFKA2 <-->|Schema Validation| SR2
    KAFKA2 -->|Consume| SPARK2
    AIRFLOW2 -->|Schedule| SPARK2
    SPARK2 -->|Write Features| ICEBERG2
    ICEBERG2 -->|Store| MINIO2
    SPARK2 -->|Store Raw| MINIO2

    style QP2 fill:#c5cae9,color:#1a237e
    style KAFKA2 fill:#ffe082,color:#000
    style SR2 fill:#ffe082,color:#000
    style SPARK2 fill:#a5d6a7,color:#1b5e20
    style AIRFLOW2 fill:#90caf9,color:#0d47a1
    style ICEBERG2 fill:#b39ddb,color:#311b92
    style MINIO2 fill:#b39ddb,color:#311b92
```

---

## Technology Stack

=== "Quantum Computing"

    - **Qiskit** - IBM's quantum computing framework
    - **Qiskit Aer** - Quantum circuit simulator
    - **PySCF** - Quantum chemistry library for Python
    - **CUDA/cuQuantum** - GPU acceleration for quantum simulations

=== "Data Engineering"

    - **Apache Kafka** - Distributed event streaming platform
    - **Apache Spark** - Unified analytics engine for big data
    - **Apache Airflow** - Workflow orchestration platform
    - **Apache Iceberg** - Open table format for data lakes
    - **MinIO** - S3-compatible object storage

=== "Infrastructure"

    - **Docker** - Container platform
    - **Prometheus** - Monitoring and alerting toolkit
    - **Grafana** - Metrics visualization and dashboards
    - **PostgreSQL** - Relational database for metadata

---

## Use Cases

!!! example "Research & Development"

    - Explore VQE convergence behavior across different molecules
    - Benchmark CPU vs GPU acceleration for quantum simulations
    - Compare optimizer performance

!!! example "Data Science & ML"

    - Analyze quantum experiment metadata at scale
    - Create time-series predictions for molecular properties

!!! example "Production Deployments"

    - Run automated quantum simulations
    - Monitor system performance and scientific accuracy
    - Scaling processing with distributed Spark clusters

---

## Next Steps

1. **[Install Quantum Pipeline](getting-started/installation.md)** - Get up and running
2. **[First Simulation](getting-started/quick-start.md)** - H₂ molecule example
3. **[Configuration Options](usage/configuration.md)** - Customize your setup
4. **[Full Platform Deployment](deployment/docker-compose.md)** - Launch all services

---

## Links related to the project

- **Codeberg**: [piotrkrzysztof/quantum-pipeline](https://codeberg.org/piotrkrzysztof/quantum-pipeline)
- **GitHub (mirror)**: [straightchlorine/quantum-pipeline](https://github.com/straightchlorine/quantum-pipeline)
- **Docker Hub**: [straightchlorine/quantum-pipeline](https://hub.docker.com/r/straightchlorine/quantum-pipeline)
- **PyPI**: [quantum-pipeline](https://pypi.org/project/quantum-pipeline/)
- **Issues**: [Report bugs or request features](https://github.com/straightchlorine/quantum-pipeline/issues)

---

!!! info "Engineering Thesis Project"
    This project was developed as part of an engineering thesis at the **DSW University of Lower Silesia**
    focusing on GPU-accelerated quantum simulations and production-grade data engineering for quantum
    computing workflows.
