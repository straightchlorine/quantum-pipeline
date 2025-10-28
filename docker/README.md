# Docker Deployment Guide

This directory contains Docker configurations for deploying the Quantum Pipeline platform with all its data engineering components.

## Table of Contents
- [Available Dockerfiles](#available-dockerfiles)
- [Docker Compose Setup](#docker-compose-setup)
- [Environment Variables](#environment-variables)
- [Service Configuration](#service-configuration)
- [Troubleshooting](#troubleshooting)

---

## Available Dockerfiles

### `Dockerfile.cpu`
CPU-optimized quantum simulation container.

**Features**:
- Optimized for multi-core CPU processing
- Qiskit Aer with tensor network simulation
- Suitable for development and small-to-medium molecules
- No GPU dependencies required

**Build**:
```bash
docker build -f docker/Dockerfile.cpu -t quantum-pipeline:cpu .
```

### `Dockerfile.gpu`
GPU-accelerated quantum simulation container.

**Features**:
- NVIDIA CUDA support
- GPU-accelerated tensor network simulation
- cuQuantum integration (for Volta/Ampere GPUs with CUDA ≥11.2)
- Optimized for large-scale molecular simulations
- Requires NVIDIA Docker runtime

**Build**:
```bash
docker build -f docker/Dockerfile.gpu -t quantum-pipeline:gpu .
```

**Prerequisites**:
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit installed
- Docker configured with nvidia runtime

### `Dockerfile.spark`
Apache Spark cluster nodes for data processing.

**Features**:
- Apache Spark with Iceberg support
- S3-compatible storage integration (MinIO)
- Supports both master and worker node configurations
- Includes all necessary Hadoop AWS dependencies

**Build**:
```bash
docker build -f docker/Dockerfile.spark -t quantum-pipeline-spark .
```

### `Dockerfile.airflow`
Apache Airflow workflow orchestration container.

**Features**:
- Apache Airflow 2.7.1
- Pre-configured with Spark connection
- DAG support for automated processing
- PostgreSQL backend for metadata storage

**Build**:
```bash
docker build -f docker/Dockerfile.airflow -t quantum-pipeline-airflow .
```

---

## Docker Compose Setup

### Quick Start

Start the complete platform:
```bash
docker-compose up -d
```

Start specific services:
```bash
# Just Kafka and dependencies
docker-compose up -d kafka schema-registry

# Quantum pipeline with Kafka streaming
docker-compose up -d quantum-pipeline kafka schema-registry
```

Stop all services:
```bash
docker-compose down
```

Stop and remove volumes (⚠️ **deletes all data**):
```bash
docker-compose down -v
```

### Service Dependencies

The platform services have the following dependency chain:
```
quantum-pipeline
  ├── kafka-connect-init
  │   └── kafka-connect
  │       ├── kafka (healthy)
  │       ├── schema-registry (healthy)
  │       └── minio (started)
  ├── kafka (healthy)
  └── schema-registry (healthy)

airflow-webserver, airflow-scheduler, airflow-triggerer
  └── postgres (healthy)

spark-worker
  └── spark-master
      └── minio (started)
```

---

## Environment Variables

The platform can be configured via environment variables. Create a `.env` file in the project root or set these in your environment.

### Quantum Pipeline Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_ITERATIONS` | `100` | Maximum VQE optimization iterations |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `SIMULATION_METHOD` | `statevector` | Backend simulation method (statevector, tensor_network, etc.) |
| `IBM_RUNTIME_CHANNEL` | - | IBM Quantum runtime channel (optional) |
| `IBM_RUNTIME_INSTANCE` | - | IBM Quantum instance ID (optional) |
| `IBM_RUNTIME_TOKEN` | - | IBM Quantum access token (optional) |

### Kafka Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `KAFKA_SERVERS` | `kafka:9092` | Kafka bootstrap servers |
| `KAFKA_VERSION` | `latest` | Bitnami Kafka image version |
| `KAFKA_EXTERNAL_HOST_IP` | `localhost` | External IP for Kafka access |
| `KAFKA_EXTERNAL_PORT` | `9094` | External port for Kafka access |
| `KAFKA_INTERNAL_PORT` | `9092` | Internal port for Kafka (container network) |

### Schema Registry Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SCHEMA_REGISTRY_VERSION` | `latest` | Confluent Schema Registry version |
| `SCHEMA_REGISTRY_TOPIC` | `_schemas` | Schema Registry internal topic |
| `SCHEMA_REGISTRY_HOSTNAME` | `schema-registry` | Schema Registry hostname |
| `SCHEMA_REGISTRY_PORT` | `8081` | Schema Registry port |

### Kafka Connect Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `KAFKA_CONNECT_VERSION` | `latest` | Confluent Kafka Connect version |
| `KAFKA_CONNECT_PORT` | `8083` | Kafka Connect REST API port |
| `KAFKA_CONNECT_LOG_LEVEL` | `INFO` | Kafka Connect log level |

### MinIO (Object Storage) Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MINIO_VERSION` | `latest` | MinIO image version |
| `MINIO_ROOT_USER` | `minio` | MinIO root username |
| `MINIO_ROOT_PASSWORD` | `minio123` | MinIO root password |
| `MINIO_ACCESS_KEY` | `minioadmin` | MinIO access key |
| `MINIO_SECRET_KEY` | `minioadmin` | MinIO secret key |
| `MINIO_REGION` | `us-east-1` | MinIO region name |
| `MINIO_HOSTNAME` | `minio` | MinIO hostname for mc client |
| `MINIO_BUCKET` | `quantum-data` | Default bucket name |
| `MINIO_API_PORT` | `9000` | MinIO API port |
| `MINIO_CONSOLE_PORT` | `9001` | MinIO web console port |

### Spark Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SPARK_MASTER_HOST` | `spark-master` | Spark master hostname |
| `SPARK_MASTER_PORT` | `7077` | Spark master port |
| `SPARK_DEFAULT_QUEUE` | `default` | Default Spark queue |

### Airflow Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `AIRFLOW_POSTGRES_USER` | `airflow` | PostgreSQL username for Airflow |
| `AIRFLOW_POSTGRES_PASSWORD` | `airflow` | PostgreSQL password for Airflow |
| `AIRFLOW_POSTGRES_DB` | `airflow` | PostgreSQL database name |
| `AIRFLOW_FERNET_KEY` | - | Fernet key for encrypting passwords (required) |
| `AIRFLOW_WEBSERVER_SECRET_KEY` | - | Secret key for webserver sessions (required) |
| `AIRFLOW_ADMIN_USERNAME` | `admin` | Airflow admin username |
| `AIRFLOW_ADMIN_PASSWORD` | `admin` | Airflow admin password |
| `AIRFLOW_ADMIN_FIRSTNAME` | `Admin` | Airflow admin first name |
| `AIRFLOW_ADMIN_LASTNAME` | `User` | Airflow admin last name |
| `AIRFLOW_ADMIN_EMAIL` | `admin@example.com` | Airflow admin email |
| `AIRFLOW_WEBSERVER_PORT` | `8084` | Airflow webserver port |
| `AIRFLOW_DAGS_PAUSED_AT_CREATION` | `True` | Pause DAGs on creation |
| `AIRFLOW_LOAD_EXAMPLES` | `False` | Load example DAGs |

### Performance Monitoring Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `QUANTUM_PERFORMANCE_ENABLED` | `false` | Enable performance monitoring |
| `QUANTUM_PERFORMANCE_COLLECTION_INTERVAL` | `30` | Metrics collection interval (seconds) |
| `QUANTUM_PERFORMANCE_PUSHGATEWAY_URL` | - | Prometheus PushGateway URL |
| `QUANTUM_PERFORMANCE_EXPORT_FORMAT` | `json` | Export format (json, prometheus, both) |
| `CONTAINER_TYPE` | `unknown` | Container type label for metrics |

---

## Service Configuration

### Access Endpoints

When running with `docker-compose up`, the following services are accessible:

| Service | URL | Description |
|---------|-----|-------------|
| Airflow Webserver | http://localhost:8084 | Workflow management UI |
| Spark Master UI | http://localhost:8080 | Spark cluster monitoring |
| MinIO Console | http://localhost:9001 | Object storage management |
| Schema Registry | http://localhost:8081 | Avro schema management |
| Kafka Connect | http://localhost:8083 | Connector management API |
| Kafka (external) | localhost:9094 | External Kafka access |
| Kafka (internal) | kafka:9092 | Container network access |

### Generating Required Secrets

Some services require secret keys. Generate them using:

```bash
# Generate Airflow Fernet key
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# Generate Airflow webserver secret key
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Add these to your `.env` file:
```env
AIRFLOW_FERNET_KEY=your_generated_fernet_key_here
AIRFLOW_WEBSERVER_SECRET_KEY=your_generated_secret_key_here
```

### Persistent Volumes

The platform uses named Docker volumes for data persistence:

| Volume | Purpose |
|--------|---------|
| `quantum-minio-data` | MinIO object storage data |
| `quantum-spark-warehouse` | Spark/Iceberg warehouse data |
| `quantum-airflow-postgres` | Airflow PostgreSQL database |
| `quantum-airflow-logs` | Airflow task logs |
| `quantum-kafka-data` | Kafka broker data |

---

## Troubleshooting

### Common Issues

**Issue**: Kafka broker not available
```
ERROR: NoBrokersAvailable
```
**Solution**: Wait for Kafka to fully start (check `docker-compose logs kafka`), or increase healthcheck retries.

---

**Issue**: Airflow webserver fails to start
```
ERROR: airflow.exceptions.AirflowException: error running migrations
```
**Solution**: Ensure `airflow-init` service completed successfully. Check `AIRFLOW_FERNET_KEY` is set.

---

**Issue**: GPU not detected in quantum-pipeline container
```
WARNING: GPU requested but not available
```
**Solution**:
- Verify NVIDIA Docker runtime is installed: `nvidia-smi`
- Check docker-compose.yaml has GPU reservation configured
- Ensure host has compatible NVIDIA GPU

---

**Issue**: MinIO buckets not created
```
ERROR: The specified bucket does not exist
```
**Solution**: Ensure `mc-setup` service completed successfully. Check logs: `docker-compose logs mc-setup`

---

**Issue**: Out of memory during large molecule simulations
```
ERROR: MemoryError or CUDA out of memory
```
**Solution**:
- Reduce `ansatz_reps` parameter
- Use fewer `shots`
- Adjust `max_memory_mb` in GPU options (see `configs/defaults.py`)
- Use CPU backend for very large systems

---

### Health Checks

Check service health status:
```bash
docker-compose ps
```

View logs for specific service:
```bash
docker-compose logs -f quantum-pipeline
docker-compose logs -f kafka
docker-compose logs -f spark-master
```

Restart a specific service:
```bash
docker-compose restart quantum-pipeline
```

---

## Future Updates

**Planned**: Migration to VMware Bitnami image catalog (https://app-catalog.vmware.com/bitnami/apps) for improved stability and security updates. This will affect Kafka and Spark images in future releases.

---

## Additional Resources

- [Main README](../README.md) - General project documentation
- [Monitoring Setup](../monitoring/README.md) - Performance monitoring configuration
- [Qiskit Documentation](https://qiskit.org/documentation/)
- [Apache Kafka Documentation](https://kafka.apache.org/documentation/)
- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [MinIO Documentation](https://docs.min.io/)
