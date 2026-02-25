# Environment Variables

The Quantum Pipeline platform is configured through environment variables defined in
a `.env` file at the project root. Docker Compose reads this file and injects the
variables into each service container. This page provides a complete reference for
all configuration options.

## Overview

An example configuration file is provided as `.env.thesis.example`. Copy it and
customize for your environment:

```bash
cp .env.thesis.example .env
chmod 600 .env
```

Variables follow a naming convention organized by service:

- `MAX_ITERATIONS`, `LOG_LEVEL`, etc. - Quantum Pipeline settings
- `IBM_RUNTIME_*` - IBM Quantum cloud backend
- `KAFKA_*` - Apache Kafka broker
- `SCHEMA_REGISTRY_*` - Confluent Schema Registry
- `KAFKA_CONNECT_*` - Kafka Connect
- `MINIO_*` - MinIO object storage
- `AIRFLOW_*` - Apache Airflow
- `SPARK_*` - Apache Spark
- `QUANTUM_PERFORMANCE_*` - Performance monitoring

## Complete Reference

### Quantum Pipeline Settings

| Variable | Default | Description | Required |
|---|---|---|---|
| `MAX_ITERATIONS` | `50` | Maximum number of VQE optimizer iterations per molecule. Higher values allow better convergence but increase runtime. | Yes |
| `LOG_LEVEL` | `INFO` | Logging verbosity. Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`. | No |
| `SIMULATION_METHOD` | `statevector` | Qiskit Aer simulation method. Options: `statevector`, `density_matrix`, `automatic`, `stabilizer`, `unitary`. | No |
| `BASIS_SET` | `sto-3g` | Quantum chemistry basis set for molecular orbital calculations. Options: `sto-3g`, `6-31g`, `cc-pvdz`, etc. | No |

### IBM Quantum Settings

These variables are optional and only required when using IBM Quantum cloud backends
instead of the local Aer simulator.

| Variable | Default | Description | Required |
|---|---|---|---|
| `IBM_RUNTIME_CHANNEL` | `ibm_quantum` | IBM Quantum Runtime channel. | No |
| `IBM_RUNTIME_INSTANCE` | -- | IBM Quantum service instance identifier. | No |
| `IBM_RUNTIME_TOKEN` | -- | Authentication token for IBM Quantum. Obtain from the IBM Quantum dashboard. | No |

!!! warning "Credential Security"
    Never commit IBM Quantum tokens to version control. Use environment variables
    or secrets management tools.

### Kafka Configuration

| Variable | Default | Description | Required |
|---|---|---|---|
| `KAFKA_SERVERS` | `kafka:9092` | Bootstrap servers for Kafka connections. Uses the internal Docker hostname. | Yes |
| `KAFKA_EXTERNAL_HOST_IP` | `localhost` | IP address or hostname for external Kafka access. | No |
| `KAFKA_EXTERNAL_PORT` | `9094` | Port for external Kafka listener. | No |
| `KAFKA_INTERNAL_PORT` | `9092` | Port for internal (Docker network) Kafka listener. | No |
| `KAFKA_VERSION` | `latest` | Bitnami Kafka Docker image tag. | No |

### Schema Registry

| Variable | Default | Description | Required |
|---|---|---|---|
| `SCHEMA_REGISTRY_VERSION` | `latest` | Confluent Schema Registry Docker image tag. | No |
| `SCHEMA_REGISTRY_TOPIC` | `_schemas` | Internal Kafka topic for schema storage. | No |
| `SCHEMA_REGISTRY_HOSTNAME` | `schema-registry` | Schema Registry hostname within the Docker network. | No |
| `SCHEMA_REGISTRY_PORT` | `8081` | Schema Registry HTTP listener port. | No |

### Kafka Connect

| Variable | Default | Description | Required |
|---|---|---|---|
| `KAFKA_CONNECT_VERSION` | `latest` | Confluent Kafka Connect Docker image tag. | No |
| `KAFKA_CONNECT_PORT` | `8083` | Kafka Connect REST API port. | No |
| `KAFKA_CONNECT_LOG_LEVEL` | `INFO` | Log level for Kafka Connect. Options: `DEBUG`, `INFO`, `WARN`, `ERROR`. | No |

### MinIO Configuration

| Variable | Default | Description | Required |
|---|---|---|---|
| `MINIO_VERSION` | `latest` | MinIO Docker image tag. | No |
| `MINIO_API_PORT` | `9000` | MinIO S3 API port. | Yes |
| `MINIO_CONSOLE_PORT` | `9002` | MinIO web console port. | Yes |
| `MINIO_ROOT_USER` | `quantum-admin` | MinIO root username (also used for S3 access). | Yes |
| `MINIO_ROOT_PASSWORD` | `quantum-secret-key` | MinIO root password. | Yes |
| `MINIO_REGION` | `us-east-1` | S3 region name. Must match Kafka Connect S3 Sink configuration. | No |
| `MINIO_HOSTNAME` | `quantum-minio` | Hostname alias used by `mc` (MinIO Client) for bucket setup. | No |
| `MINIO_BUCKET` | `quantum-data` | Default bucket name for experiment data. | Yes |
| `MINIO_ACCESS_KEY` | `quantum-admin` | S3 access key for Spark and Kafka Connect. | Yes |
| `MINIO_SECRET_KEY` | `quantum-secret-key` | S3 secret key for Spark and Kafka Connect. | Yes |

### Airflow Configuration

| Variable | Default | Description | Required |
|---|---|---|---|
| `AIRFLOW_POSTGRES_USER` | `airflow` | PostgreSQL username for Airflow metadata database. | Yes |
| `AIRFLOW_POSTGRES_PASSWORD` | `airflow-password` | PostgreSQL password for Airflow metadata database. | Yes |
| `AIRFLOW_POSTGRES_DB` | `airflow` | PostgreSQL database name for Airflow. | Yes |
| `AIRFLOW_FERNET_KEY` | -- | Fernet encryption key for Airflow connection credentials. Generate with `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`. | Yes |
| `AIRFLOW_WEBSERVER_SECRET_KEY` | -- | Secret key for Airflow webserver session signing. | Yes |
| `AIRFLOW_WEBSERVER_PORT` | `8084` | Host port mapping for the Airflow web interface. | No |
| `AIRFLOW_DAGS_PAUSED_AT_CREATION` | `True` | Whether new DAGs start in a paused state. | No |
| `AIRFLOW_LOAD_EXAMPLES` | `False` | Load Airflow example DAGs. Set to `False` for production. | No |
| `AIRFLOW_ADMIN_USERNAME` | `admin` | Initial admin user username. | Yes |
| `AIRFLOW_ADMIN_PASSWORD` | `admin` | Initial admin user password. | Yes |
| `AIRFLOW_ADMIN_FIRSTNAME` | `Admin` | Admin user first name. | No |
| `AIRFLOW_ADMIN_LASTNAME` | `User` | Admin user last name. | No |
| `AIRFLOW_ADMIN_EMAIL` | `admin@example.com` | Admin user email address. | No |

### Spark Configuration

| Variable | Default | Description | Required |
|---|---|---|---|
| `SPARK_MASTER_HOST` | `spark-master` | Spark master hostname within the Docker network. | Yes |
| `SPARK_MASTER_PORT` | `7077` | Spark master RPC port. | Yes |
| `SPARK_DEFAULT_QUEUE` | `default` | Default resource queue for Spark jobs. | No |

### Monitoring Configuration

| Variable | Default | Description | Required |
|---|---|---|---|
| `EXTERNAL_PUSHGATEWAY_URL` | `http://your-monitoring-server:9091` | URL of an external Prometheus PushGateway for cross-host monitoring. | No |
| `QUANTUM_PERFORMANCE_ENABLED` | `true` | Enable performance metric collection from pipeline containers. | No |
| `QUANTUM_PERFORMANCE_COLLECTION_INTERVAL` | `10` | Metric collection interval in seconds. | No |
| `QUANTUM_PERFORMANCE_PUSHGATEWAY_URL` | `http://monit:9091` | PushGateway URL for performance metrics. | No |
| `QUANTUM_PERFORMANCE_EXPORT_FORMAT` | `json,prometheus` | Export format for metrics. Options: `json`, `prometheus`, or both (comma-separated). | No |

## Resource Allocation Notes

The thesis experiment environment used the following resource distribution across a
system with 6 CPUs, 56 GB RAM, and 2 GPUs:

| Component | CPUs | RAM | GPU |
|---|---|---|---|
| CPU Pipeline | 2.0 (reserved: 1.5) | 10 GB (reserved: 6 GB) | -- |
| GPU Pipeline 1 | 2.0 (reserved: 1.5) | 10 GB (reserved: 6 GB) | GTX 1060 6GB |
| GPU Pipeline 2 | 2.0 (reserved: 1.5) | 10 GB (reserved: 6 GB) | GTX 1050 Ti 4GB |
| Spark Worker | 1.0 (reserved: 0.5) | 4 GB (reserved: 2 GB) | -- |
| Airflow Webserver | 0.5 (reserved: 0.2) | 2 GB (reserved: 1 GB) | -- |
| Infrastructure (Kafka, MinIO, etc.) | Shared | Shared | -- |

Docker Compose `deploy.resources` settings enforce these limits. The `limits` values
define hard caps, while `reservations` define the minimum guaranteed resources.

!!! tip "Adjusting for Your Hardware"
    Scale CPU and memory allocations proportionally to your available resources.
    The pipeline containers benefit most from additional memory, while infrastructure
    services (Kafka, MinIO) run well within modest allocations.

## Security Considerations

Never commit `.env` to version control and set restrictive permissions (`chmod 600 .env`). Generate strong keys for Airflow:

```bash
# Fernet key
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# Webserver secret key
python -c "import secrets; print(secrets.token_hex(32))"
```

In production, restrict service port exposure by binding to `127.0.0.1` (e.g., `"127.0.0.1:8081:8081"`) and use unique passwords for each service.

## Example .env File

The following is the complete `.env.thesis.example` provided with the project:

```bash
# Quantum Pipeline Thesis Configuration
# Copy to .env and customize for your environment

# Quantum Pipeline Settings
MAX_ITERATIONS=50
LOG_LEVEL=INFO
SIMULATION_METHOD=statevector
BASIS_SET=sto-3g

# IBM Quantum Settings (optional - for cloud backend comparison)
IBM_RUNTIME_CHANNEL=ibm_quantum
IBM_RUNTIME_INSTANCE=your-instance-id
IBM_RUNTIME_TOKEN=your-token

# Kafka Configuration
KAFKA_SERVERS=kafka:9092
KAFKA_EXTERNAL_HOST_IP=localhost
KAFKA_EXTERNAL_PORT=9094
KAFKA_INTERNAL_PORT=9092
KAFKA_VERSION=latest

# Schema Registry
SCHEMA_REGISTRY_VERSION=latest
SCHEMA_REGISTRY_TOPIC=_schemas
SCHEMA_REGISTRY_HOSTNAME=schema-registry
SCHEMA_REGISTRY_PORT=8081

# Kafka Connect
KAFKA_CONNECT_VERSION=latest
KAFKA_CONNECT_PORT=8083
KAFKA_CONNECT_LOG_LEVEL=INFO

# MinIO Configuration
MINIO_VERSION=latest
MINIO_API_PORT=9000
MINIO_CONSOLE_PORT=9002
MINIO_ROOT_USER=quantum-admin
MINIO_ROOT_PASSWORD=quantum-secret-key
MINIO_REGION=us-east-1
MINIO_HOSTNAME=quantum-minio
MINIO_BUCKET=quantum-data
MINIO_ACCESS_KEY=quantum-admin
MINIO_SECRET_KEY=quantum-secret-key

# Airflow Configuration
AIRFLOW_POSTGRES_USER=airflow
AIRFLOW_POSTGRES_PASSWORD=airflow-password
AIRFLOW_POSTGRES_DB=airflow
AIRFLOW_FERNET_KEY=your-fernet-key
AIRFLOW_WEBSERVER_SECRET_KEY=your-webserver-secret
AIRFLOW_WEBSERVER_PORT=8084
AIRFLOW_DAGS_PAUSED_AT_CREATION=True
AIRFLOW_LOAD_EXAMPLES=False
AIRFLOW_ADMIN_USERNAME=admin
AIRFLOW_ADMIN_PASSWORD=admin
AIRFLOW_ADMIN_FIRSTNAME=Admin
AIRFLOW_ADMIN_LASTNAME=User
AIRFLOW_ADMIN_EMAIL=admin@example.com

# Spark Configuration
SPARK_MASTER_HOST=spark-master
SPARK_MASTER_PORT=7077
SPARK_DEFAULT_QUEUE=default

# External Monitoring Configuration
EXTERNAL_PUSHGATEWAY_URL=http://your-monitoring-server:9091

# Performance Monitoring Settings
QUANTUM_PERFORMANCE_ENABLED=true
QUANTUM_PERFORMANCE_COLLECTION_INTERVAL=10
QUANTUM_PERFORMANCE_PUSHGATEWAY_URL=http://monit:9091
QUANTUM_PERFORMANCE_EXPORT_FORMAT=json,prometheus

# Resource Allocation Notes:
# Total System: 6 CPUs, 56GB RAM, 2 GPUs
# CPU Pipeline: 3 CPUs, 16GB RAM
# GPU Pipeline: 3 CPUs, 16GB RAM + GTX 1060
# Infrastructure: 1 CPU, 24GB RAM (Kafka, Spark, etc.)
# Monitoring: 0.5 CPU, 8GB RAM
```
