# Environment Variables

The Quantum Pipeline platform is configured through environment variables defined in
a `.env` file at the project root. Docker Compose reads this file and injects the
variables into each service container.

## Overview

An example configuration file is provided as [`.env.thesis.example`](https://github.com/straightchlorine/quantum-pipeline/blob/master/.env.thesis.example). Copy it and
customize for your environment:

```bash
cp .env.thesis.example .env
chmod 600 .env
```

## Complete Reference

### Quantum Pipeline Settings

| Variable | Default | Description |
|---|---|---|
| `MAX_ITERATIONS` | `50` | Maximum VQE optimizer iterations per molecule. |
| `LOG_LEVEL` | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR`. |
| `SIMULATION_METHOD` | `statevector` | Qiskit Aer simulation method: `statevector`, `density_matrix`, `automatic`, `stabilizer`, `unitary`. **Note:** Docker default is `statevector`; Python CLI default is `tensor_network` (see `defaults.py`). |
| `BASIS_SET` | `sto-3g` | Quantum chemistry basis set: `sto-3g`, `6-31g`, `cc-pvdz`, etc. **Note:** Docker env var uses `sto-3g` (hyphenated); Python CLI default is `sto3g`. Not currently referenced in compose files. |

### IBM Quantum Settings

Optional — only needed when using IBM Quantum cloud backends instead of the local Aer simulator.

| Variable | Default | Description |
|---|---|---|
| `IBM_RUNTIME_CHANNEL` | `ibm_quantum` | IBM Quantum Runtime channel. |
| `IBM_RUNTIME_INSTANCE` | - | IBM Quantum service instance identifier. |
| `IBM_RUNTIME_TOKEN` | - | Authentication token for IBM Quantum. |

### Kafka Configuration

| Variable | Default | Description |
|---|---|---|
| `KAFKA_SERVERS` | `kafka:9092` | Kafka bootstrap servers. |
| `KAFKA_EXTERNAL_HOST_IP` | `localhost` | IP/hostname for external Kafka access. |
| `KAFKA_EXTERNAL_PORT` | `9094` | External Kafka listener port. |
| `KAFKA_INTERNAL_PORT` | `9092` | Internal Kafka listener port. |
| `KAFKA_VERSION` | `latest` | Bitnami Kafka Docker image tag. |

### Schema Registry

| Variable | Default | Description |
|---|---|---|
| `SCHEMA_REGISTRY_VERSION` | `latest` | Docker image tag. |
| `SCHEMA_REGISTRY_TOPIC` | `_schemas` | Internal Kafka topic for schema storage. |
| `SCHEMA_REGISTRY_HOSTNAME` | `schema-registry` | Schema Registry hostname. |
| `SCHEMA_REGISTRY_PORT` | `8081` | HTTP listener port. |

### Kafka Connect

| Variable | Default | Description |
|---|---|---|
| `KAFKA_CONNECT_VERSION` | `latest` | Docker image tag. |
| `KAFKA_CONNECT_PORT` | `8083` | REST API port. |
| `KAFKA_CONNECT_LOG_LEVEL` | `INFO` | Log level: `DEBUG`, `INFO`, `WARN`, `ERROR`. |

### MinIO Configuration

| Variable | Default | Description |
|---|---|---|
| `MINIO_VERSION` | `latest` | Docker image tag. |
| `MINIO_API_PORT` | `9000` | S3 API port. |
| `MINIO_CONSOLE_PORT` | `9002` | Web console port. |
| `MINIO_ROOT_USER` | `quantum-admin` | Root username (also used for S3 access). |
| `MINIO_ROOT_PASSWORD` | `quantum-secret-key` | Root password. |
| `MINIO_REGION` | `us-east-1` | S3 region. Must match Kafka Connect S3 Sink config. |
| `MINIO_HOSTNAME` | `quantum-minio` | Hostname alias used by `mc` for bucket setup. |
| `MINIO_BUCKET` | `quantum-data` | Default bucket for experiment data. |
| `MINIO_ACCESS_KEY` | `quantum-admin` | S3 access key for Spark and Kafka Connect. |
| `MINIO_SECRET_KEY` | `quantum-secret-key` | S3 secret key for Spark and Kafka Connect. |

### Airflow Configuration

| Variable | Default | Description |
|---|---|---|
| `AIRFLOW_POSTGRES_USER` | `airflow` | PostgreSQL username. |
| `AIRFLOW_POSTGRES_PASSWORD` | `airflow-password` | PostgreSQL password. **Note:** compose fallback is `airflow` (`:-airflow`); `.env.thesis.example` sets `airflow-password`. |
| `AIRFLOW_POSTGRES_DB` | `airflow` | PostgreSQL database name. |
| `AIRFLOW_FERNET_KEY` | - | Fernet encryption key for connection credentials. See [Security Considerations](#security-considerations) to generate. |
| `AIRFLOW_WEBSERVER_SECRET_KEY` | - | Secret key for webserver session signing. See [Security Considerations](#security-considerations) to generate. |
| `AIRFLOW_WEBSERVER_PORT` | `8084` | Host port for the Airflow web UI. |
| `AIRFLOW_DAGS_PAUSED_AT_CREATION` | `True` | Whether new DAGs start paused. |
| `AIRFLOW_LOAD_EXAMPLES` | `False` | Load Airflow example DAGs. |
| `AIRFLOW_ADMIN_USERNAME` | `admin` | Initial admin username. |
| `AIRFLOW_ADMIN_PASSWORD` | `admin` | Initial admin password. |
| `AIRFLOW_ADMIN_FIRSTNAME` | `Admin` | Admin user first name. |
| `AIRFLOW_ADMIN_LASTNAME` | `User` | Admin user last name. |
| `AIRFLOW_ADMIN_EMAIL` | `admin@example.com` | Admin user email. |

### Spark Configuration

| Variable | Default | Description |
|---|---|---|
| `SPARK_MASTER_HOST` | `spark-master` | Spark master hostname. |
| `SPARK_MASTER_PORT` | `7077` | Spark master RPC port. |
| `SPARK_DEFAULT_QUEUE` | `default` | Default resource queue for Spark jobs. |

### Monitoring Configuration

| Variable | Default | Description |
|---|---|---|
| `EXTERNAL_PUSHGATEWAY_URL` | `http://your-monitoring-server:9091` | External Prometheus PushGateway URL for cross-host monitoring. Not currently referenced in compose files. |
| `QUANTUM_PERFORMANCE_ENABLED` | `true` | Enable performance metric collection. |
| `QUANTUM_PERFORMANCE_COLLECTION_INTERVAL` | `10` | Metric collection interval in seconds. |
| `QUANTUM_PERFORMANCE_PUSHGATEWAY_URL` | `http://monit:9091` | PushGateway URL for performance metrics. |
| `QUANTUM_PERFORMANCE_EXPORT_FORMAT` | `json,prometheus` | Export format: `json`, `prometheus`, or both (comma-separated). |

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
MAX_ITERATIONS=50
LOG_LEVEL=INFO
SIMULATION_METHOD=statevector
BASIS_SET=sto-3g

IBM_RUNTIME_CHANNEL=ibm_quantum
IBM_RUNTIME_INSTANCE=your-instance-id
IBM_RUNTIME_TOKEN=your-token

KAFKA_SERVERS=kafka:9092
KAFKA_EXTERNAL_HOST_IP=localhost
KAFKA_EXTERNAL_PORT=9094
KAFKA_INTERNAL_PORT=9092
KAFKA_VERSION=latest

SCHEMA_REGISTRY_VERSION=latest
SCHEMA_REGISTRY_TOPIC=_schemas
SCHEMA_REGISTRY_HOSTNAME=schema-registry
SCHEMA_REGISTRY_PORT=8081

KAFKA_CONNECT_VERSION=latest
KAFKA_CONNECT_PORT=8083
KAFKA_CONNECT_LOG_LEVEL=INFO

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

SPARK_MASTER_HOST=spark-master
SPARK_MASTER_PORT=7077
SPARK_DEFAULT_QUEUE=default

EXTERNAL_PUSHGATEWAY_URL=http://your-monitoring-server:9091

QUANTUM_PERFORMANCE_ENABLED=true
QUANTUM_PERFORMANCE_COLLECTION_INTERVAL=10
QUANTUM_PERFORMANCE_PUSHGATEWAY_URL=http://monit:9091
QUANTUM_PERFORMANCE_EXPORT_FORMAT=json,prometheus
```
