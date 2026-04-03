# Docker Compose Deployment

The full platform runs as a set of Docker containers managed by Docker Compose.
This deployment includes the simulation engine, Apache Kafka for data streaming,
Redpanda Connect for S3 sink streaming, Apache Spark for data processing,
Apache Airflow for workflow orchestration, Garage for S3-compatible object storage,
MLflow for experiment tracking, and monitoring exporters.

MLflow has not yet been integrated into the actual workflow. Ensuring stability
of other components, including the simulation module, takes precedence.

## Overview

Each component runs in its own container on a shared Docker bridge network.
The compose file is at
[`compose/docker-compose.ml.yaml`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/compose/docker-compose.ml.yaml).
It defines:

- **Simulation containers** - Quantum Pipeline GPU and CPU containers (in the `batch` profile, started by Airflow DAGs)
- **Streaming layer** - Apache Kafka (KRaft mode), Schema Registry, Redpanda Connect (S3 sink to Garage)
- **Processing layer** - Spark master and worker
- **Orchestration** - Airflow 3.x with CeleryExecutor: API server, scheduler, DAG processor, Celery worker, triggerer, with PostgreSQL and Redis
- **Storage** - Garage v2.2.0 S3-compatible object storage
- **ML tracking** - MLflow experiment tracking server (backed by PostgreSQL + Garage)
- **Monitoring exporters** - StatsD exporter, PostgreSQL exporter, Redis exporter, NVIDIA GPU exporter

Port mappings and dependencies are listed in the service tables below.

## Quick Start

### Prerequisites

- Docker Engine 24.0 or later
- Docker Compose v2.20 or later
- NVIDIA Container Toolkit (for GPU containers)
- [just](https://github.com/casey/just) command runner (recommended)

### Step 1: Clone the Repository

```bash
# codeberg mirror
git clone https://codeberg.org/piotrkrzysztof/quantum-pipeline.git

# or github
git clone https://github.com/straightchlorine/quantum-pipeline.git

cd quantum-pipeline
```

### Step 2: Configure Environment

Run the first-time setup script, which generates secrets and Garage configuration:

```bash
just ml-setup
```

This creates a `.env` file from `.env.ml.example` and generates the required secrets.
You can also do it manually:

```bash
cp .env.ml.example .env
chmod 600 .env
```

Edit `.env` to set credentials, resource limits, and service ports. See the
[Environment Variables](environment-variables.md) reference for all options.

### Step 3: Build Images

Build the simulation container images:

```bash
# CPU only
just docker-build cpu

# GPU (set CUDA_ARCH for your GPU, default 8.6/Ampere)
CUDA_ARCH=6.1 just docker-build gpu

# Both
just docker-build all
```

### Step 4: Deploy

```bash
just ml-up
```

This runs `docker compose --env-file .env -f compose/docker-compose.ml.yaml up -d`.

### Step 5: Verify

Check that all services are running:

```bash
docker compose --env-file .env -f compose/docker-compose.ml.yaml ps
```

## Service Configuration

### Streaming Services

| Service | Image | Port(s) | Description |
|---|---|---|---|
| `kafka` | `apache/kafka:4.2.0` | 9092 (internal), 9094 (external) | Message broker with KRaft mode |
| `schema-registry` | `confluentinc/cp-schema-registry:8.2.0` | 8081 | Avro schema management |
| `redpanda-connect` | `docker.redpanda.com/redpandadata/connect` | 4195 (metrics) | S3 sink streaming from Kafka to Garage |

Kafka runs in KRaft mode (no ZooKeeper). Schema Registry manages Avro schemas.
Redpanda Connect replaces the older Kafka Connect + S3 Sink connector setup, streaming
data from Kafka topics directly to Garage (S3).

#### Alternative: Confluent Kafka Connect

An override file at `compose/docker-compose.ml.kafka-connect.yaml` provides a
Confluent Kafka Connect worker with the S3 Sink connector (v12.1.0) as an alternative
to Redpanda Connect. To use it, pass both compose files and scale Redpanda Connect
to zero:

```bash
docker compose --env-file .env \
  -f compose/docker-compose.ml.yaml \
  -f compose/docker-compose.ml.kafka-connect.yaml \
  up -d --scale redpanda-connect=0
```

The override adds two services: `kafka-connect` (port 8083) which installs the S3
connector plugin on startup, and `kafka-connect-init` which registers the
`garage-sink` connector pointing at the `raw-results` bucket. Both depend on Kafka,
Schema Registry, and Garage being healthy.

### Processing Services

| Service | Image | Port(s) | Description |
|---|---|---|---|
| `spark-master` | `quantum-pipeline-spark:4.0.2-python3.12` | 8080 (UI), 7077 (RPC) | Spark cluster master |
| `spark-worker` | `quantum-pipeline-spark:4.0.2-python3.12` | - | Spark executor node (default: 2 GB memory, 2 cores) |

### Orchestration Services

| Service | Image | Port(s) | Description |
|---|---|---|---|
| `airflow-apiserver` | Custom (docker/airflow) | 8084 | Airflow API server and web UI |
| `airflow-scheduler` | Custom (docker/airflow) | - | DAG scheduling |
| `airflow-dag-processor` | Custom (docker/airflow) | - | DAG parsing and processing |
| `airflow-worker` | Custom (docker/airflow) | - | Celery task execution |
| `airflow-triggerer` | Custom (docker/airflow) | - | Deferred task execution |
| `airflow-init` | Custom (docker/airflow) | - | Database migration, user creation, Spark connection |
| `postgres` | `postgres:16-alpine` | - (internal only) | Airflow + MLflow metadata database (see note below) |
| `redis` | `redis:7.2-bookworm` | 6379 (internal only) | Celery broker |


The Airflow common environment also configures rclone remotes for Garage and
Cloudflare R2 via `RCLONE_CONFIG_*` environment variables. This allows the R2 sync
DAG to copy data between Garage (local S3) and Cloudflare R2 (remote S3) without a
separate rclone configuration file.

### Storage Services

| Service | Image | Port(s) | Description |
|---|---|---|---|
| `garage` | `dxflrs/garage:v2.2.0` | 3901 (S3 API), 3903 (admin) | S3-compatible object storage |

Garage stores raw VQE results, feature datasets, and Iceberg warehouse data. Buckets
are configured during `just ml-setup`. The S3 endpoint is `http://garage:3901` from
within the Docker network. The admin API on port 3903 exposes cluster status and
metrics that can be scraped by Prometheus.

### ML Tracking

| Service | Image | Port(s) | Description |
|---|---|---|---|
| `mlflow` | `ghcr.io/mlflow/mlflow:v3.10.1-full` | 5000 | Experiment tracking and artifact storage |

MLflow uses PostgreSQL for the backend store and Garage (S3) for artifact storage
(`s3://mlflow-artifacts/`).

### Monitoring Exporters

These services expose Prometheus metrics for scraping by an external monitoring server:

| Service | Image | Port(s) | Description |
|---|---|---|---|
| `airflow-statsd-exporter` | `prom/statsd-exporter` | 9102 (metrics), 9125/udp (StatsD) | Converts Airflow StatsD metrics to Prometheus format |
| `postgres-exporter` | `prometheuscommunity/postgres-exporter` | 9187 | PostgreSQL metrics |
| `redis-exporter` | `oliver006/redis_exporter` | 9121 | Redis metrics |
| `nvidia-gpu-exporter` | `utkuozdemir/nvidia_gpu_exporter:1.2.0` | 9835 | NVIDIA GPU utilization, memory, temperature |

### Batch Simulation Containers

These containers are defined in the `batch` profile and are not started by `just ml-up`.
They are launched on-demand by Airflow DAGs via `docker compose run`:

| Service | Image | Description |
|---|---|---|
| `quantum-pipeline-gpu1` | `quantum-pipeline:gpu` | GPU simulation (device 0, GTX 1060 6GB) |
| `quantum-pipeline-gpu2` | `quantum-pipeline:gpu` | GPU simulation (device 1, GTX 1050 Ti) |
| `quantum-pipeline-cpu` | `quantum-pipeline:cpu` | CPU simulation |

Each batch container connects to Kafka and Schema Registry, and exports monitoring
metrics via the `MONITORING_ENABLED`, `PUSHGATEWAY_URL`, and `MONITORING_EXPORT_FORMAT`
environment variables.

## Networking

All services share a single Docker bridge network (`quantum-ml-network`). Services
communicate using container names as hostnames (e.g., `kafka:9092`, `garage:3901`).
The Kafka external listener on port 9094 allows access from outside the Docker network.

## Volumes

The deployment uses named Docker volumes for persistent data:

| Volume | Service | Purpose |
|---|---|---|
| `quantum-garage-data` | Garage | Object storage data |
| `quantum-garage-meta` | Garage | Object storage metadata |
| `quantum-ml-spark-warehouse` | Spark | Spark SQL warehouse |
| `quantum-ml-spark-ivy-cache` | Spark | Cached JAR dependencies |
| `quantum-ml-airflow-postgres` | PostgreSQL | Airflow + MLflow metadata |
| `quantum-ml-airflow-logs` | Airflow | Task execution logs |
| `quantum-ml-airflow-ivy-cache` | Airflow | Cached JARs for SparkSubmit |
| `quantum-ml-redis-data` | Redis | Celery broker data |
| `quantum-ml-kafka-data` | Kafka | Kafka log segments |

Bind mounts provide access to project files: DAG definitions (`docker/airflow/`),
Spark configuration (`compose/spark-defaults.conf`), Garage configuration
(`compose/garage.toml`), Redpanda Connect config (`compose/redpanda-connect.yaml`),
simulation data (`data/`), and output (`gen/`).

## Justfile Commands

| Command | Description |
|---|---|
| `just ml-setup` | First-time setup: generates secrets, Garage config, `.env` file |
| `just ml-up` | Start the ML pipeline stack |
| `just ml-down` | Force-remove running batch containers, then stop the ML stack (includes `--profile batch`) |
| `just docker-build cpu` | Build CPU simulation image |
| `just docker-build gpu` | Build GPU simulation image |
| `just docker-build all` | Build both CPU and GPU images |
| `just docker-up` | Start the base compose stack (`compose/docker-compose.yaml`) |
| `just docker-down` | Stop the base compose stack |
| `just docker-logs [service]` | Tail logs from the base compose stack |

## Scaling

Scale Spark workers with:

```bash
docker compose --env-file .env -f compose/docker-compose.ml.yaml up -d --scale spark-worker=3
```

For additional simulation instances, add new service definitions with a unique
`container_name`, distinct Kafka topic, appropriate GPU device assignment, and
separate output volume mounts.

## Stopping the Deployment

```bash
# Stop services, preserve data (also force-removes any running batch containers)
just ml-down

# Or manually, including volume cleanup
docker compose --env-file .env -f compose/docker-compose.ml.yaml --profile batch down -v
```
