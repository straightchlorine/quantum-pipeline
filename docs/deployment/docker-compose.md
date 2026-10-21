---
title: "Deploy the Full Stack with Docker Compose"
description: "Docker Compose setup of the full platform: simulation, Kafka, Spark, Airflow, Garage, MLflow and monitoring exporters, with service ports and just commands."
---

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
- **Streaming layer** - Apache Kafka (KRaft mode), Schema Registry, Redpanda Connect (S3 sink)
- **Processing layer** - Spark master and worker
- **Orchestration** - Airflow 3.x with CeleryExecutor: API server, scheduler, DAG processor, Celery worker, triggerer, with PostgreSQL and Redis
- **Storage** - Garage v2.2.0 S3-compatible object storage, or an external S3 service named by `S3_ENDPOINT`
- **ML tracking** - MLflow experiment tracking server (backed by PostgreSQL + S3)
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

Run the first-time setup script, which generates secrets, the Spark configuration
and the object storage setup:

```bash
just setup
```

This creates a `.env` file from `.env.ml.example`, generates the required
secrets, renders `compose/spark-defaults.conf` and creates the buckets. By default
it configures the compose Garage. To use an existing S3 service instead, create
`.env` first, set `S3_ENDPOINT`, `S3_REGION`, `S3_ACCESS_KEY` and `S3_SECRET_KEY`
in it, then run `just setup`; see
[External S3 storage](#external-s3-storage).

You can also create `.env` manually:

```bash
cp .env.ml.example .env
chmod 600 .env
```

`just setup` (or `bash scripts/ml-setup.sh`) is still required afterwards: it
fills in the secrets and renders the Spark configuration the stack mounts.

Edit `.env` to set credentials, resource limits, and service ports. See the
[Environment Variables](environment-variables.md) reference for all options.

### Step 3: Build Images

Build the simulation container images:

```bash
# CPU only
just docker-build cpu

# GPU (optional: set CUDA_ARCH to your GPU for a faster local build)
CUDA_ARCH=6.1 just docker-build gpu

# Both
just docker-build all
```

### Step 4: Deploy

```bash
just up
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
| `kafka-init` | `apache/kafka:4.2.0` | - | One-shot: creates the `_schemas` and results topics, then exits |
| `schema-registry` | `confluentinc/cp-schema-registry:8.2.0` | 8081 | Avro schema management |
| `redpanda-connect` | `docker.redpanda.com/redpandadata/connect` | 4195 (HTTP API: health, metrics) | S3 sink streaming from Kafka to object storage (`S3_ENDPOINT`) |

Kafka runs in KRaft mode (no ZooKeeper). Schema Registry manages Avro schemas.
Redpanda Connect replaces the older Kafka Connect + S3 Sink connector setup, streaming
data from Kafka topics directly to object storage (the compose Garage by default).

#### Kafka record size

A converged run streams its result as one record with a row per energy
evaluation. That record often reaches tens of MB. Kafka default of 1 MiB would
reject it, so the broker accepts records up to 136 MiB and requests up to 256 MiB.

The producer sends up to 128 MiB: `max_request_size` and `buffer_memory` both
default to that, and the broker allows 8 MiB more for batch overhead. The limits
are set with `KAFKA_MESSAGE_MAX_BYTES`, `KAFKA_REPLICA_FETCH_MAX_BYTES` and
`KAFKA_SOCKET_REQUEST_MAX_BYTES`; see
[Environment Variables](environment-variables.md#kafka-configuration). A broker
keeps its old limits until its container is recreated.

Redpanda Connect's Kafka client rejects any single broker response above 100 MiB
and author is not aware of any settings for it, so the sink cannot write records
approaching that size to storage.

#### Kafka topics and data directory

A one-shot `kafka-init` service creates the two topics before anything connects:
`_schemas` with the `compact` cleanup policy the Schema Registry requires, and the
results topic. Without it the first client to touch `_schemas` would create it with the
default `delete` policy and the registry would refuse to start. The broker also keeps its
log directory on the `kafka-data` volume (`KAFKA_LOG_DIRS`), so topics survive a recreate
of the container; the image's default directory is inside the container and is lost
with it.

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
Schema Registry, and Garage being healthy. The connector writes to the compose
Garage (`http://garage:3901`) whatever `S3_ENDPOINT` says, so it does not work
with external S3 storage.

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

The object store holds raw VQE results, feature datasets, Iceberg warehouse data
and MLflow artifacts. Buckets are created during `just setup`. With the default
`S3_ENDPOINT` this is the compose Garage, reachable at `http://garage:3901` from
within the Docker network. Its admin API on port 3903 exposes cluster status and
metrics that can be scraped by Prometheus.

#### External S3 storage

`S3_ENDPOINT` in `.env` can name any S3 service instead of the compose Garage.
`just setup` checks that the endpoint answers and creates the four buckets there
with the key from `.env`, which must be allowed to create buckets. If the
endpoint does not answer, setup points `S3_ENDPOINT` back at the compose Garage
and configures that instead.

The Spark configuration, Airflow, the Redpanda Connect sink and MLflow all read
`S3_ENDPOINT` and `S3_REGION`. The compose Garage is still started by `just up`
(Spark depends on it) but nothing uses it, and its configuration file is not
generated in this mode. Stop it with
`docker compose --env-file .env -f compose/docker-compose.ml.yaml stop garage`,
or start only the services you need by name. The
[Kafka Connect alternative](#alternative-confluent-kafka-connect) always writes to
the compose Garage.

### ML Tracking

| Service | Image | Port(s) | Description |
|---|---|---|---|
| `mlflow` | `ghcr.io/mlflow/mlflow:v3.10.1-full` | 5000 | Experiment tracking and artifact storage |

MLflow uses PostgreSQL for the backend store and the object store at
`S3_ENDPOINT` for artifact storage (`s3://mlflow-artifacts/`).

### Monitoring Exporters

These services expose Prometheus metrics for scraping by an external monitoring server:

| Service | Image | Port(s) | Description |
|---|---|---|---|
| `airflow-statsd-exporter` | `prom/statsd-exporter` | 9102 (metrics), 9125/udp (StatsD) | Converts Airflow StatsD metrics to Prometheus format |
| `postgres-exporter` | `prometheuscommunity/postgres-exporter` | 9187 | PostgreSQL metrics |
| `redis-exporter` | `oliver006/redis_exporter` | 9121 | Redis metrics |
| `nvidia-gpu-exporter` | `utkuozdemir/nvidia_gpu_exporter:1.2.0` | 9835 | NVIDIA GPU utilization, memory, temperature |

Some of the images above float on an unpinned or `latest` tag: Redpanda Connect and
the StatsD, PostgreSQL, and Redis exporters. A redeploy can pull a newer image and
change behavior, so pin these to specific versions before any long-running data
generation.

### Batch Simulation Containers

These containers are defined in the `batch` profile and are not started by `just up`.
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
simulation data (`data/`), and output (`gen/`). The Spark and Garage files are
rendered by `just setup` from their `.template` siblings, so the S3 endpoint and
region in them follow `S3_ENDPOINT` and `S3_REGION` from `.env`. Without the
rendered `compose/spark-defaults.conf`, Docker mounts an empty directory in its
place and Spark does not start.

## Justfile Commands

| Command | Description |
|---|---|
| `just setup` | First-time setup: `.env` file, secrets, Spark config, object storage and buckets |
| `just up` | Start the compose stack |
| `just down` | Force-remove running batch containers, then stop the stack (includes `--profile batch`) |
| `just logs [service]` | Tail compose logs (all services, or a specific one) |
| `just docker-build cpu` | Build CPU simulation image |
| `just docker-build gpu` | Build GPU simulation image |
| `just docker-build all` | Build both CPU and GPU images |

## Scaling

Scale Spark workers with:

```bash
docker compose --env-file .env -f compose/docker-compose.ml.yaml up -d --scale spark-worker=3
```

For additional simulation instances, add new service definitions with a unique
`container_name`, appropriate GPU device assignment, and separate output volume
mounts. Instances publish to the `experiment.vqe` topic by default, which is the
only topic the Redpanda Connect sink reads.

## Stopping the Deployment

```bash
# Stop services, preserve data (also force-removes any running batch containers)
just down

# Or manually, including volume cleanup
docker compose --env-file .env -f compose/docker-compose.ml.yaml --profile batch down -v
```
