# Docker

Container images for running simulations and the data platform.

## Images

### `Dockerfile.cpu`

CPU-only simulation container. Runs VQE on the Aer statevector simulator.

```bash
docker build -f docker/Dockerfile.cpu -t quantum-pipeline:cpu .
```

### `Dockerfile.gpu`

GPU simulation container with CUDA support for Aer. `CUDA_ARCH` defaults
to 8.6 (Ampere) and can be overridden at build time. Requires
`nvidia-container-toolkit` on the host.

```bash
docker build -f docker/Dockerfile.gpu -t quantum-pipeline:gpu .
# Override for your GPU architecture:
docker build -f docker/Dockerfile.gpu --build-arg CUDA_ARCH="8.9" -t quantum-pipeline:gpu .
```

### `Dockerfile.spark`

Spark worker and master nodes for the feature processing pipeline.
Used by `docker-compose.ml.yaml`.

```bash
docker build -f docker/Dockerfile.spark -t quantum-pipeline-spark .
```

### `docker/airflow/Dockerfile`

Airflow container with docker-ce-cli, buildx, rclone, and the DAG
scripts pre-installed. Used by `docker-compose.ml.yaml`.

```bash
docker build -f docker/airflow/Dockerfile -t quantum-pipeline-airflow .
```

## Compose stacks

The compose files live in `compose/`:

- **`docker-compose.ml.yaml`** — Full ML data platform: Kafka,
  Schema Registry, Redpanda Connect, Spark (master + worker),
  Airflow (apiserver, scheduler, dag-processor, worker, triggerer),
  MLflow, Garage (S3-compatible storage), Postgres, Redis, and VQE
  simulation containers (CPU + 2x GPU). Includes monitoring exporters
  (statsd, postgres, redis, nvidia-gpu).
- **`docker-compose.ml.kafka-connect.yaml`** — Redpanda Connect
  configuration for routing Kafka topics to S3/Garage.

See the [deployment docs](https://docs.qp.piotrkrzysztof.dev/deployment/)
for setup instructions.
