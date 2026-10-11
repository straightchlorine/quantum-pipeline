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
to `6.1 7.0 8.0 9.0+PTX` (Pascal to Hopper, Blackwell via PTX). Set it to one
value for a faster local build. Requires
`nvidia-container-toolkit` on the host.

```bash
docker build -f docker/Dockerfile.gpu -t quantum-pipeline:gpu .
# Faster local build for one GPU (see nvidia-smi --query-gpu=compute_cap):
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

- **`docker-compose.ml.yaml`** - Full ML data platform: Kafka,
    Schema Registry, Redpanda Connect, Spark (master + worker),
    Airflow (apiserver, scheduler, dag-processor, worker, triggerer),
    MLflow, Garage (S3-compatible storage), Postgres, Redis, and VQE
    simulation containers (CPU + 2x GPU). Includes monitoring exporters
    (statsd, postgres, redis, nvidia-gpu).
- **`docker-compose.ml.kafka-connect.yaml`** - The same platform as above,
    with kafka connect replacing Redpanda Connect.

See the [deployment docs](https://docs.qp.piotrkrzysztof.dev/deployment/)
for setup instructions.
