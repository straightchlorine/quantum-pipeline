# Docker Basics

Quantum Pipeline provides pre-built Docker images for CPU and GPU workloads, as well
as specialized images for Spark and Airflow services. This page covers the available
images, building from source, and running containers.

## Available Images

All images are published to Docker Hub under the
[`straightchlorine/quantum-pipeline`](https://hub.docker.com/r/straightchlorine/quantum-pipeline)
repository.

| Image Tag | Base Image | Purpose |
|---|---|---|
| [`straightchlorine/quantum-pipeline:latest`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/docker/Dockerfile.cpu) | `python:3.12-slim-bullseye` | CPU-only simulations |
| [`straightchlorine/quantum-pipeline:latest-gpu`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/docker/Dockerfile.gpu) | `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04` | GPU-accelerated simulations |
| [`straightchlorine/quantum-pipeline:spark`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/docker/Dockerfile.spark) | `bitnamilegacy/spark` | Spark master and worker nodes |
| [`straightchlorine/quantum-pipeline:airflow`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/docker/Dockerfile.airflow) | `apache/airflow:2.10.5` | Airflow scheduler, webserver, triggerer |

### CPU Image

The CPU image is a lightweight container based on Python 3.12. It installs the
Quantum Pipeline package and its dependencies directly from the project source.

### GPU Image

The GPU image is built on the official NVIDIA CUDA 11.8.0 base with cuDNN 8. It
includes a custom compilation of Qiskit Aer from source with CUDA Thrust backend
support, providing hardware-accelerated quantum circuit simulation.

### Spark Image

The Spark image extends the Bitnami Spark base with pre-downloaded JAR dependencies
for Avro processing, Hadoop AWS (S3A connector), and Apache Iceberg table support.

### Airflow Image

The Airflow image adds Java 17 (required for Spark job submission) and additional
Python dependencies to the official Apache Airflow base image.

## Building from Source

Build instructions are provided in the repository Dockerfiles under [`docker/`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/docker) — see [`Dockerfile.cpu`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/docker/Dockerfile.cpu), [`Dockerfile.gpu`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/docker/Dockerfile.gpu), [`Dockerfile.spark`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/docker/Dockerfile.spark), and [`Dockerfile.airflow`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/docker/Dockerfile.airflow). Each image can be built with:

```bash
docker build -t quantum-pipeline:<variant> -f docker/Dockerfile.<variant> .
```

where `<variant>` is `cpu`, `gpu`, `spark`, or `airflow`.

!!! warning "GPU Build Time"
    Building the GPU image compiles Qiskit Aer from source with CUDA Thrust backend support. To target a different GPU architecture, set `AER_CUDA_ARCH` in the Dockerfile (e.g., `7.5` for Turing, `8.0` for Ampere).

See the [Dockerfiles in the repository](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/docker) for full build details and configuration options.

## Running Containers

### CPU Simulation

Run a basic CPU simulation with:

```bash
docker run --rm \
  -v $(pwd)/data:/usr/src/quantum_pipeline/data \
  -v $(pwd)/gen:/usr/src/quantum-pipeline/gen \
  straightchlorine/quantum-pipeline:latest \
  --file ./data/molecules.json \
  --simulation-method statevector \
  --max-iterations 150 \
  --convergence
```

### GPU Simulation

For GPU-accelerated simulation, pass the `--gpus` flag and add the `--gpu` argument:

```bash
docker run --rm --gpus all \
  -v $(pwd)/data:/usr/src/quantum_pipeline/data \
  -v $(pwd)/gen:/usr/src/quantum-pipeline/gen \
  straightchlorine/quantum-pipeline:latest-gpu \
  --file ./data/molecules.json \
  --gpu \
  --simulation-method statevector \
  --max-iterations 150 \
  --convergence
```

### Connecting to Kafka

To send results to a Kafka broker, add the `--kafka` flag and set the
`KAFKA_SERVERS` environment variable:

```bash
docker run --rm --gpus all \
  --network quantum-pipeline-network \
  -e KAFKA_SERVERS=kafka:9092 \
  -v $(pwd)/data:/usr/src/quantum_pipeline/data \
  straightchlorine/quantum-pipeline:latest-gpu \
  --file ./data/molecules.json \
  --gpu \
  --kafka \
  --simulation-method statevector \
  --max-iterations 150 \
  --convergence
```

### Common Options

| Flag | Description |
|---|---|
| `--file <path>` | Path to molecule definition file (JSON) |
| `--gpu` | Enable GPU acceleration |
| `--kafka` | Enable Kafka output |
| `--topic <name>` | Kafka topic name for results |
| `--simulation-method <method>` | Simulation method (`statevector`, `automatic`) |
| `--max-iterations <n>` | Maximum VQE optimizer iterations |
| `--convergence` | Enable convergence threshold |
| `--threshold <value>` | Convergence threshold value (default: `1e-6`) |
| `--optimizer <name>` | Optimizer algorithm (e.g., `L-BFGS-B`, `COBYLA`) |
| `--basis <set>` | Basis set (e.g., `sto3g`, `cc-pvdz`) |
| `--log-level <level>` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `--report` | Generate a report after simulation |
| `--enable-performance-monitoring` | Enable resource monitoring |
| `--performance-pushgateway <url>` | Prometheus PushGateway URL |
