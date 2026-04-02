# Docker Basics

Quantum Pipeline provides Docker images for CPU and GPU workloads, as well
as images for Spark and Airflow services. This page covers the available
images, building from source, and running containers.

## Available Images

All images are published to Docker Hub under the
[`straightchlorine/quantum-pipeline`](https://hub.docker.com/r/straightchlorine/quantum-pipeline)
repository.

| Image Tag | Base Image | Purpose |
|---|---|---|
| [`straightchlorine/quantum-pipeline:cpu`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/docker/Dockerfile.cpu) | `python:3.12-slim-bookworm` | CPU-only simulations |
| [`straightchlorine/quantum-pipeline:gpu`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/docker/Dockerfile.gpu) | `nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04` | GPU-accelerated simulations |
| [`quantum-pipeline-spark`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/docker/Dockerfile.spark) | `apache/spark:4.0.2-python3` | Spark master and worker nodes |
| [`airflow` (custom)](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/docker/airflow/Dockerfile) | `apache/airflow:3.1.8` | Airflow API server, scheduler, workers, triggerer |

### CPU Image

The CPU image is a lightweight container based on Python 3.12 (Bookworm). It installs
the Quantum Pipeline package and its dependencies directly from the project source.

### GPU Image

The GPU image is built on NVIDIA CUDA 12.6.3 with cuDNN. It includes a custom
compilation of Qiskit Aer from source with CUDA Thrust backend support, providing
hardware-accelerated quantum circuit simulation.

The `CUDA_ARCH` build argument controls which GPU architecture the Aer binary targets:

| `CUDA_ARCH` | Architecture | Example GPUs |
|---|---|---|
| `6.1` | Pascal | GTX 1060, GTX 1050 Ti |
| `7.5` | Turing | RTX 2070, RTX 2080 |
| `8.6` | Ampere (default) | RTX 3060, RTX 3080 |
| `8.9` | Ada Lovelace | RTX 4070, RTX 4090 |

### Spark Image

The Spark image extends the Apache Spark 4.0.2 base by replacing Python 3.10 with
3.12 so it matches the Airflow driver environment. The image
itself contains no application code or JARs - those are resolved at runtime via
`spark.jars.packages` in `spark-defaults.conf` and cached in a named Ivy volume.

### Airflow Image

The Airflow image adds Java 17 (required for Spark job submission), Docker CE CLI
with buildx and compose plugins (for batch generation DAGs that spawn simulation
containers), rclone (for R2 sync), and Python dependencies (`apache-airflow-providers-apache-spark`,
`pyspark`) to the Apache Airflow 3.1.8 base image.

The `DOCKER_GID` build argument sets the GID of the `docker` group inside the
container so the Airflow user can access the host Docker socket. Check your host
GID with `stat -c '%g' /var/run/docker.sock` and pass it at build time if it
differs from the default (`970`).

## Building from Source

### Using the justfile (recommended)

The simplest way to build images:

```bash
# Build CPU image
just docker-build cpu

# Build GPU image (default CUDA_ARCH=8.6/Ampere)
just docker-build gpu

# Build GPU image for Pascal GPUs
CUDA_ARCH=6.1 just docker-build gpu

# Build both CPU and GPU images
just docker-build all
```

### Manual builds

Each image can be built directly with Docker:

```bash
# CPU
docker build -t quantum-pipeline:cpu -f docker/Dockerfile.cpu .

# GPU (default Ampere)
docker build -t quantum-pipeline:gpu -f docker/Dockerfile.gpu .

# GPU targeting Pascal
docker build -t quantum-pipeline:gpu -f docker/Dockerfile.gpu \
  --build-arg CUDA_ARCH="6.1" .

# Airflow (with custom Docker GID)
docker build -t quantum-pipeline-airflow -f docker/airflow/Dockerfile \
  --build-arg DOCKER_GID="$(stat -c '%g' /var/run/docker.sock)" \
  docker/airflow/
```

!!! warning "GPU Build Time"
    Building the GPU image compiles Qiskit Aer from source with CUDA Thrust backend
    support. This takes a while. Make sure `CUDA_ARCH` matches your target GPU!

## Running Containers

### CPU Simulation

Run a basic CPU simulation with:

```bash
docker run --rm \
  -v $(pwd)/data:/usr/src/quantum_pipeline/data \
  -v $(pwd)/gen:/usr/src/quantum_pipeline/gen \
  quantum-pipeline:cpu \
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
  -v $(pwd)/gen:/usr/src/quantum_pipeline/gen \
  quantum-pipeline:gpu \
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
  --network quantum-ml-network \
  -e KAFKA_SERVERS=kafka:9092 \
  -v $(pwd)/data:/usr/src/quantum_pipeline/data \
  quantum-pipeline:gpu \
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
| `--ansatz <type>` | Ansatz type (`EfficientSU2`, `RealAmplitudes`, `ExcitationPreserving`) |
| `--seed <n>` | Random seed for reproducible parameter initialization |
| `--init-strategy <strategy>` | Parameter initialization (`random`, `hf`) |
| `--log-level <level>` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `--report` | Generate a report after simulation |
| `--enable-performance-monitoring` | Enable resource monitoring |
