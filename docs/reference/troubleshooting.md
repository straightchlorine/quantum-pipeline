---
title: "Troubleshooting Docker, GPU, Kafka, Spark"
description: "Fixes for common install, simulation, Docker, GPU, Kafka, Spark, Airflow and monitoring problems, listed by symptom and solution, with a cause where known."
---

# Troubleshooting

This page provides solutions to common issues encountered when installing, configuring, and running the Quantum Pipeline. Problems are grouped by category, with a symptom, a solution and a cause where it is known.

## Installation Issues

### Unsupported Python Version

**Symptom:** `pip install quantum-pipeline` fails with an error stating the package requires a different Python version, or imports fail after install.

**Cause:** The active interpreter is outside the supported range.

**Solution:** Install into a supported Python version. Tools such as `uv`, `conda`, or `pyenv` make it easy to pin the right interpreter. See the [Installation](../getting-started/installation.md) guide for the supported versions and detailed setup.

### GPU Container Fails to Start

**Symptom:** A GPU container exits immediately with `could not select device driver "nvidia"` or `unknown or invalid runtime name: nvidia`.

**Cause:** The NVIDIA Container Toolkit is not installed, or the Docker daemon is not configured with the `nvidia` runtime.

**Solution:** Install the NVIDIA Container Toolkit on the host and restart Docker. See [GPU Acceleration](../deployment/gpu-acceleration.md) for the full driver and runtime setup.

## Simulation Issues

### VQE Convergence Failure

**Symptom:** The simulation completes but reports that convergence was not achieved. The optimizer reached the maximum iteration limit without finding a minimum.

**Cause:** The default iteration limit may be too low for complex molecules, or the initial parameters are stuck in a barren plateau region.

**Solution:**

```bash
# Increase the maximum number of iterations
quantum-pipeline -f data/molecules.json --max-iterations 500

# Try Hartree-Fock initialization; with EfficientSU2 it needs 3 or more repetitions
quantum-pipeline -f data/molecules.json --init-strategy hf --ansatz-reps 3

# Try a different optimizer
quantum-pipeline -f data/molecules.json --optimizer COBYLA

# Loosen convergence tolerance for faster (less precise) convergence
quantum-pipeline -f data/molecules.json --convergence --threshold 1e-4
```

### Run Exits with Code 2 and One Error Line

**Symptom:** The command exits with code 2 after printing a single error line.

**Cause:** A run without `--shots` is [exact](../usage/configuration.md#-exact): expectation values come from the statevector, with no shot sampling. A gradient optimizer (the default L-BFGS-B included) needs those exact values, so it refuses sampled shots. Sampling can still start without `--shots`: `--ibm` samples 1024 shots by default, and a `--load` file can store shots. A `--noise` run on any method other than `density_matrix` also needs `--shots N`.

**Solution:** Drop `--shots`, or switch to a gradient-free optimizer such as COBYLA. For a noisy run, pass `--simulation-method density_matrix` or `--shots N`.

### Out of Memory During Simulation

**Symptom:** The process is killed with `MemoryError` or the OOM killer terminates the container during simulation of large molecules.

**Cause:** Large molecules (12+ qubits) require significant memory for statevector simulation. Memory usage scales exponentially with qubit count.

**Solution:**

```bash
# For Docker deployments, increase container memory limit
docker run --memory=16g straightchlorine/quantum-pipeline:latest ...

# Use a simpler basis set to reduce qubit count
quantum-pipeline -f data/molecules.json --basis sto3g

# Reduce ansatz repetitions (default is 2, lowering to 1 cuts parameter count)
quantum-pipeline -f data/molecules.json --ansatz-reps 1
```

### Slow Simulation Performance

**Symptom:** Iterations take significantly longer than expected based on the performance baselines.

**Cause:** The simulation may be running on CPU when GPU acceleration is available, or system resources are contended.

**Solution:**

```bash
# Verify the GPU is used: the log shows "Initializing Aer simulator backend with GPU
# acceleration..." and "Using exact Aer estimator on aer_simulator_statevector_gpu"
quantum-pipeline -f data/molecules.json --gpu --log-level DEBUG

# Ensure no other heavy processes are running
top

# Check that the CUDA runtime is accessible
nvidia-smi
```

### Incorrect Energy Values

**Symptom:** The final energy value is significantly higher than expected reference values.

**Cause:** Random parameter initialization may have led to convergence on a local minimum rather than the global minimum. This is especially common for molecules with complex optimization landscapes.

**Solution:**

Run multiple simulations with different seeds, or use `--init-strategy hf` to start from the Hartree-Fock state.
In one H2/6-31g run with L-BFGS-B, random init spent over 1000 iterations and ended at a positive energy,
while HF init reached -1.857 Ha in 50 iterations.
With `hf` at the default 2 repetitions a gradient optimizer can stop at the Hartree-Fock energy; see [Hartree-Fock initialization](../scientific/vqe-algorithm.md#hartree-fock-initialization).

## Docker Issues

### Docker Socket Permission Denied (GID Mismatch)

**Symptom:** Airflow or batch generation containers fail with `permission denied` when trying to access `/var/run/docker.sock`.

**Cause:** The `DOCKER_GID` build argument does not match the Docker group ID on the host. The Airflow container needs this GID to access the Docker daemon for launching simulation containers.

**Solution:**

```bash
# Find the Docker group ID on your host
getent group docker | cut -d: -f3

# Rebuild with the correct GID (e.g., if your Docker GID is 999)
DOCKER_GID=999 docker compose build airflow-worker

# Or set it in your .env file
echo "DOCKER_GID=999" >> .env
```

The default `DOCKER_GID` is `970`. If your host uses a different value, the container's `airflow` user will not be in the right group and Docker socket access will fail.

### Volume Mount Permissions

**Symptom:** Permission denied errors when containers try to write to mounted volumes.

**Cause:** The container user does not have write access to the host directory.

**Solution:** Pre-create the mounted directories on the host as your own user, and/or run with `--user "$(id -u):$(id -g)"`. The images work under any user.

### Stale Docker State

**Symptom:** Configuration changes are not taking effect.

**Solution:** Rebuild without cache (`docker compose build --no-cache`) and restart with `docker compose down -v && docker compose up -d`.

## GPU Issues

### Wrong CUDA_ARCH for Your GPU

**Symptom:** Simulations crash with `CUDA error: no kernel image is available` or fall back to CPU.

**Cause:** The default image covers Pascal 6.1 and newer. This only happens with a custom single-arch `CUDA_ARCH` build for a different card, or a GPU older than 6.1 (such as the P100, 6.0).

**Solution:** Use the default image, or rebuild with your card's compute capability:

```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
CUDA_ARCH=7.5 just docker-build gpu
```

If you are unsure, check the [NVIDIA CUDA GPUs page](https://developer.nvidia.com/cuda-gpus) for your card's compute capability.

### CUDA Not Available

A `--gpu` run that stops with `No CUDA device available!` was started without `--gpus`, or the host lacks the NVIDIA Container Toolkit or a recent enough driver. One that stops with `no kernel image is available` runs an image not compiled for this GPU. See [GPU Acceleration](../deployment/gpu-acceleration.md) for the requirements and the `CUDA_ARCH` table.

### GPU Out of Memory

**Symptom:** `CUDA out of memory. Tried to allocate X MiB` error during simulation.

**Cause:** The molecule requires more GPU memory than is available on the device.

**Solution:**

```bash
# Check GPU memory usage
nvidia-smi

# Use a simpler basis set to reduce memory requirements
quantum-pipeline -f data/molecules.json --basis sto3g

# Fall back to CPU for very large molecules (omit --gpu to use CPU)
quantum-pipeline -f data/molecules.json
```

## Kafka Issues

### Schema Registry Errors

**Symptom:** `SchemaRegistryError: Subject not found` or schema compatibility errors.

**Solution:** Check health with `curl http://localhost:8081/subjects`. For development, reset compatibility with:

```bash
curl -X PUT http://localhost:8081/config \
    -H "Content-Type: application/json" \
    -d '{"compatibility": "NONE"}'
```

### Message Delivery Failures

**Symptom:** Messages produced but never appear in object storage.

**Solution:** Check Redpanda Connect health with `curl http://localhost:4195/ping`. Check Garage admin status on port 3903. Review logs with `docker compose logs redpanda-connect | tail -50`.

### Topic Not Created

**Symptom:** Messages fail to publish because the target topic does not exist.

**Solution:**

```bash
# List existing topics
docker compose exec kafka kafka-topics --list --bootstrap-server localhost:9092

# Create topic manually
docker compose exec kafka kafka-topics --create \
    --topic experiment.vqe \
    --bootstrap-server localhost:9092 \
    --partitions 3 \
    --replication-factor 1
```

## Spark Issues

### Spark Job Out of Memory

**Symptom:** Spark job fails with `java.lang.OutOfMemoryError: Java heap space` or executor lost errors.

**Cause:** The Spark workers do not have enough memory to process the data volume.

**Solution:**

```bash
# Increase Spark executor memory in docker-compose
environment:
  SPARK_WORKER_MEMORY: 8g

# Or set memory in the Spark configuration
spark.executor.memory=4g
spark.driver.memory=2g
```

### Spark Job Submission Failure

**Symptom:** Airflow task fails with `SparkSubmitOperator` errors, or the job never appears in the Spark UI.

**Solution:** Verify Spark master is accessible (`curl http://spark-master:8080`). Check the Airflow connection at **Admin > Connections > spark_default** (Host: `spark://spark-master`, Port: `7077`).

### S3A Connection Errors

**Symptom:** Spark fails with `com.amazonaws.SdkClientException: Unable to execute HTTP request` when reading from or writing to object storage.

**Cause:** The S3 endpoint configuration is incorrect, or credentials are missing.

**Solution:**

The endpoint and region in `compose/spark-defaults.conf` are filled in from
`S3_ENDPOINT` and `S3_REGION` in `.env` when `just setup` renders the file. After
changing either, run `just setup` again and recreate the Spark and Airflow
containers. Verify that the following Spark configuration values match your
object store:

```properties
spark.hadoop.fs.s3a.endpoint = http://garage:3901
spark.hadoop.fs.s3a.access.key = <your-access-key>
spark.hadoop.fs.s3a.secret.key = <your-secret-key>
spark.hadoop.fs.s3a.path.style.access = true
spark.hadoop.fs.s3a.connection.ssl.enabled = false
```

Also verify that Garage is running:

```bash
# Check Garage container status
docker compose ps garage
```

## Airflow Issues

### DAG Not Visible in Web UI

**Symptom:** The [`quantum_feature_processing`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/docker/airflow/quantum_processing_dag.py) DAG does not appear in the Airflow web interface.

**Solution:** Verify the DAG file exists (`docker compose exec airflow-apiserver ls /opt/airflow/dags/`), check for import errors in **Admin > DAG Import Errors**, and force a rescan with `docker compose exec airflow-scheduler airflow dags reserialize`.

### Task Failures with Retry Exhaustion

**Symptom:** Tasks fail repeatedly and exhaust all retries.

**Solution:** Check task logs in the Airflow UI (**DAG > Task Instance > Logs**), verify dependent services with `docker compose ps`, then clear the failed task:

```bash
docker compose exec airflow-apiserver airflow tasks clear \
    quantum_feature_processing -t run_quantum_processing \
    -s 2025-01-01 -e 2025-12-31 --yes
```

### Database Connection Errors

**Symptom:** Airflow fails to start with `sqlalchemy.exc.OperationalError: could not connect to server`.

**Solution:** Check PostgreSQL status (`docker compose ps postgres`), then initialize with `docker compose exec airflow-apiserver airflow db migrate`.

## Monitoring Issues

### PushGateway Not Reachable

**Symptom:** `ConnectionError: HTTPConnectionPool host='pushgateway' port=9091` in simulation logs, or metrics silently not appearing.

**Cause:** PushGateway is not running (the Compose stack does not include it, so you run it separately), the `PUSHGATEWAY_URL` environment variable points to the wrong address, or there is a network mismatch between the simulation container and PushGateway.

**Solution:**

```bash
# Check PushGateway is running (it is run separately from the Compose stack)
docker ps | grep -i pushgateway
curl -s http://localhost:9091/metrics | head -5

# Check the configured URL in a batch service
docker compose exec quantum-pipeline-cpu env | grep PUSHGATEWAY

# Common fixes:
# - From the host, use http://localhost:9091
# - From Docker containers, use an address that reaches your PushGateway
#   (for example its container name on a shared network)
# - Make sure MONITORING_ENABLED is set to true
```

### Metrics Not Appearing in Prometheus

**Symptom:** The Grafana dashboard shows "No data" for all panels. Prometheus queries return empty results.

**Cause:** The PushGateway is not receiving metrics, or Prometheus is not scraping the PushGateway.

**Solution:**

```bash
# Check PushGateway has received metrics
curl http://localhost:9091/metrics | head -50

# Check Prometheus targets
# Navigate to: http://localhost:9090/targets
# Verify pushgateway target shows "UP"

# Verify the simulation was started with monitoring enabled
# Either set the environment variable:
export MONITORING_ENABLED=true

# Or pass the CLI flag:
quantum-pipeline -f data/molecules.json --enable-performance-monitoring
```

### Grafana Cannot Connect to Prometheus

**Symptom:** Grafana data source test fails with "Connection refused".

**Solution:** Use `http://prometheus:9090` from within Docker, or `http://localhost:9090` from the host. Verify via **Configuration > Data Sources > Prometheus > Save & Test**.

### Dashboard Import Fails

**Symptom:** Importing the thesis dashboard JSON fails with a validation error.

**Solution:** The dashboard was created with Grafana 10.x. Ensure the Prometheus data source is named `Prometheus` (the default) or update the `DS_PROMETHEUS` variable in the dashboard JSON to match your data source name.

## Batch Generation Issues

### Container Exits with rc=125

**Symptom:** Batch generation reports a container failure with return code 125.

**Cause:** The Docker image was not found. This usually means the image has not been built locally or the image name in the batch configuration does not match.

**Solution:**

```bash
# Build the required images
just docker-build cpu
just docker-build gpu

# Verify images exist
docker images | grep quantum-pipeline
```

### Container Exits with rc=1

**Symptom:** Batch generation reports a container failure with return code 1.

**Cause:** The simulation application itself encountered an error. This could be an invalid molecule/optimizer/basis combination, out-of-memory, or a bug.

**Solution:**

```bash
# Check the container logs for the failed run
docker logs <container-id>

# Look at the batch state file for details
cat gen/ml_batch_state.json | python -m json.tool
```

The batch system is idempotent - rerunning the generation script will skip completed configurations and retry failed ones. Trigger via the `vqe_batch_generation` Airflow DAG or directly with `python scripts/generate_ml_batch.py`.

### Batch Progress Not Visible

**Symptom:** Batch generation is running but `qp_batch_*` metrics do not appear in Grafana.

**Cause:** The batch script pushes progress metrics to the PushGateway. If the PushGateway is not reachable from the host (where the batch script runs), metrics will not be recorded.

**Solution:**

Make sure the PushGateway is running and accessible from the host at `http://localhost:9091`. The batch script runs on the host, not inside a container, so it needs host-level access to the PushGateway.

If your issue is not covered here, check the [FAQ](faq.md) or open an issue on the [Codeberg repository](https://codeberg.org/piotrkrzysztof/quantum-pipeline/issues).
