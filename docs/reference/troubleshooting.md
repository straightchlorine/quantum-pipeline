---
title: "Troubleshooting Docker, GPU, Kafka, Spark"
description: "Fixes for common install, simulation, Docker, GPU, Kafka, Spark, Airflow and monitoring problems, listed by symptom and solution, with a cause where known."
---

# Troubleshooting

Fixes for common problems when installing, configuring and running Quantum Pipeline,
grouped by component. Each entry lists the symptom, the cause where known, and the fix.

## Installation Issues

### Unsupported Python Version

**Symptom:** `pip install quantum-pipeline` fails with a Python version error, or imports
fail after installation.

**Cause:** The active interpreter is outside the supported range: Python 3.10 to 3.14 for
the package (`3.11+` for the `ml` extra, `3.12+` when installing from source with PDM or
in the Docker images).

**Solution:** Install into a supported Python version. PDM, uv, conda or pyenv can pin the
interpreter. See [Installation](../getting-started/installation.md) for the full setup.

### GPU Container Fails to Start

**Symptom:** A GPU container exits immediately with
`could not select device driver "nvidia"` or `unknown or invalid runtime name: nvidia`.

**Cause:** The NVIDIA Container Toolkit is not installed on the host, or the Docker daemon
was not restarted after installing it.

**Solution:** Install the NVIDIA Container Toolkit and restart Docker. See
[GPU Acceleration](../deployment/gpu-acceleration.md) for the driver and runtime setup.

## Simulation Issues

### VQE Convergence Failure

**Symptom:** The log shows `VQE stopped after N iterations - convergence not achieved` or
`Hard evaluation limit reached after N function evaluations`, and the result records
`success=False`.

**Cause:** The evaluation budget (`--max-iterations`, default `100`, or
`--max-evaluations`, default `30000` under `--convergence`) ran out before the optimizer
met its tolerance, or random initialization landed in a poor local minimum or flat region.
On the Aer path (`--gpu`, `--noise`, or more than 16 qubits), a gradient optimizer spends
about one evaluation per parameter on each finite-difference step, so 100 evaluations is
only a few steps.

**Solution:**

```bash
# Raise the evaluation limit, or let the convergence threshold stop the run
quantum-pipeline -f data/molecules.json --max-iterations 500
quantum-pipeline -f data/molecules.json --convergence --threshold 1e-4

# Start from the Hartree-Fock state (use 3+ reps with EfficientSU2)
quantum-pipeline -f data/molecules.json --init-strategy hf --ansatz-reps 3

# Switch to a gradient-free optimizer
quantum-pipeline -f data/molecules.json --optimizer COBYLA
```

### Run Exits with Code 2 and One Error Line

**Symptom:** `quantum-pipeline` prints the usage text and a single `error:` line, then
exits with code 2 before running any simulation.

**Cause:** An invalid combination of CLI flags (or values in a `--load` config file, which
override CLI flags). Common cases:

- **Gradient optimizer with sampled shots.** A run without `--shots` is
  [exact](../usage/configuration.md#-exact): expectation values come from the statevector,
  with no shot noise. Gradient optimizers (including the default `L-BFGS-B`) need exact
  values and refuse sampled shots. Sampling can be active even when you did not pass
  `--shots`: `--ibm` defaults to 1024 shots, and a `--load` file can store `"shots"`.
- **`Newton-CG` outside the analytic path.** `Newton-CG` requires an analytic gradient,
  which is only available on an exact, local, noiseless CPU run with at most 16 qubits and
  without `--no-analytic`.
- **Exact noisy run without `density_matrix`.** A `--noise` run without `--shots` needs
  `--simulation-method density_matrix`.
- **Missing parent flag.** `--simulation-method tensor_network` requires `--gpu`,
  `--min-qubits` requires `--ibm`, and any Kafka, SSL or SASL option requires `--kafka`.

**Solution:** Drop `--shots` (or set `"shots": null` in the loaded config file), or pick a
gradient-free optimizer such as `COBYLA`. For `--noise`, pass
`--simulation-method density_matrix` or `--shots N` with `COBYLA`. See
[Validation Rules](../usage/configuration.md#validation-rules) for the complete list.

### Out of Memory During Simulation

**Symptom:** Python exits with `MemoryError`, or the Linux OOM killer terminates the
process or container on larger molecules.

**Cause:** Statevector memory doubles with each qubit ($2^n \times 16$ bytes, and
$2^{2n} \times 16$ bytes for `density_matrix`). Larger basis sets (`6-31g`, `cc-pvdz`)
add orbitals and qubits, and the analytic gradient path builds a sparse Hamiltonian that
becomes memory-heavy past 16 qubits (where the solver switches to Aer automatically).

**Solution:**

```bash
# Raise the container memory limit
docker run --rm --memory=16g straightchlorine/quantum-pipeline:cpu \
    -f data/molecules.json --basis sto3g

# Use a smaller basis set to reduce the qubit count
quantum-pipeline -f data/molecules.json --basis sto3g

# Use the matrix product state method for memory-constrained CPU runs
quantum-pipeline -f data/molecules.json --simulation-method matrix_product_state

# Lower ansatz repetitions (default 2) to cut parameter count and trajectory size
quantum-pipeline -f data/molecules.json --ansatz-reps 1
```

### Slow Simulation Performance

**Symptom:** Optimization steps take much longer than expected.

**Cause:** Two common causes work in opposite directions:

1. **Up to 16 qubits with a gradient optimizer:** a local, exact, noiseless CPU run takes
   the [analytic statevector path](../usage/configuration.md#-exact), which computes the
   energy and its exact gradient in one pass. Adding `--gpu` turns the analytic path off
   and switches to Aer with finite-difference gradients (about one evaluation per
   parameter per step), which is slower at small qubit counts.
2. **Larger circuits or Aer runs:** above 12 to 14 qubits on the Aer path (and above 16
   qubits overall), CPU statevector simulation is much slower than GPU.

**Solution:** For exact gradient runs up to 16 qubits, omit `--gpu` so the analytic path
is used (the log shows `Analytic-gradient statevector path: ...`). For larger molecules,
enable `--gpu` and check the log and GPU state:

```bash
# Confirm the GPU backend is active: the INFO log shows
# "Initializing Aer simulator backend with GPU acceleration..." and
# "Using exact Aer estimator on aer_simulator_statevector_gpu"
quantum-pipeline -f data/molecules.json --gpu --log-level INFO

# Check GPU utilization and competing host processes
nvidia-smi
top
```

### Incorrect Energy Values

**Symptom:** The final energy sits well above the expected ground-state or Hartree-Fock
reference energy.

**Cause:** Random initialization (`--init-strategy random`, the default) often lands in a
poor local minimum on non-convex VQE landscapes. Alternatively, with `--init-strategy hf`
and the default `EfficientSU2` ansatz at 2 repetitions (`--ansatz-reps 2`), the
Hartree-Fock starting point is near-stationary: a gradient optimizer sees a near-zero
gradient and stops at the Hartree-Fock energy without recovering correlation energy.

**Solution:** Try several `--seed` values, or start from Hartree-Fock with at least 3
repetitions (`--init-strategy hf --ansatz-reps 3`) or `--ansatz ExcitationPreserving`. In
one H2/6-31G run with `L-BFGS-B`, `random` spent 1029 iterations and ended at `+2.101` Ha
(electronic), while `hf` reached `-1.857` Ha in 50 iterations. See
[Hartree-Fock Initialization](../scientific/vqe-algorithm.md#hartree-fock-initialization)
for measured stall rates and remedies.

## Docker Issues

### Docker Socket Permission Denied (GID Mismatch)

**Symptom:** The `vqe_batch_generation` DAG in `airflow-worker` fails with
`permission denied` when accessing `/var/run/docker.sock`.

**Cause:** The `DOCKER_GID` build argument (default `970`) does not match the `docker`
socket group ID on the host, so the `airflow` user inside the container cannot talk to the
host Docker daemon.

**Solution:**

```bash
# Check the GID of the Docker socket on the host
stat -c '%g' /var/run/docker.sock

# Set DOCKER_GID in .env (for example 999), then rebuild and recreate Airflow
echo "DOCKER_GID=999" >> .env
docker compose --env-file .env -f compose/docker-compose.ml.yaml build airflow-worker
just up
```

### Volume Mount Permissions

**Symptom:** `PermissionError` when containers write reports, plots or configs to mounted
host directories (`gen/`, `run_configs/`), or missing input files in batch runs.

**Cause:** The host directory is not writable by the user running inside the container, or
`QUANTUM_PIPELINE_HOST_ROOT` in `.env` does not match the repository path on the host when
`airflow-worker` spawns batch containers via the Docker socket.

**Solution:** Pre-create `gen/` on the host as your user and pass
`--user "$(id -u):$(id -g)"` to `docker run` (the CPU and GPU images work under any user).
For the Compose stack, if the repository is not at `$HOME/code/quantum-pipeline`, set
`QUANTUM_PIPELINE_HOST_ROOT` in `.env` to its absolute host path.

### Configuration Changes Not Taking Effect

**Symptom:** Edits to `.env`, `compose/spark-defaults.conf.template` or bind-mounted
config files are not picked up by running services.

**Cause:** `compose/spark-defaults.conf` and `compose/garage.toml` are rendered from their
`.template` files by `just setup`. In addition, Docker Compose hashes the service
definition rather than the contents of bind-mounted files, so `up -d` alone does not
recreate a container whose mounted config file changed.

**Solution:**

```bash
# Re-render compose/spark-defaults.conf and compose/garage.toml after .env edits
just setup

# Recreate containers so updated bind mounts and env vars take effect
docker compose --env-file .env -f compose/docker-compose.ml.yaml up -d --force-recreate

# If a Dockerfile changed, rebuild without cache first
docker compose --env-file .env -f compose/docker-compose.ml.yaml build --no-cache
```

## GPU Issues

### Wrong CUDA_ARCH for Your GPU

**Symptom:** A `--gpu` run exits with
`parallel_for failed: cudaErrorNoKernelImageForDevice: no kernel image is available for execution on the device`
and `RuntimeError: N of N molecules failed`. An architecture that CUDA 12.6 dropped
entirely (such as `3.5`) fails earlier during `docker build` with
`nvcc fatal : Unsupported gpu architecture`.

**Cause:** Qiskit Aer was compiled without kernels for your GPU's compute capability. The
image builds cleanly for any architecture `nvcc` accepts and only fails at runtime when a
kernel launches. The default image covers `6.1 7.0 8.0 9.0+PTX` (Pascal 6.1 and newer), so
this only happens with a custom single-arch `CUDA_ARCH` build for a different card (note
that `.env.ml.example` sets `CUDA_ARCH=6.1`), or on a GPU older than 6.1 (such as the
P100, `6.0`).

**Solution:** Use the published `straightchlorine/quantum-pipeline:gpu` image, or rebuild
for your card's compute capability:

```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
CUDA_ARCH=7.5 just docker-build gpu
```

See the [CUDA_ARCH table](../deployment/gpu-acceleration.md#cuda_arch-build-argument) or
the [NVIDIA CUDA GPUs page](https://developer.nvidia.com/cuda-gpus) for your card's value.

### No CUDA Device Available

**Symptom:** A `--gpu` run fails immediately with `RuntimeError: No CUDA device available!`.
The image never falls back to the CPU.

**Cause:** The container was started without `--gpus all` (or `runtime: nvidia` in
Compose), the NVIDIA Container Toolkit is missing, or the host driver is too old for the
CUDA 12.6 runtime in the image (`525.60.13` minimum for compiled architectures; `560` or
newer when running through PTX on GPUs newer than Hopper 9.0).

**Solution:** Pass `--gpus all` to `docker run`, check the host driver with `nvidia-smi`,
and see [GPU Acceleration](../deployment/gpu-acceleration.md) for the full setup.

### GPU Out of Memory

**Symptom:** The simulation fails on GPU with a CUDA `out of memory` (`std::bad_alloc`)
error or Qiskit Aer's `Insufficient memory to run circuit`.

**Cause:** The circuit exceeds available GPU VRAM, or exceeds the `max_memory_mb` cap in
[`defaults.py`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/configs/defaults.py)
(default `5500` MB, sized for a 6 GB GTX 1060).

**Solution:**

```bash
# Check available GPU memory
nvidia-smi

# Use a smaller basis set to reduce qubit count
quantum-pipeline -f data/molecules.json --basis sto3g

# Omit --gpu to run on host RAM instead
quantum-pipeline -f data/molecules.json --basis 6-31g
```

If your GPU has more than 6 GB of VRAM, raise `max_memory_mb` under `gpu_opts` in
`quantum_pipeline/configs/defaults.py`.

## Kafka Issues

### Schema Registry Errors

**Symptom:** The producer logs `Schema registry is not available!` or
`Failed to publish the schema`, or the `schema-registry` container exits on startup
because `_schemas` has `cleanup.policy=delete`.

**Cause:** Schema Registry is unreachable at `SCHEMA_REGISTRY_URL`
(`http://schema-registry:8081` in Compose), the `_schemas` topic was auto-created by a
client before `kafka-init` set `cleanup.policy=compact`, or a schema change violated the
registry's `BACKWARD` compatibility rule.

**Solution:** Check that the registry responds and inspect registered subjects:

```bash
curl -s http://localhost:8081/subjects

# Recreate _schemas with compact cleanup if kafka-init did not run first
docker compose --env-file .env -f compose/docker-compose.ml.yaml up kafka-init

# Relax compatibility during local development if a schema change is rejected
curl -X PUT http://localhost:8081/config \
    -H "Content-Type: application/vnd.schemaregistry.v1+json" \
    -d '{"compatibility": "NONE"}'
```

### Message Delivery Failures

**Symptom:** Results are written to `gen/undelivered/molecule_<id>.json` and the run exits
with `KafkaProducerError`, or messages reach Kafka but never appear in object storage.

**Cause:** When the broker is unreachable or a serialized record exceeds the 128 MiB
producer cap (`max_request_size` and `buffer_memory`), `VQERunner` spools the result to
`gen/undelivered/` as JSON and exits non-zero. When messages reach Kafka but no files
land in `raw-results`, either `redpanda-connect` cannot reach `S3_ENDPOINT`, the run
published to a `--topic` other than `experiment.vqe` (the only topic the sink reads), or a
converged record exceeded Redpanda Connect's 100 MiB broker response limit.

**Solution:**

```bash
# Check Redpanda Connect health and recent logs
curl -f http://localhost:4195/ping
just logs redpanda-connect

# Check compose Garage status (if using the bundled object store)
docker exec ml-garage /garage status
```

If converged runs produce records near 100 MiB, lower `--max-evaluations` (default
`30000`); see [Kafka record size](../deployment/docker-compose.md#kafka-record-size).

### Topic Not Created

**Symptom:** Publishing or consuming fails because `experiment.vqe` or `_schemas` does not
exist.

**Cause:** The one-shot `kafka-init` service did not complete, or the `kafka-data` volume
was recreated without running it.

**Solution:**

```bash
# Run the one-shot topic initializer
docker compose --env-file .env -f compose/docker-compose.ml.yaml up kafka-init

# Or list and create topics directly in the Kafka container
docker exec ml-kafka /opt/kafka/bin/kafka-topics.sh \
    --bootstrap-server kafka:9092 --list

docker exec ml-kafka /opt/kafka/bin/kafka-topics.sh \
    --bootstrap-server kafka:9092 --create --if-not-exists \
    --topic experiment.vqe --partitions 1 --replication-factor 1
```

## Spark Issues

### Spark Job Out of Memory

**Symptom:** A Spark job fails with `java.lang.OutOfMemoryError: Java heap space` or
`ExecutorLostFailure`.

**Cause:** The data volume exceeds `spark.executor.memory` (`1536m`),
`spark.driver.memory` (`1g`), `SPARK_WORKER_MEMORY` (`2G` in `.env`), or the `3G`
container memory limit on `spark-worker` in `compose/docker-compose.ml.yaml`.

**Solution:** Raise `SPARK_WORKER_MEMORY` in `.env` and the `spark-worker` container memory
limit in `compose/docker-compose.ml.yaml`, then adjust the Spark memory settings in
`compose/spark-defaults.conf.template` (keeping `spark.executor.memory` below
`SPARK_WORKER_MEMORY`) and re-render the config:

```properties
# compose/spark-defaults.conf.template
spark.driver.memory     2g
spark.executor.memory   4g
```

```bash
just setup
just up
```

### Spark Job Submission Failure

**Symptom:** An Airflow task fails in `SparkSubmitOperator`, or the submitted job never
appears in the Spark Master UI (`http://localhost:8080`).

**Cause:** `spark-master` is down, the `spark_default` connection in Airflow was not
created by `airflow-init`, `compose/spark-defaults.conf` was not rendered before `just up`
(so Docker mounted an empty directory in its place), or two runs raced on the shared Ivy
JAR cache (`unresolved dependency`).

**Solution:** Check that the master UI responds at `http://localhost:8080` and that the
Airflow connection under **Admin > Connections > spark_default** points to
`spark://spark-master:7077`. If `compose/spark-defaults.conf` is missing or a directory,
remove it, run `just setup`, and recreate the Spark and Airflow containers. Keep
`max_active_runs=1` on the Spark DAGs so concurrent runs do not race on `/tmp/.ivy2`.

### S3A Connection Errors

**Symptom:** Spark fails with
`com.amazonaws.SdkClientException: Unable to execute HTTP request` or `403 Forbidden` when
reading from or writing to `s3a://`.

**Cause:** `S3_ENDPOINT`, `S3_REGION`, `S3_ACCESS_KEY` or `S3_SECRET_KEY` in `.env` does
not match the object store, `just setup` was not re-run after editing `.env`, or the
object store is down.

**Solution:** `just setup` fills `S3_ENDPOINT` and `S3_REGION` from `.env` into
`compose/spark-defaults.conf`, while `compose/docker-compose.ml.yaml` passes
`S3_ACCESS_KEY` and `S3_SECRET_KEY` to Spark and Airflow as `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY`. After changing any of these in `.env`, re-run `just setup` and
recreate the containers:

```properties
# Rendered in compose/spark-defaults.conf by `just setup`
spark.hadoop.fs.s3a.endpoint                http://garage:3901
spark.hadoop.fs.s3a.endpoint.region         garage
spark.hadoop.fs.s3a.impl                    org.apache.hadoop.fs.s3a.S3AFileSystem
spark.hadoop.fs.s3a.path.style.access       true
spark.hadoop.fs.s3a.connection.ssl.enabled  false
```

```bash
# Re-render spark-defaults.conf, then recreate Spark and Airflow
just setup
docker compose --env-file .env -f compose/docker-compose.ml.yaml up -d --force-recreate \
    spark-master spark-worker airflow-worker airflow-scheduler airflow-apiserver

# If using the compose Garage, verify it is healthy
docker exec ml-garage /garage status
```

## Airflow Issues

### DAG Not Visible in Web UI

**Symptom:** The
[`quantum_feature_processing`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/docker/airflow/quantum_processing_dag.py)
DAG (or another pipeline DAG) does not appear at `http://localhost:8084`.

**Cause:** `airflow-dag-processor` has not parsed the file yet, or hit an import error.

**Solution:**

```bash
# Verify the DAG files are mounted
docker exec ml-airflow-dag-processor ls /opt/airflow/dags/

# Check for import errors and force a reparse
docker exec ml-airflow-dag-processor airflow dags list-import-errors
docker exec ml-airflow-dag-processor airflow dags reserialize
```

### Task Failures with Retry Exhaustion

**Symptom:** A DAG task fails on every retry and lands in the `failed` state.

**Cause:** A downstream service (Spark, Garage, or Cloudflare R2) is unreachable or not
configured. Note that `r2_sync` is scheduled on the ML features asset and runs
automatically after `quantum_ml_feature_processing` succeeds; if `R2_ACCOUNT_ID`,
`R2_ACCESS_KEY_ID` and `R2_SECRET_ACCESS_KEY` are not set in `.env`, `r2_sync` fails at
`rclone_health_check`.

**Solution:** Read the task log in the Airflow UI (**DAG > Task Instance > Logs**). If you
do not sync to Cloudflare R2, keep `r2_sync` paused in the UI. Once the underlying service
is fixed, clear the failed task to retry it:

```bash
docker exec ml-airflow-apiserver airflow tasks clear \
    quantum_feature_processing -t run_quantum_processing \
    -s 2025-01-01 -e 2025-12-31 --yes
```

### Database Connection Errors

**Symptom:** Airflow containers fail to start with
`sqlalchemy.exc.OperationalError: could not connect to server` or password authentication
errors.

**Cause:** `postgres` (`ml-postgres`) is not healthy, `airflow-init` has not run schema
migrations, or `AIRFLOW_POSTGRES_PASSWORD` in `.env` was changed after the
`quantum-ml-airflow-postgres` volume was already initialized.

**Solution:**

```bash
# Check Postgres status and run Airflow migrations
docker exec ml-postgres pg_isready -U airflow
docker exec ml-airflow-apiserver airflow db migrate
```

## Monitoring Issues

### PushGateway Not Reachable

**Symptom:** Simulation logs show
`Failed to export system metrics to Prometheus: HTTPConnectionPool(host='...', port=9091)`,
or `qp_*` metrics never arrive.

**Cause:** The Compose stack does not bundle a PushGateway, so you run one separately. In
addition, `.env.ml.example` sets `PUSHGATEWAY_URL=http://localhost:9091`; inside a Docker
container, `localhost` resolves to the container itself rather than the host.

**Solution:** Run a PushGateway and point `PUSHGATEWAY_URL` at an address the producer can
reach:

```bash
# Run PushGateway on the Compose network and expose port 9091 on the host
docker run -d --name pushgateway --network quantum-ml-network -p 9091:9091 prom/pushgateway
curl -s http://localhost:9091/metrics | head -5
```

- From the host CLI, use `PUSHGATEWAY_URL=http://localhost:9091` (or
  `--performance-pushgateway http://localhost:9091`).
- From Compose containers (`quantum-pipeline-*`, `airflow-worker`), set
  `PUSHGATEWAY_URL=http://pushgateway:9091` in `.env`.

### Metrics Not Appearing in Prometheus

**Symptom:** Grafana panels show "No data" and Prometheus queries for `qp_*` return empty
results.

**Cause:** Monitoring was not enabled for the run, PushGateway received no metrics, or
Prometheus is not scraping the PushGateway target.

**Solution:**

```bash
# Check whether PushGateway holds any qp_* metrics
curl -s http://localhost:9091/metrics | grep qp_

# Enable monitoring via environment variable or CLI flag
export MONITORING_ENABLED=true
quantum-pipeline -f data/molecules.json --enable-performance-monitoring
```

In Prometheus (`http://localhost:9090/targets`), confirm the `pushgateway` target is `UP`
(see [`monitoring/prometheus.yml`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/monitoring/prometheus.yml)).

### Grafana Cannot Connect to Prometheus

**Symptom:** Testing the Prometheus data source in Grafana fails with `connection refused`.

**Solution:** Use `http://prometheus:9090` when Grafana and Prometheus share a Docker
network, or `http://localhost:9090` when running on the host. Verify under
**Connections > Data sources > Prometheus > Save & test**.

### Dashboard Import Fails or Panels Cannot Find Data Source

**Symptom:** After importing
[`monitoring/grafana/dashboards/quantum-pipeline.json`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/monitoring/grafana/dashboards/quantum-pipeline.json),
panels report `Datasource eewnvr188581sa was not found`.

**Cause:** The dashboard JSON records the Prometheus data source UID (`eewnvr188581sa`)
from the Grafana instance where it was exported.

**Solution:** Import the file via **Dashboards > Import** and pick your Prometheus data
source in the dropdown, or replace `eewnvr188581sa` in `quantum-pipeline.json` with the UID
of your Prometheus data source before provisioning.

## Batch Generation Issues

### Missing Images or Container Exits with rc=125

**Symptom:** `scripts/generate_ml_batch.py` aborts before starting with
`RuntimeError: Missing Docker images`, or a batch run fails with `rc=125`.

**Cause:** `quantum-pipeline:cpu` or `quantum-pipeline:gpu` has not been built locally, or
`docker compose run` failed in the Docker daemon before the container entrypoint started
(for example `quantum-pipeline-gpu2` requesting `NVIDIA_VISIBLE_DEVICES=1` on a single-GPU
machine, or `QUANTUM_PIPELINE_HOST_ROOT` pointing to the wrong host path).

**Solution:**

```bash
# Build both simulation images
just docker-build all

# Verify both tags exist locally
docker images quantum-pipeline
```

If both images exist and `rc=125` persists, inspect `stderr_tail` for the failed key in
`gen/ml_batch_state.json`. On a single-GPU host, set `NVIDIA_VISIBLE_DEVICES: 0` on
`quantum-pipeline-gpu2` in `compose/docker-compose.ml.yaml` or run a single GPU lane.

### Container Exits with rc=1

**Symptom:** `scripts/generate_ml_batch.py` logs `Failed (rc=1)` for a run key.

**Cause:** `quantum-pipeline` exited non-zero inside the container (a molecule calculation
failed, the Kafka broker was unreachable, or the process ran out of memory). Batch
containers are started with `docker compose ... run --rm`, so the container is removed on
exit and the last 2000 bytes of `stderr` are saved in `gen/ml_batch_state.json`.

**Solution:**

```bash
# Inspect the recorded stderr_tail and command for failed runs
python -m json.tool gen/ml_batch_state.json | grep -A 8 '"failed"'

# Check overall progress
python scripts/generate_ml_batch.py --status
```

Once the cause is fixed, rerun `python scripts/generate_ml_batch.py --tier 1` (or re-trigger
the `vqe_batch_generation` DAG in Airflow). Completed runs (`"state": "done"`) are skipped
and failed ones are retried.

### Batch Progress Not Visible

**Symptom:** Batch generation is running and `python scripts/generate_ml_batch.py --status`
shows progress, but `qp_batch_*` metrics do not appear in Grafana.

**Cause:** `scripts/generate_ml_batch.py` only pushes `qp_batch_*` metrics when
`PUSHGATEWAY_URL` is set in its environment (unlike `PerformanceMonitor`, it has no fallback
default). When triggered via the `vqe_batch_generation` DAG, the script runs inside
`airflow-worker`, where `http://localhost:9091` from `.env` points to the worker container
itself.

**Solution:** Export `PUSHGATEWAY_URL=http://localhost:9091` in your shell when running
`scripts/generate_ml_batch.py` on the host, or set `PUSHGATEWAY_URL` in `.env` to an
address reachable from `quantum-ml-network` (such as `http://pushgateway:9091`) when
running through Airflow.

If your issue is not covered here, check the [FAQ](faq.md) or open an issue on
[GitHub](https://github.com/straightchlorine/quantum-pipeline/issues) or
[Codeberg](https://codeberg.org/piotrkrzysztof/quantum-pipeline/issues).
