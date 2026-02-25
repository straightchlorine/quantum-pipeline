# Troubleshooting

This page provides solutions to common issues encountered when installing, configuring, and running the Quantum Pipeline. Problems are organized by category, each following a consistent **Symptom / Cause / Solution** format.

---

## Installation Issues

### Import Error: No module named 'qiskit'

**Symptom:** Running `quantum-pipeline` raises `ModuleNotFoundError: No module named 'qiskit'` or similar import errors for core dependencies.

**Cause:** The package was installed without its required dependencies, or the virtual environment is not activated.

**Solution:**

```bash
# Verify you are in the correct virtual environment
which python

# Reinstall with all dependencies
pip install quantum-pipeline[all]

# Or install from source
pip install -e ".[all]"
```

### Dependency Conflict with NumPy

**Symptom:** Installation fails with a message about incompatible NumPy versions, such as `numpy>=2.0 is required but numpy==1.26.4 is installed`.

**Cause:** Other packages in the environment pin an older version of NumPy that conflicts with Qiskit requirements.

**Solution:**

```bash
# Create a fresh virtual environment
python -m venv .venv
source .venv/bin/activate

# Install quantum-pipeline first to let it resolve dependencies
pip install quantum-pipeline
```

### Unsupported Python Version

**Symptom:** Installation fails with `ERROR: Package 'quantum-pipeline' requires a different Python: X.X.X not in '>=3.10,<3.13'`.

**Cause:** The installed Python version is outside the supported range.

**Solution:**

```bash
# Check current Python version
python --version

# Install a supported version (3.10, 3.11, or 3.12)
# Using pyenv:
pyenv install 3.12.0
pyenv local 3.12.0
```

### Avro Schema Import Errors

**Symptom:** `ModuleNotFoundError: No module named 'avro'` when attempting to run simulations with Kafka integration.

**Cause:** The Avro serialization dependencies are not installed.

**Solution:**

```bash
pip install python-avro confluent-kafka[avro]
```

---

## Simulation Issues

### VQE Convergence Failure

**Symptom:** The simulation completes but reports that convergence was not achieved. The optimizer reached the maximum iteration limit without finding a minimum.

**Cause:** The default iteration limit may be too low for complex molecules, or the initial parameters are stuck in a barren plateau region.

**Solution:**

```bash
# Increase the maximum number of iterations
quantum-pipeline run --molecule H2O --max-iterations 500

# Try a different optimizer
quantum-pipeline run --molecule H2O --optimizer COBYLA

# Reduce convergence tolerance for faster (less precise) convergence
quantum-pipeline run --molecule H2O --convergence 1e-4
```

### Out of Memory During Simulation

**Symptom:** The process is killed with `MemoryError` or the OOM killer terminates the container during simulation of large molecules.

**Cause:** Large molecules (12+ qubits) require significant memory for statevector simulation. Memory usage scales exponentially with qubit count.

**Solution:**

```bash
# For Docker deployments, increase container memory limit
docker run --memory=16g straightchlorine/quantum-pipeline:latest ...

# Use a simpler basis set to reduce qubit count
quantum-pipeline run --molecule H2O --basis-set sto-3g

# Reduce ansatz repetitions
quantum-pipeline run --molecule H2O --ansatz-reps 1
```

### Slow Simulation Performance

**Symptom:** Iterations take significantly longer than expected based on the performance baselines.

**Cause:** The simulation may be running on CPU when GPU acceleration is available, or system resources are contended.

**Solution:**

```bash
# Verify GPU is being used (check logs for "Using GPU backend")
quantum-pipeline run --molecule LiH --verbose

# Ensure no other heavy processes are running
top

# Check that the CUDA runtime is accessible
nvidia-smi
```

### Incorrect Energy Values

**Symptom:** The final energy value is significantly higher than expected reference values.

**Cause:** Random parameter initialization may have led to convergence on a local minimum rather than the global minimum. This is especially common for molecules with complex optimization landscapes.

**Solution:**

Run multiple simulations to increase the chance of finding a good minimum. The GPU-accelerated configuration allows more iterations in the same time budget, improving exploration of the parameter space.

---

## Docker Issues

### Container Fails to Start

**Symptom:** `docker compose up` exits immediately or containers enter a restart loop.

**Solution:** Verify `.env` exists (`cp .env.example .env` if missing), check for port conflicts (`ss -tlnp | grep -E '9091|3000|9092|8084'`), and inspect logs with `docker compose logs <service>`.

### Networking Issues Between Containers

**Symptom:** `Connection refused` errors between services (e.g., Kafka Connect to Schema Registry).

**Solution:** Verify containers share the same network (`docker network inspect quantum-net`), then restart with `docker compose down && docker compose up -d`.

### Volume Mount Permissions

**Symptom:** Permission denied errors when containers try to write to mounted volumes.

**Cause:** The container user does not have write access to the host directory.

**Solution:** Fix host permissions with `chmod -R 777 ./data ./logs` or run with `--user "$(id -u):$(id -g)"`.

### Stale Docker State

**Symptom:** Configuration changes are not taking effect.

**Solution:** Rebuild without cache (`docker compose build --no-cache`) and restart with `docker compose down -v && docker compose up -d`.

---

## Kafka Issues

### Connection Refused to Kafka Broker

**Symptom:** `kafka.errors.NoBrokersAvailable` when publishing results.

**Cause:** Kafka has not finished starting (wait 30-60s after `docker compose up`), or the bootstrap server address is wrong.

**Solution:** Check logs with `docker compose logs kafka | tail -20`. The bootstrap server should be `kafka:9092` inside Docker.

### Schema Registry Errors

**Symptom:** `SchemaRegistryError: Subject not found` or schema compatibility errors.

**Solution:** Check health with `curl http://localhost:8081/subjects`. For development, reset compatibility with:

```bash
curl -X PUT http://localhost:8081/config \
    -H "Content-Type: application/json" \
    -d '{"compatibility": "NONE"}'
```

### Message Delivery Failures

**Symptom:** Messages produced but never appear in MinIO.

**Solution:** Check connector status with `curl http://localhost:8083/connectors/minio-sink/status | python -m json.tool`. Restart with `curl -X POST http://localhost:8083/connectors/minio-sink/restart`. Check logs with `docker compose logs kafka-connect | tail -50`.

### Topic Not Created

**Symptom:** Messages fail to publish because the target topic does not exist.

**Solution:**

```bash
# List existing topics
docker compose exec kafka kafka-topics --list --bootstrap-server localhost:9092

# Create topic manually
docker compose exec kafka kafka-topics --create \
    --topic vqe_decorated_result_v1 \
    --bootstrap-server localhost:9092 \
    --partitions 3 \
    --replication-factor 1
```

---

## Spark Issues

### Spark Job Out of Memory

**Symptom:** Spark job fails with `java.lang.OutOfMemoryError: Java heap space` or executor lost errors.

**Cause:** The Spark workers do not have enough memory to process the data volume.

**Solution:**

```bash
# Increase Spark executor memory in docker-compose.yml
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

**Symptom:** Spark fails with `com.amazonaws.SdkClientException: Unable to execute HTTP request` when reading from or writing to MinIO.

**Cause:** MinIO endpoint configuration is incorrect, or credentials are missing.

**Solution:**

Verify that the following Spark configuration values are correct:

```python
spark.hadoop.fs.s3a.endpoint = http://minio:9000
spark.hadoop.fs.s3a.access.key = <your-access-key>
spark.hadoop.fs.s3a.secret.key = <your-secret-key>
spark.hadoop.fs.s3a.path.style.access = true
spark.hadoop.fs.s3a.connection.ssl.enabled = false
```

Also verify that MinIO is running and the target bucket exists:

```bash
# Check MinIO health
curl http://minio:9000/minio/health/live

# List buckets (using mc client)
docker compose exec minio mc ls local/
```

---

## Airflow Issues

### DAG Not Visible in Web UI

**Symptom:** The `quantum_feature_processing` DAG does not appear in the Airflow web interface.

**Solution:** Verify the DAG file exists (`docker compose exec airflow-webserver ls /opt/airflow/dags/`), check for import errors in **Admin > DAG Import Errors**, and force a rescan with `docker compose exec airflow-scheduler airflow dags reserialize`.

### Task Failures with Retry Exhaustion

**Symptom:** Tasks fail repeatedly and exhaust all retries.

**Solution:** Check task logs in the Airflow UI (**DAG > Task Instance > Logs**), verify dependent services with `docker compose ps`, then clear the failed task:

```bash
docker compose exec airflow-webserver airflow tasks clear \
    quantum_feature_processing -t run_quantum_processing \
    -s 2025-01-01 -e 2025-12-31 --yes
```

### Database Connection Errors

**Symptom:** Airflow fails to start with `sqlalchemy.exc.OperationalError: could not connect to server`.

**Solution:** Check PostgreSQL status (`docker compose ps postgres`), then initialize with `docker compose exec airflow-webserver airflow db init`.

---

## GPU Issues

For CUDA setup issues (`RuntimeError: CUDA is not available`) or driver version mismatches (`CUDA driver version is insufficient`), see [GPU Acceleration](../deployment/gpu-acceleration.md) for complete setup and troubleshooting instructions. The GPU image requires host driver version 520+.

### GPU Out of Memory

**Symptom:** `CUDA out of memory. Tried to allocate X MiB` error during simulation.

**Cause:** The molecule requires more GPU memory than is available on the device.

**Solution:**

```bash
# Check GPU memory usage
nvidia-smi

# Use a simpler basis set to reduce memory requirements
quantum-pipeline run --molecule H2O --basis-set sto-3g

# Fall back to CPU for very large molecules
quantum-pipeline run --molecule H2O --no-gpu
```

---

## Monitoring Issues

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
quantum-pipeline run --molecule H2 --enable-performance-monitoring
```

### Grafana Cannot Connect to Prometheus

**Symptom:** Grafana data source test fails with "Connection refused".

**Solution:** Use `http://prometheus:9090` from within Docker, or `http://localhost:9090` from the host. Verify via **Configuration > Data Sources > Prometheus > Save & Test**.

### PushGateway Errors

**Symptom:** `ConnectionError: HTTPConnectionPool host='pushgateway' port=9091` in simulation logs.

**Solution:** Verify the PushGateway is running (`docker compose ps pushgateway`) and reachable from the simulation container (`docker compose exec quantum-pipeline curl http://pushgateway:9091/metrics`).

### Dashboard Import Fails

**Symptom:** Importing the thesis dashboard JSON fails with a validation error.

**Solution:** The dashboard was created with Grafana 10.x. Ensure the Prometheus data source is named `Prometheus` (the default) or update the `DS_PROMETHEUS` variable in the dashboard JSON to match your data source name.

---

If your issue is not covered here, check the [FAQ](faq.md) or open an issue on the [GitHub repository](https://github.com/straightchlorine/quantum-pipeline/issues).
