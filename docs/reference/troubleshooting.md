# Troubleshooting

This page provides solutions to common issues encountered when installing, configuring, and running the Quantum Pipeline. Problems are organized by category, each following a consistent **Symptom / Cause / Solution** format.

---

## Installation Issues

## Simulation Issues

### VQE Convergence Failure

**Symptom:** The simulation completes but reports that convergence was not achieved. The optimizer reached the maximum iteration limit without finding a minimum.

**Cause:** The default iteration limit may be too low for complex molecules, or the initial parameters are stuck in a barren plateau region.

**Solution:**

```bash
# Increase the maximum number of iterations
python quantum_pipeline.py -f molecules.json --max-iterations 500

# Try a different optimizer
python quantum_pipeline.py -f molecules.json --optimizer COBYLA

# Reduce convergence tolerance for faster (less precise) convergence
python quantum_pipeline.py -f molecules.json --convergence --threshold 1e-4
```

### Out of Memory During Simulation

**Symptom:** The process is killed with `MemoryError` or the OOM killer terminates the container during simulation of large molecules.

**Cause:** Large molecules (12+ qubits) require significant memory for statevector simulation. Memory usage scales exponentially with qubit count.

**Solution:**

```bash
# For Docker deployments, increase container memory limit
docker run --memory=16g straightchlorine/quantum-pipeline:latest ...

# Use a simpler basis set to reduce qubit count
python quantum_pipeline.py -f molecules.json --basis sto3g

# Reduce ansatz repetitions (default is 2)
python quantum_pipeline.py -f molecules.json --ansatz-reps 2
```

### Slow Simulation Performance

**Symptom:** Iterations take significantly longer than expected based on the performance baselines.

**Cause:** The simulation may be running on CPU when GPU acceleration is available, or system resources are contended.

**Solution:**

```bash
# Verify GPU is being used (check logs for "Using GPU backend")
python quantum_pipeline.py -f molecules.json --log-level DEBUG

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

### Volume Mount Permissions

**Symptom:** Permission denied errors when containers try to write to mounted volumes.

**Cause:** The container user does not have write access to the host directory.

**Solution:** Fix host permissions with `chmod -R 777 ./data ./logs` or run with `--user "$(id -u):$(id -g)"`.

### Stale Docker State

**Symptom:** Configuration changes are not taking effect.

**Solution:** Rebuild without cache (`docker compose build --no-cache`) and restart with `docker compose down -v && docker compose up -d`.

---

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

**Symptom:** Messages produced but never appear in MinIO.

**Solution:** Check connector status with `curl http://localhost:8083/connectors/minio-sink/status | python -m json.tool` (see [`minio-sink-config.json`](https://github.com/straightchlorine/quantum-pipeline/blob/master/docker/connectors/minio-sink-config.json)). Restart with `curl -X POST http://localhost:8083/connectors/minio-sink/restart`. Check logs with `docker compose logs kafka-connect | tail -50`.

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
# Increase Spark executor memory (see docker-compose.thesis.yaml for thesis defaults)
# https://github.com/straightchlorine/quantum-pipeline/src/branch/master/docker-compose.thesis.yaml
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

Verify that the following Spark configuration values are correct (these are set in [`docker-compose.thesis.yaml`](https://github.com/straightchlorine/quantum-pipeline/blob/master/docker-compose.thesis.yaml)):

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

**Symptom:** The [`quantum_feature_processing`](https://github.com/straightchlorine/quantum-pipeline/blob/master/docker/airflow/quantum_processing_dag.py#L72) DAG does not appear in the Airflow web interface.

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
python quantum_pipeline.py -f molecules.json --basis sto3g

# Fall back to CPU for very large molecules (omit --gpu to use CPU)
python quantum_pipeline.py -f molecules.json
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
python quantum_pipeline.py -f molecules.json --enable-performance-monitoring
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
