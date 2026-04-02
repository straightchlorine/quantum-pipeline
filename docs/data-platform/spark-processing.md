# Spark Processing

Apache Spark 4.0.2 transforms raw VQE simulation results into structured ML
feature tables. Processing is batch-oriented and incremental - only new data
since the last run is processed.

For how Spark fits into the overall architecture, see
[System Design](../architecture/system-design.md#incremental-processing). For
feature table schemas, see
[System Design - Feature Tables](../architecture/system-design.md#feature-tables-schema).

## Cluster Architecture

The cluster runs in standalone master-worker mode via a custom Docker image
built from
[`docker/Dockerfile.spark`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/docker/Dockerfile.spark)
(based on `apache/spark:4.0.2-python3` with Python upgraded to 3.12 to match
the Airflow driver).

| Node | Role | Resources |
|------|------|-----------|
| `spark-master` | Coordinator | 1 GB RAM limit |
| `spark-worker` | Executor | 3 GB container limit, 2 GB Spark worker memory, 2 cores |

Worker memory and cores are configurable via `SPARK_WORKER_MEMORY` and
`SPARK_WORKER_CORES` in
[`compose/docker-compose.ml.yaml`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/compose/docker-compose.ml.yaml#L245).

Key details:

- Workers register with the master at `spark://spark-master:7077`
- Airflow submits jobs via `SparkSubmitOperator`
- Workers access Garage through the S3A filesystem connector
- Configuration is loaded from
  [`compose/spark-defaults.conf`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/compose/spark-defaults.conf)
  mounted into containers
- S3 credentials are passed via `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
  environment variables
- JAR dependencies (Iceberg runtime, Hadoop AWS, Spark Avro) are resolved via
  Maven/Ivy at startup and cached in the `spark-ivy-cache` volume
- Spark Web UI is available at port `8080` on the master node

<figure>
  <img src="https://qp-docs.codextechnologies.org/mkdocs/spark_view.png"
       alt="Spark Web UI (master node)">
  <figcaption>Figure 1. Spark Web UI (master node).</figcaption>
</figure>

## Spark Configuration

All Spark scripts use a shared session factory
([`docker/airflow/common/spark_factory.py`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/docker/airflow/common/spark_factory.py))
that creates a `SparkSession` with only the app name. All other settings come
from `spark-defaults.conf`:

| Setting | Value |
|---------|-------|
| S3A endpoint | `http://garage:3901`, path-style access, fast upload |
| Iceberg catalog | `quantum_catalog`, Hadoop type, warehouse at `s3a://features/warehouse/` |
| Serialization | KryoSerializer |
| Memory | 1 GB driver, 1536 MB executor, 8 shuffle partitions |
| JARs | `iceberg-spark-runtime-4.0_2.13:1.10.1`, `hadoop-aws:3.4.1`, `spark-avro_2.13:4.0.2` |

Default S3 paths from
[`common/pipeline_config.py`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/docker/airflow/common/pipeline_config.py):

| Path | Default |
|------|---------|
| Experiment bucket | `s3a://raw-results/experiments/` |
| Feature warehouse | `s3a://features/warehouse/` |

## Feature Engineering Pipeline

The processing script
[`quantum_incremental_processing.py`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/docker/airflow/scripts/quantum_incremental_processing.py)
follows a 6-step workflow:

1. **Create Spark session** - via the shared factory
2. **Initialize Iceberg metadata** - create catalog/database on first run, load
   existing metadata on subsequent runs
3. **Filter new data** - join against existing Iceberg table keys using a
   marker-column approach (see
   [`identify_new_records()`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/docker/airflow/scripts/quantum_incremental_processing.py#L134))
4. **Extract features** - `transform_quantum_data()` flattens nested VQE fields
   into 9 specialized tables with metadata columns (`experiment_id`,
   `processing_timestamp`, `processing_date`, `processing_batch_id`)
5. **Write Parquet to Garage** - via
   [`process_incremental_data()`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/docker/airflow/scripts/quantum_incremental_processing.py#L190),
   creating tables on first run, appending only new records on subsequent runs
6. **Tag Iceberg snapshots** - version tags (`v_{batch_id}` or
   `v_incr_{batch_id}`) for time-travel queries

## Feature Tables

### Base Feature Tables (9 tables)

Produced by the `quantum_feature_processing` DAG and stored under
`quantum_catalog.quantum_features`. For the full schema of all 9 tables, see
[System Design - Feature Tables](../architecture/system-design.md#feature-tables-schema).


| Table | Key Columns | Partition Columns |
|-------|-------------|-------------------|
| `molecules` | experiment_id, molecule_id | processing_date |
| `ansatz_info` | experiment_id, molecule_id | processing_date, basis_set |
| `performance_metrics` | experiment_id, molecule_id, basis_set | processing_date, basis_set |
| `vqe_results` | experiment_id, molecule_id, basis_set | processing_date, basis_set, backend |
| `initial_parameters` | parameter_id | processing_date, basis_set |
| `optimal_parameters` | parameter_id | processing_date, basis_set |
| `vqe_iterations` | iteration_id | processing_date, basis_set, backend |
| `iteration_parameters` | parameter_id | processing_date, basis_set |
| `hamiltonian_terms` | term_id | processing_date, basis_set, backend |

### ML Feature Tables (2 tables)

Produced by the `quantum_ml_feature_processing` DAG
([`quantum_ml_feature_processing.py`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/docker/airflow/scripts/quantum_ml_feature_processing.py)).
These join and aggregate the 9 base tables into ML-ready datasets.

| Table | Purpose |
|-------|---------|
| `ml_iteration_features` | Per-iteration feature vectors combining energy data with rolling statistics, parameter snapshots, and molecular context |
| `ml_run_summary` | Per-run aggregate features summarizing convergence metrics, timing, and molecule-level data |

## Incremental Processing

Each run processes only new data:

1. **List current files** in the Garage experiments bucket
2. **Check Iceberg snapshots** for already-processed file paths
3. **Compute delta** - new files not yet processed
4. **Process delta only** - read, transform, append to feature tables
5. **Update snapshot** - new Iceberg snapshot records the processed set

The deduplication logic is in
[`identify_new_records()`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/docker/airflow/scripts/quantum_incremental_processing.py#L134)
and the write logic in
[`process_incremental_data()`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/docker/airflow/scripts/quantum_incremental_processing.py#L190).

## Related Documentation

- [System Design](../architecture/system-design.md) - Full architecture and feature table schemas
- [Airflow Orchestration](airflow-orchestration.md) - Spark job scheduling
- [Iceberg Storage](iceberg-storage.md) - Table format and snapshots
- [Kafka Streaming](kafka-streaming.md) - Raw data ingestion

## References

- [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
- [Apache Iceberg](https://iceberg.apache.org/docs/latest/)
- [Hadoop S3A Connector](https://hadoop.apache.org/docs/stable/hadoop-aws/tools/hadoop-aws/index.html)
