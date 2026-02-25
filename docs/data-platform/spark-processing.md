# Spark Processing

## Overview

**Apache Spark** transforms raw VQE simulation results into structured ML feature tables. The processing is batch-oriented and incremental - only new data since the last run is processed.

[:octicons-arrow-right-24: System Design](../architecture/system-design.md) | [:octicons-arrow-right-24: Airflow Orchestration](airflow-orchestration.md)

---

## Cluster Architecture

The cluster runs in standalone master-worker mode with one master and two worker nodes. See the [Spark documentation](https://spark.apache.org/docs/latest/) for general cluster configuration.

| Node | Role | Resources |
|------|------|-----------|
| `spark-master` | Coordinator | 2 cores, 2 GB RAM |
| `spark-worker-1` | Executor | 4 cores, 8 GB RAM |
| `spark-worker-2` | Executor | 4 cores, 8 GB RAM |

- Workers register with the master at `spark://spark-master:7077`
- Airflow submits jobs via the `SparkSubmitOperator`
- Workers access MinIO directly through the S3A filesystem connector
- Spark Web UI is available at port `8080` on the master node

<figure>
  <img src="https://qp-docs.codextechnologies.org/mkdocs/spark_view.png"
       alt="Spark Web UI showing the master node with registered workers and completed applications">
  <figcaption>Figure 1. Spark Web UI displaying the master node with registered workers and job execution history.</figcaption>
</figure>

---

## Feature Engineering Pipeline

A 6-step workflow transforms raw VQE results into nine feature tables. Each run is incremental.

### Step 1: Create Spark Session

A `SparkSession` is initialized with configuration for S3A (MinIO connectivity), Iceberg catalog integration, and adaptive query execution.

```python
spark = (
    SparkSession.builder
    .appName(config.get('APP_NAME'))
    .master(config.get('SPARK_MASTER'))
    .config('spark.sql.extensions',
            'org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions')
    .config('spark.sql.catalog.quantum_catalog',
            'org.apache.iceberg.spark.SparkCatalog')
    .config('spark.sql.catalog.quantum_catalog.type', 'hadoop')
    .config('spark.sql.catalog.quantum_catalog.warehouse',
            config.get('S3_WAREHOUSE'))
    .config('spark.hadoop.fs.s3a.impl',
            'org.apache.hadoop.fs.s3a.S3AFileSystem')
    .config('spark.hadoop.fs.s3a.endpoint', config.get('S3_ENDPOINT'))
    .config('spark.hadoop.fs.s3a.path.style.access', 'true')
    .config('spark.sql.adaptive.enabled', 'true')
    .config('spark.sql.shuffle.partitions', '200')
    .getOrCreate()
)
```

### Step 2: Initialize Iceberg Metadata

The job reads or initializes the Iceberg metadata catalog. If this is the first run, the catalog and database are created. On subsequent runs, existing metadata is loaded to determine which data has already been processed.

```python
def list_available_topics(spark, bucket_path):
    """List available topics from the storage."""
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(
        spark._jvm.java.net.URI.create(bucket_path),
        spark._jsc.hadoopConfiguration()
    )
    path = spark._jvm.org.apache.hadoop.fs.Path(bucket_path)
    if fs.exists(path) and fs.isDirectory(path):
        return [f.getPath().getName() for f in fs.listStatus(path)
                if f.isDirectory()]
    return []
```

### Step 3: Filter New Data

New records are identified via anti-join against existing Iceberg table keys.

```python
def identify_new_records(spark, new_data_df, table_name, key_columns):
    """Identify records not yet present in the target Iceberg table."""
    if not spark.catalog.tableExists(
            f'quantum_catalog.quantum_features.{table_name}'):
        return new_data_df

    existing_keys = spark.sql(
        f'SELECT DISTINCT {", ".join(key_columns)} '
        f'FROM quantum_catalog.quantum_features.{table_name}'
    )

    if existing_keys.isEmpty():
        return new_data_df

    # Anti-join to find only new records
    new_keys = (new_data_df.select(*key_columns).distinct()
                .join(existing_keys, on=key_columns, how='left_anti'))
    return new_data_df.join(new_keys, on=key_columns, how='inner')
```

### Step 4: Extract Features

Raw VQE data is transformed into nine specialized feature tables covering molecular structure, optimization, iterations, parameters, and Hamiltonians.

```python
def transform_quantum_data(df):
    """Transform raw VQE data into specialized feature tables."""
    base_df = add_metadata_columns(df, 'quantum_base_processing')
    return {
        'molecules': extract_molecule_features(base_df),
        'ansatz_info': extract_ansatz_features(base_df),
        'performance_metrics': extract_performance_features(base_df),
        'vqe_results': extract_vqe_features(base_df),
        'initial_parameters': extract_initial_params(base_df),
        'optimal_parameters': extract_optimal_params(base_df),
        'vqe_iterations': extract_iteration_features(base_df),
        'iteration_parameters': extract_iteration_params(base_df),
        'hamiltonian_terms': extract_hamiltonian_features(base_df),
    }
```

### Step 5: Write Parquet to MinIO

Feature data is written as Parquet through the Iceberg table API. New tables are created on first run; subsequent runs append only new records.

```python
def process_incremental_data(spark, new_data_df, table_name, key_columns,
                             partition_columns=None):
    """Write new data to an Iceberg table, creating it if necessary."""
    table_path = f'quantum_catalog.quantum_features.{table_name}'

    if not spark.catalog.tableExists(table_path):
        writer = new_data_df.write.format('iceberg') \
            .option('write-format', 'parquet')
        if partition_columns:
            writer = writer.partitionBy(*partition_columns)
        writer.mode('overwrite').saveAsTable(table_path)
    else:
        truly_new = identify_new_records(spark, new_data_df,
                                         table_name, key_columns)
        if truly_new.count() > 0:
            truly_new.write.format('iceberg') \
                .option('write-format', 'parquet') \
                .mode('append') \
                .saveAsTable(table_path)
```

### Step 6: Update Iceberg Snapshots

After writing, a tagged Iceberg snapshot records the processing batch ID for time-travel queries.

```python
snapshot_id = spark.sql(
    f'SELECT snapshot_id '
    f'FROM quantum_catalog.quantum_features.{table_name}.snapshots '
    f'ORDER BY committed_at DESC LIMIT 1'
).collect()[0][0]

version_tag = f'v_{processing_batch_id}'
spark.sql(f"""
    ALTER TABLE quantum_catalog.quantum_features.{table_name}
    CREATE TAG {version_tag} AS OF VERSION {snapshot_id}
""")
```

---

## Feature Tables

Nine tables stored as Iceberg-managed Parquet files in MinIO under `quantum_catalog.quantum_features`.

<figure>
  <img src="https://qp-docs.codextechnologies.org/mkdocs/feature_parquet_preview.png"
       alt="Preview of processed Parquet feature data showing structured columns and values">
  <figcaption>Figure 2. Preview of processed feature data in Parquet format, queried with DuckDB.</figcaption>
</figure>

### 1. molecules

Molecular structure information for each simulated molecule.

| Column | Type | Description |
|--------|------|-------------|
| `experiment_id` | string | Unique experiment identifier |
| `molecule_id` | int | Molecule identifier |
| `atom_symbols` | array&lt;string&gt; | Chemical symbols (e.g., ["H", "H"]) |
| `coordinates` | array&lt;array&lt;double&gt;&gt; | 3D atomic coordinates |
| `multiplicity` | int | Spin multiplicity |
| `charge` | int | Molecular charge |
| `coordinate_units` | string | Units (angstrom or bohr) |
| `atomic_masses` | array&lt;double&gt; | Atomic masses (amu) |

**Partitioned by:** `processing_date`

### 2. ansatz_info

Quantum circuit ansatz configurations used in each experiment.

| Column | Type | Description |
|--------|------|-------------|
| `experiment_id` | string | Unique experiment identifier |
| `molecule_id` | int | Molecule identifier |
| `basis_set` | string | Basis set (e.g., sto-3g, cc-pvdz) |
| `ansatz` | string | QASM3 circuit representation |
| `ansatz_reps` | int | Number of ansatz repetitions |

**Partitioned by:** `processing_date`, `basis_set`

### 3. performance_metrics

Execution timing and performance data for each simulation.

| Column | Type | Description |
|--------|------|-------------|
| `experiment_id` | string | Unique experiment identifier |
| `molecule_id` | int | Molecule identifier |
| `basis_set` | string | Basis set |
| `hamiltonian_time` | double | Hamiltonian construction time (seconds) |
| `mapping_time` | double | Fermionic-to-qubit mapping time (seconds) |
| `vqe_time` | double | VQE optimization time (seconds) |
| `total_time` | double | Total simulation time (seconds) |
| `minimization_time` | double | Optimizer execution time (seconds) |

**Partitioned by:** `processing_date`, `basis_set`

### 4. vqe_results

Core VQE optimization results including the minimum energy found.

| Column | Type | Description |
|--------|------|-------------|
| `experiment_id` | string | Unique experiment identifier |
| `molecule_id` | int | Molecule identifier |
| `basis_set` | string | Basis set |
| `backend` | string | Qiskit backend (e.g., aer_simulator) |
| `num_qubits` | int | Number of qubits in the circuit |
| `optimizer` | string | Optimizer algorithm (e.g., L-BFGS-B, COBYLA) |
| `minimum_energy` | double | Ground state energy estimate (Hartree) |
| `total_iterations` | int | Number of optimizer iterations |

**Partitioned by:** `processing_date`, `basis_set`, `backend`

### 5. initial_parameters

Starting variational parameters for each experiment.

| Column | Type | Description |
|--------|------|-------------|
| `parameter_id` | string | Unique parameter identifier |
| `experiment_id` | string | Experiment identifier |
| `molecule_id` | int | Molecule identifier |
| `parameter_index` | int | Parameter position in the circuit |
| `initial_parameter_value` | double | Starting parameter value |

**Partitioned by:** `processing_date`, `basis_set`

### 6. optimal_parameters

Optimized variational parameters after VQE convergence.

| Column | Type | Description |
|--------|------|-------------|
| `parameter_id` | string | Unique parameter identifier |
| `experiment_id` | string | Experiment identifier |
| `molecule_id` | int | Molecule identifier |
| `parameter_index` | int | Parameter position in the circuit |
| `optimal_parameter_value` | double | Optimized parameter value |

**Partitioned by:** `processing_date`, `basis_set`

### 7. vqe_iterations

Per-iteration optimization data recording the energy at each step.

| Column | Type | Description |
|--------|------|-------------|
| `iteration_id` | string | Unique iteration identifier |
| `experiment_id` | string | Experiment identifier |
| `molecule_id` | int | Molecule identifier |
| `iteration_step` | int | Iteration number |
| `iteration_energy` | double | Energy at this step (Hartree) |
| `energy_std_dev` | double | Standard deviation of measurement |

**Partitioned by:** `processing_date`, `basis_set`, `backend`

### 8. iteration_parameters

Parameter values at each optimizer iteration, enabling convergence analysis.

| Column | Type | Description |
|--------|------|-------------|
| `parameter_id` | string | Unique parameter identifier |
| `iteration_id` | string | Iteration identifier |
| `experiment_id` | string | Experiment identifier |
| `iteration_step` | int | Iteration number |
| `parameter_value` | double | Parameter value at this iteration |

**Partitioned by:** `processing_date`, `basis_set`

### 9. hamiltonian_terms

Hamiltonian Pauli operator terms and their complex coefficients.

| Column | Type | Description |
|--------|------|-------------|
| `term_id` | string | Unique term identifier |
| `experiment_id` | string | Experiment identifier |
| `molecule_id` | int | Molecule identifier |
| `term_label` | string | Pauli string (e.g., "XYZI", "IIZZ") |
| `coeff_real` | double | Real part of the coefficient |
| `coeff_imag` | double | Imaginary part of the coefficient |

**Partitioned by:** `processing_date`, `basis_set`, `backend`

---

## Incremental Processing

Spark uses Iceberg metadata to identify and process only new records on each run.

### How It Works

1. **List current files** - Spark enumerates all Avro files in the MinIO experiments bucket.
2. **Check Iceberg snapshots** - If an Iceberg snapshot exists from a previous run, the system retrieves the set of already-processed file paths.
3. **Compute delta** - New files are identified as the difference between current files and previously processed files.
4. **Process delta only** - Only the new files are read, transformed, and appended to the feature tables.
5. **Update snapshot** - A new Iceberg snapshot is created, recording the current set of processed files.

### Performance Impact

| Operation | 10K records | 100K records |
|-----------|-------------|--------------|
| Full reprocessing | ~45 seconds | ~8 minutes |
| Incremental (1K new) | ~5 seconds | ~5 seconds |
| Metadata-only query | ~100 ms | ~100 ms |
| Partition pruning | ~2 seconds | ~10 seconds |

!!! info "Performance Improvement"
    Incremental processing reduces processing time by 90-95% compared to full reprocessing.

---

## Related Documentation

- [System Design](../architecture/system-design.md) - Full architecture overview
- [Airflow Orchestration](airflow-orchestration.md) - Spark job scheduling
- [Iceberg Storage](iceberg-storage.md) - Table format and snapshots
- [Kafka Streaming](kafka-streaming.md) - Raw data ingestion

## References

- [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
- [Apache Iceberg](https://iceberg.apache.org/docs/latest/)
- [Hadoop S3A Connector](https://hadoop.apache.org/docs/stable/hadoop-aws/tools/hadoop-aws/index.html)
