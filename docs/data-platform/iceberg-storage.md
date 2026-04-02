# Iceberg Storage

Apache Iceberg provides the table format for feature tables, with Garage
(v2.2.0) as the S3-compatible storage backend. Together they deliver ACID
transactions, time-travel queries, schema evolution, and snapshot management.

For how Iceberg and Garage fit into the overall architecture, see
[System Design](../architecture/system-design.md#garage-storage).

## Catalog Structure

All feature tables are organized under a single Iceberg catalog:

```
quantum_catalog                         -- Iceberg catalog
  └── quantum_features                  -- Database
        ├── molecules                   -- Table
        ├── ansatz_info                 -- Table
        ├── performance_metrics         -- Table
        ├── vqe_results                 -- Table
        ├── initial_parameters          -- Table
        ├── optimal_parameters          -- Table
        ├── vqe_iterations              -- Table
        ├── iteration_parameters        -- Table
        ├── hamiltonian_terms           -- Table
        ├── ml_iteration_features       -- ML feature table
        ├── ml_run_summary              -- ML feature table
        └── processing_metadata         -- Audit table
```

## Catalog Configuration

Configured via
[`compose/spark-defaults.conf`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/compose/spark-defaults.conf),
mounted into Spark containers:

| Configuration | Value | Description |
|---------------|-------|-------------|
| Catalog name | `quantum_catalog` | Identifier used in SQL queries |
| Catalog type | `hadoop` | Metadata stored in the warehouse path |
| Warehouse | `s3a://features/warehouse/` | Root location for table data and metadata in Garage |
| IO implementation | `HadoopFileIO` | File I/O layer for reading/writing through S3A |

## S3A Configuration

Spark reaches Garage through the Hadoop S3A filesystem driver, also in
`spark-defaults.conf`:

| Setting | Value |
|---------|-------|
| Endpoint | `http://garage:3901` |
| Region | `garage` |
| Path-style access | `true` |
| SSL | `false` |
| Fast upload | `true` |

Credentials are provided via `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
environment variables. In `docker-compose.ml.yaml`, these map from the
project-level `S3_ACCESS_KEY` and `S3_SECRET_KEY`.

## Physical Storage Layout

```
s3a://features/warehouse/
└── quantum_features/
    ├── vqe_results/
    │   ├── metadata/
    │   │   ├── v1.metadata.json
    │   │   ├── v2.metadata.json
    │   │   ├── snap-1234567890.avro
    │   │   └── snap-1234567891.avro
    │   └── data/
    │       ├── processing_date=2025-01-10/
    │       │   ├── part-00000.parquet
    │       │   └── part-00001.parquet
    │       └── processing_date=2025-01-11/
    │           └── part-00000.parquet
    ├── molecules/
    │   ├── metadata/
    │   └── data/
    ├── ml_iteration_features/
    │   ├── metadata/
    │   └── data/
    └── ml_run_summary/
        ├── metadata/
        └── data/
```

### Metadata Files

| File Type | Description |
|-----------|-------------|
| `v*.metadata.json` | Table schema, partition spec, snapshot pointer, table properties |
| `snap-*.avro` | Snapshot metadata with manifest list references |
| Manifest lists | Lists of manifest files for a given snapshot |
| Manifest files | Lists of data files, partition values, file-level statistics |

## Snapshot Tagging

Each Spark write creates a new snapshot, tagged with a version identifier from
the processing batch ID. This enables reproducible ML training by referencing
a specific snapshot tag.

Tagging is done in
[`process_incremental_data()`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/docker/airflow/scripts/quantum_incremental_processing.py#L190).
Initial writes use `v_{batch_id}` tags; incremental appends use
`v_incr_{batch_id}`.

## Garage Integration

Garage (v2.2.0) provides S3-compatible storage for raw JSON data and processed
Parquet feature tables. It replaced MinIO from v1.x..

The Garage service is defined in
[`compose/docker-compose.ml.yaml`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/compose/docker-compose.ml.yaml#L191)
and configured via
[`compose/garage.toml.template`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/compose/garage.toml.template)
(uses env var substitution for secrets).

### Buckets

| Bucket | Purpose | Writer |
|--------|---------|--------|
| `raw-results` | JSON files from Redpanda Connect | Redpanda Connect S3 output |
| `features` | Processed Parquet files and Iceberg metadata | Apache Spark |
| `mlflow-artifacts` | MLflow experiment artifacts | MLflow tracking server |

### Ports

| Port | Service |
|------|---------|
| `3901` | S3 API (used by Redpanda Connect, Spark, rclone) |
| `3903` | Admin API (bucket creation, key management) |

### Raw Data Layout

Written by Redpanda Connect (see [Kafka Streaming](kafka-streaming.md)):

```
s3://raw-results/
└── experiments/
    └── experiment.vqe/
        ├── 1-1711900800000000000.json
        ├── 2-1711900801000000000.json
        └── ...
```

File naming: `{counter}-{unix_nano_timestamp}.json`.

!!! tip
    Garage is S3 compatible. Use `aws configure --profile garage` with your
    `S3_ACCESS_KEY` / `S3_SECRET_KEY` from `.env` to browse files with
    `aws s3`.

## Partitioning Strategy

Partitioning is set for expected query patterns:

| Table | Partition Columns | Rationale |
|-------|-------------------|-----------|
| `molecules` | `processing_date` | Time-based filtering for incremental loads |
| `ansatz_info` | `processing_date`, `basis_set` | Filter by date and basis set |
| `performance_metrics` | `processing_date`, `basis_set` | Performance comparisons across basis sets |
| `vqe_results` | `processing_date`, `basis_set`, `backend` | Query by date, basis set, and backend |
| `initial_parameters` | `processing_date`, `basis_set` | Parameter analysis by date and basis set |
| `optimal_parameters` | `processing_date`, `basis_set` | Optimized parameter lookup |
| `vqe_iterations` | `processing_date`, `basis_set`, `backend` | Iteration analysis with backend filtering |
| `iteration_parameters` | `processing_date`, `basis_set` | Per-iteration parameter tracking |
| `hamiltonian_terms` | `processing_date`, `basis_set`, `backend` | Hamiltonian structure analysis |

Iceberg uses partition metadata to skip irrelevant data files at query time,
reducing I/O for partition-filtered queries.

## Processing Metadata Table

An audit table tracks all processing runs:

| Column | Type | Description |
|--------|------|-------------|
| `processing_batch_id` | string | Batch identifier |
| `processing_name` | string | Processing job name |
| `processing_timestamp` | timestamp | When the batch was processed |
| `processing_date` | date | Processing date |
| `table_names` | array&lt;string&gt; | Tables written in this batch |
| `table_versions` | array&lt;string&gt; | Snapshot tags for each table |
| `record_counts` | array&lt;bigint&gt; | Records written per table |
| `source_data_info` | string | Source data description |

## Table Maintenance

Iceberg tables require periodic maintenance including snapshot expiration, data
file compaction, and orphan file cleanup. See the
[Iceberg Maintenance documentation](https://iceberg.apache.org/docs/latest/maintenance/).

## Related Documentation

- [System Design](../architecture/system-design.md) - Full architecture overview
- [Spark Processing](spark-processing.md) - Feature engineering pipeline
- [Kafka Streaming](kafka-streaming.md) - How raw data arrives via Redpanda Connect
- [Airflow Orchestration](airflow-orchestration.md) - Pipeline scheduling

## References

- [Apache Iceberg Documentation](https://iceberg.apache.org/docs/latest/)
- [Iceberg Table Maintenance](https://iceberg.apache.org/docs/latest/maintenance/)
- [Garage Documentation](https://garagehq.deuxfleurs.fr/documentation/quick-start/)
