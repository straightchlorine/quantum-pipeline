# Data Platform

Processes, transforms, and stores VQE simulation results into analytics-ready
ML feature tables. For the full component-by-component architecture, see
[System Design](../architecture/system-design.md).

<div class="grid cards" markdown>

-   :material-broadcast:{ .lg .middle } **Kafka Streaming**

    ---

    Single-topic ingestion with Schema Registry validation. Redpanda Connect
    decodes Avro messages and writes JSON to Garage. Covers producer config,
    wire format, and security options.

    [:octicons-arrow-right-24: Kafka Streaming](kafka-streaming.md)

-   :material-lightning-bolt:{ .lg .middle } **Spark Processing**

    ---

    Standalone Spark cluster transforms raw JSON into 9 base feature tables
    and 2 ML feature tables. Incremental processing via Iceberg metadata.

    [:octicons-arrow-right-24: Spark Processing](spark-processing.md)

-   :material-calendar-clock:{ .lg .middle } **Airflow Orchestration**

    ---

    Four DAGs handle feature processing, ML materialization, batch generation,
    and R2 cloud sync. CeleryExecutor with Redis broker and PostgreSQL metadata.

    [:octicons-arrow-right-24: Airflow Orchestration](airflow-orchestration.md)

-   :material-database:{ .lg .middle } **Iceberg Storage**

    ---

    Iceberg table format on top of Garage (S3-compatible storage). ACID
    transactions, snapshot tagging, partition pruning, and time-travel queries.

    [:octicons-arrow-right-24: Iceberg Storage](iceberg-storage.md)

</div>
