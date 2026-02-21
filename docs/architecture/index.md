# Architecture Overview

The Quantum Pipeline is built on a **microservices architecture** that combines quantum simulation,
data streaming, distributed processing, and scalable storage.

This section explains the system design, data flow patterns, and key architectural decisions.

---

## Design Philosophy

The architecture follows these core principles:

!!! tip "Separation of Concerns"
    Each component handles a specific responsibility:

    - **Quantum Simulation** - VQE algorithm execution
    - **Streaming** - Real-time data transport (Kafka)
    - **Processing** - Feature engineering (Spark)
    - **Storage** - Persistent data lake (Iceberg/MinIO)
    - **Orchestration** - Workflow automation (Airflow)

!!! tip "Loose Coupling"
    Components communicate through well-defined interfaces (Avro schemas) allowing independent scaling and deployment.

!!! tip "Scalability"
    - Horizontal scaling for Spark workers
    - Vertical/Horizontal scaling for GPU containers
    - Partitioned Kafka topics for parallelism

!!! tip "Fault Tolerance"
    - Kafka message persistence
    - Airflow retry mechanisms
    - Iceberg ACID transactions

---

## High-Level Architecture

```mermaid
graph TD
    QP[Quantum Pipeline] -->|Publish Results| KAFKA[Kafka]
    QP -->|Export Metrics| PROM[Prometheus + Grafana]
    KAFKA <-->|Validate| SR[Schema Registry]
    KAFKA -->|S3 Sink| KC[Kafka Connect]
    KC -->|Write Avro| MINIO[MinIO]
    AIRFLOW[Airflow] -->|Trigger| SPARK[Spark Cluster]
    SPARK -->|Read Raw Data| MINIO
    SPARK -->|Write Features| ICE[Iceberg]
    ICE -->|Store Parquet| MINIO

    style QP fill:#c5cae9,color:#1a237e
    style KAFKA fill:#ffe082,color:#000
    style SR fill:#ffe082,color:#000
    style KC fill:#ffe082,color:#000
    style AIRFLOW fill:#90caf9,color:#0d47a1
    style SPARK fill:#a5d6a7,color:#1b5e20
    style MINIO fill:#b39ddb,color:#311b92
    style ICE fill:#b39ddb,color:#311b92
    style PROM fill:#e8f5e9,color:#1b5e20
```

---

## Component Overview

###  Quantum Simulation Layer

**Quantum Pipeline Container**

- Executes VQE simulations using Qiskit Aer
- Supports CPU and GPU backends
- Monitors iteration-level metrics
- Produces structured result data

**Performance Monitor**

- Collects system metrics (CPU, memory, GPU)
- Tracks VQE-specific metrics (energy, iterations)
- Exports to Prometheus PushGateway
- Non-blocking background thread

[:octicons-arrow-right-24: System Design Details](system-design.md)

---

###  Messaging Layer

**Apache Kafka**

- Distributed message broker
- Topic-based publish/subscribe
- Message persistence and replay
- Partitioning for parallelism

**Schema Registry**

- Centralized Avro schema management
- Schema versioning and evolution
- Backward/forward compatibility checks
- Automatic topic suffix generation

**Kafka Connect**

- S3 Sink Connector for MinIO
- Automatic Avro file writing
- Flush size configuration
- Error tolerance and retry

[:octicons-arrow-right-24: Avro Serialization Pattern](avro-serialization.md)

---

###  Orchestration Layer

**Apache Airflow**

- DAG-based workflow orchestration
- Daily scheduling for batch processing
- Retry logic with exponential backoff
- Email alerting on success/failure
- PostgreSQL for metadata storage

**Key DAG**: `quantum_processing_dag`

- Ingests VQE results from Kafka
- Triggers Spark feature engineering jobs
- Loads processed data into Iceberg tables
- Manages incremental processing state

[:octicons-arrow-right-24: Data Flow Pattern](data-flow.md)

---

###  Processing Layer

**Apache Spark Cluster**

- Master node for job coordination
- Worker nodes for distributed processing
- In-memory computation engine
- Support for Avro, Parquet, Iceberg

**Processing Pattern**

1. Read raw Avro files from MinIO
2. Deserialize using Avro schemas
3. Transform into ML feature tables
4. Write to Iceberg in Parquet format
5. Update metadata snapshots

**Feature Tables** (9 tables):

- `molecules` - Molecular structures
- `vqe_results` - Optimization results
- `vqe_iterations` - Iteration history
- `vqe_parameters` - Optimal parameters
- `hamiltonians` - Operator coefficients
- `timing_metrics` - Performance data
- `accuracy_metrics` - Scientific validation
- `system_metrics` - Resource usage
- `processing_metadata` - Data lineage

---

###  Storage Layer

**MinIO (S3-Compatible Object Storage)**

- Raw Avro files from Kafka Connect
- Parquet files from Spark processing
- Iceberg table data files
- Bucket: `local-vqe-results`

**Apache Iceberg (Data Lake Metadata)**

- ACID transaction support
- Time-travel queries
- Schema evolution
- Snapshot isolation
- Partition management

**PostgreSQL**

- Airflow metadata database
- DAG run history
- Task state tracking
- Connection management

---

###  Monitoring Layer

**Prometheus PushGateway**

- Receives metrics from short-lived jobs
- Gateway for container metrics
- Time-series data storage
- Label-based querying

**Grafana**

- Visualization dashboards
- Real-time metric monitoring
- Custom query panels
- Alert management

**Monitored Metrics**:

- System: CPU%, memory%, disk I/O
- VQE: Energy convergence, iterations, timing
- Scientific: Accuracy vs reference database
- Data: Kafka lag, Spark job duration

---

## Data Flow Pattern

```mermaid
sequenceDiagram
    participant QP as Quantum Pipeline
    participant K as Kafka
    participant SR as Schema Registry
    participant KC as Kafka Connect
    participant M as MinIO
    participant A as Airflow
    participant S as Spark
    participant I as Iceberg

    QP->>SR: Register Avro Schema
    SR-->>QP: Schema ID
    QP->>K: Publish VQE Results (Avro)
    K->>KC: Consume Messages
    KC->>M: Write Avro Files (S3 Sink)

    Note over A: Daily Schedule Trigger
    A->>S: Submit Feature Engineering Job
    S->>M: Read Raw Avro Files
    S->>S: Transform to ML Features
    S->>I: Write Feature Tables (Parquet)
    I->>M: Store Data Files
    I->>I: Create Snapshot
    S-->>A: Job Complete
    A->>A: Email Success Notification
```

[:octicons-arrow-right-24: Detailed Data Flow](data-flow.md)

---

## Key Architectural Patterns

### 1. Event-Driven Architecture

Example 

```mermaid
graph LR
    E[VQE Simulation<br/>Event] --> K[Kafka Topic]
    K --> C1[Consumer 1:<br/>MinIO Storage]
    K --> C2[Consumer 2:<br/>Real-time Analytics]
    K --> C3[Consumer 3:<br/>ML Pipeline]

    style E fill:#e1f5ff,color:#3f51b5
    style K fill:#f9a825,color:#000
```

**Benefits**:

- Decoupling producers from consumers
- Multiple consumers for same data
- Message persistence and replay
- Asynchronous processing

### 2. Schema Evolution

```mermaid
graph TB
    V1[VQE Result<br/>Schema v1] -->|Add Field| V2[VQE Result<br/>Schema v2]
    V2 -->|Optimize Structure| V3[VQE Result<br/>Schema v3]

    V1 --> T1[Topic: vqe_result_v1]
    V2 --> T2[Topic: vqe_result_v2]
    V3 --> T3[Topic: vqe_result_v3]

    T1 --> KC[Kafka Connect<br/>topics.regex pattern]
    T2 --> KC
    T3 --> KC

    style V3 fill:#4caf50,color:#fff
```

**Benefits**:

- Backward compatibility
- Forward compatibility
- Zero-downtime schema updates
- Parallel version support

### 3. Incremental Processing

```mermaid
graph LR
    S1[Snapshot 1<br/>Files: A, B] --> S2[Snapshot 2<br/>Files: A, B, C]
    S2 --> S3[Snapshot 3<br/>Files: A, B, C, D]

    S3 --> P[Spark Job:<br/>Process only D]

    style S3 fill:#7e57c2,color:#fff
    style P fill:#4caf50,color:#fff
```

**Benefits**:

- Process only new data
- Reduce computation time
- Lower resource usage
- Maintain complete history

[:octicons-arrow-right-24: System Design](system-design.md)

---

## Next Steps

You can learn more about specific architectural components:

- **[System Design](system-design.md)** - Detailed component design and interactions
- **[Avro Serialization](avro-serialization.md)** - Schema registry pattern and evolution
- **[Data Flow](data-flow.md)** - End-to-end data pipeline with examples

Or explore related topics:

- **[Deployment Guide](../deployment/index.md)** - How to deploy the architecture
- **[Monitoring](../monitoring/index.md)** - Observability and metrics
- **[Configuration](../usage/configuration.md)** - Tune the components
