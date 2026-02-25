# Kafka Streaming

## Overview

**Apache Kafka** handles real-time ingestion of VQE simulation results, decoupling producers from consumers with durable, ordered, exactly-once delivery.

- Receive serialized VQE results from simulation containers
- Validate schemas through **Confluent Schema Registry**
- Buffer messages for reliable delivery
- Feed **Kafka Connect** to write Avro files to MinIO

[:octicons-arrow-right-24: System Design](../architecture/system-design.md) | [:octicons-arrow-right-24: Avro Serialization](../architecture/avro-serialization.md)

---

## Topic Naming Convention

Topics follow a naming pattern encoding the data type and schema version:

```
vqe_decorated_result_{suffix}
```

The suffix is generated automatically by the Schema Registry and encodes the schema version. When the `VQEDecoratedResult` structure changes, a new topic with an updated suffix is created.

### Suffix Components

The topic suffix is derived from the schema version identifier and may include information about:

| Component | Description | Example |
|-----------|-------------|---------|
| Molecule ID | Identifier for the simulated molecule | `1`, `2`, `42` |
| Symbols | Atomic symbols in the molecule | `H2`, `LiH` |
| Iteration Count | Number of optimizer iterations | `100`, `500` |
| Basis Set | Quantum chemistry basis set used | `sto-3g`, `cc-pvdz` |
| Backend | Qiskit simulation backend | `aer_simulator` |

### Example Topic Names

```
vqe_decorated_result_v1
vqe_decorated_result_v2
vqe_decorated_result_v3
```

This naming scheme enables Kafka Connect to subscribe to all VQE result topics using a single regex pattern, regardless of how many schema versions exist.

---

## Schema Versioning

The **Confluent Schema Registry** manages Avro schema lifecycle with automatic versioning and compatibility checking. For details on Schema Registry concepts, see the [Confluent Schema Registry documentation](https://docs.confluent.io/platform/current/schema-registry/index.html).

### Schema Registry Configuration

```yaml
schema-registry:
  image: confluentinc/cp-schema-registry:7.5.0
  environment:
    SCHEMA_REGISTRY_HOST_NAME: schema-registry
    SCHEMA_REGISTRY_KAFKASTORE_BOOTSTRAP_SERVERS: 'kafka:9092'
    SCHEMA_REGISTRY_LISTENERS: 'http://0.0.0.0:8081'
    SCHEMA_REGISTRY_SCHEMA_COMPATIBILITY_LEVEL: 'none'
```

!!! warning "Compatibility Mode"
    The system currently uses `NONE` compatibility mode, allowing unrestricted schema changes. For production deployments, consider stricter modes (`BACKWARD`, `FORWARD`, or `FULL`).

---

## Producer Configuration

The producer is configured for reliability and data integrity.

```python
producer_config = {
    'bootstrap.servers': 'kafka:9092',
    'client.id': 'quantum-pipeline-producer',
    'compression.type': 'snappy',
    'batch.size': 16384,
    'linger.ms': 10,
    'acks': 'all',
    'retries': 3,
    'max.in.flight.requests.per.connection': 5,
    'enable.idempotence': True,
}
```

### Key Parameters Explained

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `acks` | `all` | Waits for all in-sync replicas to acknowledge the write, ensuring no data loss |
| `compression.type` | `snappy` | Fast compression algorithm that reduces network bandwidth with minimal CPU overhead |
| `retries` | `3` | Automatically retries failed sends up to 3 times |
| `enable.idempotence` | `True` | Ensures exactly-once delivery semantics, preventing duplicate messages on retry |
| `batch.size` | `16384` | Batches up to 16 KB of messages before sending, improving throughput |
| `linger.ms` | `10` | Waits up to 10 ms to accumulate messages into a batch before sending |
| `max.in.flight.requests.per.connection` | `5` | Allows up to 5 unacknowledged requests per connection for higher throughput |

!!! info "Idempotent Producer"
    With `enable.idempotence=True`, the producer assigns a sequence number to each message. The broker uses this to detect and discard duplicates, guaranteeing that retries do not result in duplicate records in the topic.

---

## Message Format

Messages use the Confluent Wire Format, prepending metadata to the Avro binary payload for schema lookup during deserialization.

### Wire Format Structure

```
+------------------+------------------+---------------------------+
| Magic Byte (1B)  | Schema ID (4B)   | Avro Binary Payload       |
| 0x00             | Big-endian int   | Serialized VQEDecResult   |
+------------------+------------------+---------------------------+
```

| Component | Size | Description |
|-----------|------|-------------|
| Magic Byte | 1 byte | Always `0x00`, identifies Confluent wire format |
| Schema ID | 4 bytes | Big-endian integer referencing the Schema Registry |
| Avro Binary | Variable | Compact binary-encoded `VQEDecoratedResult` |

### Serialization Code

```python
def to_avro_bytes(self, obj: T, schema_name: str = 'vqe_decorated_result') -> bytes:
    """Convert object to Avro binary format with Confluent wire header."""
    schema = self.schema
    parsed_schema = avro.schema.parse(json.dumps(schema))

    writer = DatumWriter(parsed_schema)
    bytes_writer = io.BytesIO()

    # Confluent wire format header
    bytes_writer.write(bytes([0]))  # Magic byte
    bytes_writer.write(self.registry.id_cache[schema_name].to_bytes(4, byteorder='big'))

    encoder = BinaryEncoder(bytes_writer)
    writer.write(self.serialize(obj), encoder)
    return bytes_writer.getvalue()
```

[:octicons-arrow-right-24: Avro Serialization](../architecture/avro-serialization.md)

---

## Dynamic Topic Subscription

**Kafka Connect** uses regex-based subscription to discover and consume all VQE result topics automatically.

### Configuration

```json
{
  "topics.regex": "vqe_decorated_result_.*",
  "refresh.topics.enabled": "true"
}
```

### How It Works

1. Kafka Connect periodically scans the broker for topics matching the regex pattern `vqe_decorated_result_.*`.
2. When a new schema version creates a new topic (e.g., `vqe_decorated_result_v3`), Kafka Connect detects it automatically.
3. The connector begins consuming from the new topic without any configuration changes or restarts.
4. Messages from all matched topics are written to MinIO with a directory structure that separates data by topic name.

### Benefits

- Zero-downtime schema evolution
- No connector reconfiguration needed
- Parallel version support across topics
- Blue-green deployment compatibility

!!! tip "Parallel Version Support"
    Multiple simulation versions can run in parallel (e.g., A/B testing or canary releases) without data structure conflicts.

---

## Security and Cluster Configuration

Kafka configuration follows Confluent best practices. For SSL/TLS encryption, SASL authentication, and production cluster tuning, see the [Confluent Security documentation](https://docs.confluent.io/platform/current/security/index.html) and [Broker Configuration reference](https://docs.confluent.io/platform/current/installation/configuration/broker-configs.html).

[:octicons-arrow-right-24: Configuration Reference](../usage/configuration.md)

---

## Related Documentation

- [System Design](../architecture/system-design.md) - Full architecture overview
- [Avro Serialization](../architecture/avro-serialization.md) - Schema definitions and wire format details
- [Spark Processing](spark-processing.md) - How Spark consumes data downstream of Kafka
- [Iceberg Storage](iceberg-storage.md) - Where Kafka Connect writes raw data
- [Configuration](../usage/configuration.md) - Full configuration reference including security settings

## References

- [Apache Kafka Documentation](https://kafka.apache.org/documentation/)
- [Confluent Schema Registry](https://docs.confluent.io/platform/current/schema-registry/index.html)
- [Confluent Kafka Security](https://docs.confluent.io/platform/current/security/index.html)
- [Confluent Broker Configuration](https://docs.confluent.io/platform/current/installation/configuration/broker-configs.html)
