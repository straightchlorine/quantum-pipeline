# Kafka Streaming

Apache Kafka 4.2.0 handles real-time ingestion of VQE simulation results,
decoupling producers from consumers with durable, ordered delivery. Kafka runs
in KRaft mode (no ZooKeeper dependency).

For how Kafka fits into the overall architecture, see
[System Design](../architecture/system-design.md#apache-kafka-integration).
For schema definitions and the Avro wire format, see
[Serialization](../architecture/serialization.md).

## Topic Design

All VQE results are published to a single topic: `experiment.vqe`. Previous
versions used dynamic per-molecule topics (`vqe_decorated_result_{suffix}`).
The current design consolidates everything into one topic, simplifying consumer
configuration and Schema Registry management. Simulation parameters (molecule,
basis set, backend) are encoded in the message payload.

## Kafka Cluster (KRaft Mode)

A single-node KRaft setup where one node acts as both broker and controller.
Replication factors are 1 since there is only one broker. The internal listener
(`PLAINTEXT`) is used by containers on the Docker network, and the external
listener is exposed for development access from the host.

The full Kafka service definition is in
[`compose/docker-compose.ml.yaml`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/compose/docker-compose.ml.yaml#L104).

| Listener | Port | Purpose |
|----------|------|---------|
| `PLAINTEXT` | `9092` | Internal, used by containers on the Docker network |
| `CONTROLLER` | `9093` | KRaft controller communication |
| `EXTERNAL` | `9094` | Host access for development |

## Schema Registry

Confluent Schema Registry 8.2.0 manages Avro schema lifecycle with automatic
versioning. The default compatibility mode is `BACKWARD`. The registry stores
schemas on the internal `_schemas` topic.

The service definition is in
[`compose/docker-compose.ml.yaml`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/compose/docker-compose.ml.yaml#L136).

For details on compatibility modes, see the
[Schema Registry documentation](https://docs.confluent.io/platform/current/schema-registry/fundamentals/schema-evolution.html).

## Producer Configuration

[`VQEKafkaProducer`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/stream/kafka_interface.py#L20)
wraps `kafka-python`'s `KafkaProducer`, adding Avro serialization and a
two-level retry mechanism. On initialization it builds the security config,
connects to Schema Registry, and sets up the
[`VQEDecoratedResultInterface`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/stream/serialization/interfaces/vqe.py)
serializer.

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `bootstrap_servers` | `localhost:9092` | Kafka broker addresses (overridden by `KAFKA_SERVERS` env var at CLI level) |
| `kafka_retries` | `5` | Client-level retries handled by `kafka-python` for transient broker errors |
| `retries` | `3` | Application-level retries in `_send_with_retry()`, with configurable `retry_delay` (default 2s) |
| `acks` | `all` | Acknowledgment level required from brokers |
| `timeout` | `10` | Timeout in seconds for each `send().get()` call |

There are two retry layers: `kafka_retries` is passed to the `KafkaProducer`
constructor for low-level retries. `retries` controls the outer application
loop in `_send_with_retry()`, which catches `KafkaError` exceptions and retries
with a delay.

The [`ProducerConfig`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/configs/module/producer.py)
dataclass holds all producer settings and can be constructed directly or via
`ProducerConfig.from_dict()`.

## Message Format

Messages use the Confluent Wire Format, prepending metadata to the Avro binary
payload for schema lookup during deserialization.

| Component | Size | Description |
|-----------|------|-------------|
| Magic Byte | 1 byte | Always `0x00`, identifies Confluent wire format |
| Schema ID | 4 bytes | Big-endian integer referencing the Schema Registry |
| Avro Binary | Variable | Compact binary-encoded `VQEDecoratedResult` |

The serialization is handled by `to_avro_bytes()` on
[`AvroInterfaceBase`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/stream/serialization/interfaces/vqe.py).
It prepends the Confluent wire header when the schema ID is found in the
registry cache, and falls back to headerless Avro if not.

For schema structure details, see
[Serialization](../architecture/serialization.md).

## Redpanda Connect (S3 Sink)

Redpanda Connect replaces Kafka Connect as the S3 sink. It is a single Go
binary (around 128 MB) compared to Kafka Connect's JVM footprint, which makes
it a better fit for a local deployment.

It subscribes to `experiment.vqe`, decodes each message through Schema Registry
via `schema_registry_decode`, and writes the resulting JSON to the `raw-results`
bucket in Garage.

The full pipeline config is at
[`compose/redpanda-connect.yaml`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/compose/redpanda-connect.yaml):

```yaml
http:
  address: 0.0.0.0:4195

input:
  kafka:
    addresses:
      - kafka:9092
    topics:
      - experiment.vqe
    consumer_group: redpanda-s3-sink

  processors:
    - schema_registry_decode:
        url: http://schema-registry:8081
        avro_raw_json: true

output:
  aws_s3:
    bucket: raw-results
    path: experiments/experiment.vqe/${!count("s3_objects")}-${!timestamp_unix_nano()}.json
    endpoint: http://garage:3901
    region: garage
    force_path_style_urls: true
    credentials:
      id: ${S3_ACCESS_KEY}
      secret: ${S3_SECRET_KEY}
    content_type: application/json
    max_in_flight: 1
    batching:
      count: 1
      period: 10s
```

Output files follow the pattern `{counter}-{unix_nano_timestamp}.json`.

!!! note "Kafka Connect Alternative"
    A Kafka Connect configuration is available as a fallback in
    [`compose/docker-compose.ml.kafka-connect.yaml`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/compose/docker-compose.ml.kafka-connect.yaml).
    It can be enabled by scaling down Redpanda Connect and starting Kafka
    Connect instead.

## Security Configuration

[`KafkaSecurity`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/stream/kafka_security.py)
builds the security dictionary passed to `KafkaProducer`. It supports two
modes, controlled by
[`SecurityConfig`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/configs/module/security.py):

| Mode | Protocol | Authentication | Status |
|------|----------|----------------|--------|
| SSL | `security_protocol: SSL` | Certificate-based (paths from `CertConfig`, resolved relative to `ssl_dir`, default `./secrets/`) | Tested |
| SASL_SSL | `security_protocol: SASL_SSL` | Username/password (PLAIN, SCRAM-SHA-256/512) | Tested |
| SASL_SSL | `security_protocol: SASL_SSL` | Kerberos (GSSAPI) | Untested - options are exposed but no Kerberos setup has been done |

All modes are disabled by default. When neither is enabled,
`build_security_config()` returns an empty dictionary and the producer connects
over PLAINTEXT. The Docker Compose stack uses PLAINTEXT within the container
network.

For the full configuration reference, see
[Configuration](../usage/configuration.md).

## Related Documentation

- [System Design](../architecture/system-design.md) - Full architecture overview
- [Serialization](../architecture/serialization.md) - Schema definitions and wire format
- [Spark Processing](spark-processing.md) - Downstream data consumption
- [Iceberg Storage](iceberg-storage.md) - Where Redpanda Connect writes raw data
- [Configuration](../usage/configuration.md) - Full configuration reference

## References

- [Apache Kafka Documentation](https://kafka.apache.org/documentation/)
- [Confluent Schema Registry](https://docs.confluent.io/platform/current/schema-registry/index.html)
- [Redpanda Connect Documentation](https://docs.redpanda.com/redpanda-connect/about/)
