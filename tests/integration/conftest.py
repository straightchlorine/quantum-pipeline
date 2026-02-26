"""Shared testcontainers fixtures for integration tests.

Provides session-scoped Kafka and Schema Registry containers
that can be reused across all integration test modules.
"""

import pytest

docker = pytest.importorskip('docker', reason='Docker SDK not available')
tc_kafka = pytest.importorskip(
    'testcontainers.kafka', reason='testcontainers[kafka] not installed'
)

from testcontainers.core.container import DockerContainer  # noqa: E402
from testcontainers.core.waiting_utils import wait_for_logs  # noqa: E402
from testcontainers.kafka import KafkaContainer  # noqa: E402

from quantum_pipeline.configs.module.producer import ProducerConfig  # noqa: E402
from quantum_pipeline.configs.module.security import SecurityConfig  # noqa: E402


@pytest.fixture(scope='session')
def docker_network():
    """Create a shared Docker network for inter-container communication."""
    client = docker.from_env()
    network = client.networks.create('test-network', driver='bridge')
    yield network
    network.remove()


@pytest.fixture(scope='session')
def kafka_container(docker_network):
    """Start a single Kafka container shared across the entire test session."""
    kafka = KafkaContainer()
    kafka.start()
    # Connect Kafka to the shared network for inter-container communication
    docker_network.connect(kafka.get_wrapped_container().id)
    yield kafka
    kafka.stop()


@pytest.fixture(scope='session')
def schema_registry_container(kafka_container, docker_network):
    """Start a Confluent Schema Registry container linked to the Kafka container.

    The Schema Registry connects to Kafka's internal BROKER listener (port 9092)
    via the shared Docker network, not the host-mapped PLAINTEXT listener.
    """
    # Get Kafka's internal IP address on the shared network
    kafka_wrapper = kafka_container.get_wrapped_container()
    kafka_wrapper.reload()  # refresh attrs after network connect
    kafka_internal_host = kafka_wrapper.attrs['NetworkSettings']['Networks']['test-network'][
        'IPAddress'
    ]
    # The BROKER listener on port 9092 uses PLAINTEXT security protocol
    # (KAFKA_LISTENER_SECURITY_PROTOCOL_MAP=BROKER:PLAINTEXT).
    # Schema Registry requires the actual security protocol, not the listener name.
    kafka_internal_address = f'PLAINTEXT://{kafka_internal_host}:9092'

    sr = DockerContainer('confluentinc/cp-schema-registry:7.6.0')
    sr.with_env('SCHEMA_REGISTRY_HOST_NAME', 'schema-registry')
    sr.with_env(
        'SCHEMA_REGISTRY_KAFKASTORE_BOOTSTRAP_SERVERS',
        kafka_internal_address,
    )
    sr.with_env('SCHEMA_REGISTRY_LISTENERS', 'http://0.0.0.0:8081')
    sr.with_exposed_ports(8081)
    sr.start()
    # Connect Schema Registry to the same network
    docker_network.connect(sr.get_wrapped_container().id)
    wait_for_logs(sr, 'Server started', timeout=60)
    yield sr
    sr.stop()


@pytest.fixture(scope='session')
def kafka_bootstrap_servers(kafka_container):
    """Return the broker address for the running Kafka container."""
    return kafka_container.get_bootstrap_server()


@pytest.fixture(scope='session')
def schema_registry_url(schema_registry_container):
    """Return the external URL for the running Schema Registry."""
    host = schema_registry_container.get_container_host_ip()
    port = schema_registry_container.get_exposed_port(8081)
    return f'http://{host}:{port}'


@pytest.fixture
def producer_config(kafka_bootstrap_servers):
    """Build a ProducerConfig pointing at the containerised broker."""
    return ProducerConfig(
        servers=kafka_bootstrap_servers,
        topic='test-e2e-pipeline',
        security=SecurityConfig.get_default(),
    )
