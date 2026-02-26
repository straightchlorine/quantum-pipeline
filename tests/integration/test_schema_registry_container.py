"""Integration tests for SchemaRegistry using testcontainers.

Spins up real Kafka and Confluent Schema Registry containers to validate
the SchemaRegistry class against a live registry instance.
"""

import json

import pytest
import requests
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs
from testcontainers.kafka import KafkaContainer

from quantum_pipeline.utils.schema_registry import SchemaRegistry

# ---------------------------------------------------------------------------
# Minimal valid Avro schema used across tests
# ---------------------------------------------------------------------------
TEST_SCHEMA = {
    'type': 'record',
    'name': 'VQEProcess',
    'fields': [
        {'name': 'iteration', 'type': 'int'},
        {'name': 'parameters', 'type': {'type': 'array', 'items': 'double'}},
        {'name': 'result', 'type': 'double'},
        {'name': 'std', 'type': 'double'},
    ],
}


# ---------------------------------------------------------------------------
# Container fixtures (session-scoped for speed)
# ---------------------------------------------------------------------------
@pytest.fixture(scope='session')
def docker_network():
    """Create a shared Docker network for inter-container communication."""
    import docker

    client = docker.from_env()
    network = client.networks.create('test-network-sr', driver='bridge')
    yield network
    network.remove()


@pytest.fixture(scope='session')
def kafka_container(docker_network):
    """Start a Kafka container for the entire test session."""
    kafka = KafkaContainer('confluentinc/cp-kafka:7.6.0')
    kafka.start()
    # Connect Kafka to the shared network for inter-container communication
    docker_network.connect(kafka.get_wrapped_container().id)
    yield kafka
    kafka.stop()


@pytest.fixture(scope='session')
def schema_registry_container(kafka_container, docker_network):
    """Start a Confluent Schema Registry container linked to Kafka.

    The Schema Registry connects to Kafka's internal BROKER listener (port 9092)
    via the shared Docker network, not the host-mapped PLAINTEXT listener.
    """
    # Get Kafka's internal IP address on the shared network
    kafka_wrapper = kafka_container.get_wrapped_container()
    kafka_wrapper.reload()  # refresh attrs after network connect
    kafka_internal_host = kafka_wrapper.attrs['NetworkSettings']['Networks']['test-network-sr'][
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
def schema_registry_url(schema_registry_container):
    """Return the external URL for the running Schema Registry."""
    host = schema_registry_container.get_container_host_ip()
    port = schema_registry_container.get_exposed_port(8081)
    return f'http://{host}:{port}'


@pytest.fixture
def schema_registry(schema_registry_url, monkeypatch):
    """Provide a SchemaRegistry instance wired to the testcontainer URL."""
    monkeypatch.setattr(
        'quantum_pipeline.configs.settings.SCHEMA_REGISTRY_URL',
        schema_registry_url,
    )
    monkeypatch.setattr(
        'quantum_pipeline.utils.schema_registry.SCHEMA_REGISTRY_URL',
        schema_registry_url,
    )
    return SchemaRegistry()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _register_schema(base_url: str, subject: str, schema_dict: dict) -> requests.Response:
    """Register an Avro schema directly via the REST API."""
    return requests.post(
        f'{base_url}/subjects/{subject}-value/versions',
        headers={'Content-Type': 'application/vnd.schemaregistry.v1+json'},
        json={'schema': json.dumps(schema_dict)},
        timeout=10,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestSchemaRegistryContainer:
    """Tests that exercise SchemaRegistry against real containers."""

    def test_registry_is_available(self, schema_registry_url):
        """GET /subjects should return 200 when the registry is healthy."""
        resp = requests.get(f'{schema_registry_url}/subjects', timeout=10)
        assert resp.status_code == 200

    def test_register_schema(self, schema_registry_url):
        """POST a schema and expect a 200 response containing an id."""
        resp = _register_schema(schema_registry_url, 'test_register', TEST_SCHEMA)
        assert resp.status_code == 200
        body = resp.json()
        assert 'id' in body
        assert isinstance(body['id'], int)

    def test_fetch_schema_from_registry(self, schema_registry_url):
        """Register then fetch a schema; content must round-trip correctly."""
        subject = 'test_fetch'
        _register_schema(schema_registry_url, subject, TEST_SCHEMA)

        resp = requests.get(
            f'{schema_registry_url}/subjects/{subject}-value/versions/latest',
            timeout=10,
        )
        assert resp.status_code == 200
        fetched = json.loads(resp.json()['schema'])
        assert fetched['name'] == TEST_SCHEMA['name']
        assert fetched['type'] == TEST_SCHEMA['type']
        assert len(fetched['fields']) == len(TEST_SCHEMA['fields'])

    def test_schema_caching(self, schema_registry, schema_registry_url):
        """After the first get_schema call the result must be in the cache."""
        subject = 'test_caching'
        _register_schema(schema_registry_url, subject, TEST_SCHEMA)

        # First call - populates cache
        result1 = schema_registry._get_schema_from_registry(subject)
        assert result1 is not None
        assert subject in schema_registry.schema_cache

        # Second call - verify cache is populated (content unchanged)
        cached = schema_registry.schema_cache[subject]
        result2 = schema_registry._get_schema_from_cache(subject)
        assert result2 is not None
        assert json.loads(cached) if isinstance(cached, str) else cached == result2

    def test_schema_versioning(self, schema_registry_url):
        """Register v1, then v2 (with an extra field); latest must be v2."""
        subject = 'test_versioning'

        # v1
        _register_schema(schema_registry_url, subject, TEST_SCHEMA)

        # v2 - add a nullable field with a default
        v2_schema = {
            **TEST_SCHEMA,
            'fields': [
                *TEST_SCHEMA['fields'],
                {'name': 'metadata', 'type': ['null', 'string'], 'default': None},
            ],
        }
        resp_v2 = _register_schema(schema_registry_url, subject, v2_schema)
        assert resp_v2.status_code == 200

        # Fetch latest
        resp = requests.get(
            f'{schema_registry_url}/subjects/{subject}-value/versions/latest',
            timeout=10,
        )
        assert resp.status_code == 200
        latest = json.loads(resp.json()['schema'])
        field_names = [f['name'] for f in latest['fields']]
        assert 'metadata' in field_names, 'Latest schema should contain the v2 field'

    def test_get_schema_with_real_registry(self, schema_registry, schema_registry_url):
        """SchemaRegistry.get_schema() should work against the live registry."""
        subject = 'test_get_real'
        _register_schema(schema_registry_url, subject, TEST_SCHEMA)

        result = schema_registry.get_schema(subject)
        assert result['name'] == TEST_SCHEMA['name']
        assert result['type'] == 'record'

    def test_save_schema_to_real_registry(
        self, schema_registry, schema_registry_url, tmp_path, monkeypatch
    ):
        """SchemaRegistry.save_schema() should persist to the live registry."""
        # Point schema_dir to a temp location so file writes don't pollute the tree
        monkeypatch.setattr(schema_registry, 'schema_dir', tmp_path)

        subject = 'test_save_real'
        schema_registry.save_schema(subject, TEST_SCHEMA)

        # Verify it landed in the registry
        resp = requests.get(
            f'{schema_registry_url}/subjects/{subject}-value/versions/latest',
            timeout=10,
        )
        assert resp.status_code == 200
        saved = json.loads(resp.json()['schema'])
        assert saved['name'] == TEST_SCHEMA['name']

        # Verify local file was written
        local_file = tmp_path / f'{subject}.avsc'
        assert local_file.exists()

    def test_file_fallback_when_registry_down(
        self, schema_registry_container, schema_registry_url, monkeypatch
    ):
        """When the registry is stopped, get_schema falls back to local .avsc files."""
        # Use an unreachable URL to simulate the registry being down,
        # rather than actually stopping the session-scoped container.
        monkeypatch.setattr(
            'quantum_pipeline.configs.settings.SCHEMA_REGISTRY_URL',
            'http://127.0.0.1:1',  # unreachable port
        )
        sr = SchemaRegistry()

        # "vqe_process" has a local .avsc file in the repo
        result = sr.get_schema('vqe_process')
        assert result is not None
        assert result['name'] == 'VQEProcess'
        assert result['type'] == 'record'
