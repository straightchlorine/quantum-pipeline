"""End-to-end Kafka pipeline integration tests.

Proves the full streaming pipeline works: build realistic VQE results,
serialize to Avro via the real Schema Registry, produce to a real Kafka
broker, consume, deserialize, and verify round-trip fidelity.
"""

import json

import numpy as np
import pytest
import requests

docker = pytest.importorskip('docker', reason='Docker SDK not available')
tc_kafka = pytest.importorskip(
    'testcontainers.kafka', reason='testcontainers[kafka] not installed'
)

from kafka import KafkaConsumer

from quantum_pipeline.configs.module.producer import ProducerConfig
from quantum_pipeline.configs.module.security import SecurityConfig
from quantum_pipeline.stream.kafka_interface import VQEKafkaProducer
from quantum_pipeline.stream.serialization.interfaces.vqe import (
    VQEDecoratedResultInterface,
)
from quantum_pipeline.structures.vqe_observation import (
    VQEDecoratedResult,
    VQEInitialData,
    VQEProcess,
    VQEResult,
)
from quantum_pipeline.utils.schema_registry import SchemaRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_consumer(bootstrap_servers, topic, **kwargs):
    """Create a KafkaConsumer with sensible test defaults."""
    return KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        auto_offset_reset='earliest',
        consumer_timeout_ms=10_000,
        **kwargs,
    )


def _build_decorated_result(
    molecule_symbols=None,
    basis_set='sto-3g',
    molecule_id=0,
    num_iterations=1,
    performance_start=None,
    performance_end=None,
):
    """Build a realistic VQEDecoratedResult for E2E tests."""
    from qiskit.circuit import QuantumCircuit
    from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
    from qiskit_nature.units import DistanceUnit

    if molecule_symbols is None:
        molecule_symbols = ['H', 'H']

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    initial_data = VQEInitialData(
        backend='aer_simulator',
        num_qubits=2,
        hamiltonian=np.array(
            [('ZZ', complex(0.5, 0.0)), ('XI', complex(-0.3, 0.1))],
            dtype=object,
        ),
        num_parameters=4,
        initial_parameters=np.array([0.1, 0.2, 0.3, 0.4]),
        noise_backend='none',
        optimizer='COBYLA',
        ansatz=qc,
        ansatz_reps=2,
        default_shots=1024,
    )

    iteration_list = [
        VQEProcess(
            iteration=i + 1,
            parameters=np.array([0.1 * (i + 1), 0.2 * (i + 1), 0.3, 0.4]),
            result=np.float64(-0.5 - 0.1 * i),
            std=np.float64(0.01),
        )
        for i in range(num_iterations)
    ]

    vqe_result = VQEResult(
        initial_data=initial_data,
        iteration_list=iteration_list,
        minimum=np.float64(-0.7),
        optimal_parameters=np.array([0.15, 0.25, 0.35, 0.45]),
        maxcv=None,
        minimization_time=np.float64(1.23),
    )

    # Build coords: simple diatomic along z-axis
    coords = [[0.0, 0.0, float(i) * 0.735] for i in range(len(molecule_symbols))]

    molecule = MoleculeInfo(
        symbols=molecule_symbols,
        coords=coords,
        multiplicity=1,
        charge=0,
        units=DistanceUnit.ANGSTROM,
        masses=None,
    )

    return VQEDecoratedResult(
        vqe_result=vqe_result,
        molecule=molecule,
        basis_set=basis_set,
        hamiltonian_time=np.float64(0.1),
        mapping_time=np.float64(0.2),
        vqe_time=np.float64(0.9),
        total_time=np.float64(1.2),
        molecule_id=molecule_id,
        performance_start=performance_start,
        performance_end=performance_end,
    )


@pytest.fixture
def pipeline_env(schema_registry_url, monkeypatch, tmp_path):
    """Wire SchemaRegistry to the real containerised registry.

    Monkeypatches both the settings module and the schema_registry module
    bindings so that newly created SchemaRegistry instances connect to
    the testcontainer. Also redirects SCHEMA_DIR to a temp directory to
    avoid polluting the repo with generated .avsc files.
    """
    schemas_dir = tmp_path / 'schemas'
    schemas_dir.mkdir()

    monkeypatch.setattr(
        'quantum_pipeline.configs.settings.SCHEMA_REGISTRY_URL',
        schema_registry_url,
    )
    monkeypatch.setattr(
        'quantum_pipeline.utils.schema_registry.SCHEMA_REGISTRY_URL',
        schema_registry_url,
    )
    monkeypatch.setattr(
        'quantum_pipeline.configs.settings.SCHEMA_DIR',
        schemas_dir,
    )
    monkeypatch.setattr(
        'quantum_pipeline.utils.schema_registry.SCHEMA_DIR',
        schemas_dir,
    )

    return {'schema_registry_url': schema_registry_url, 'schemas_dir': schemas_dir}


@pytest.fixture
def e2e_producer(producer_config, pipeline_env):
    """Create a VQEKafkaProducer wired to the real Kafka + Schema Registry."""
    producer = VQEKafkaProducer(producer_config)
    yield producer
    producer.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestKafkaPipelineE2E:
    """End-to-end tests exercising Kafka + Schema Registry via testcontainers."""

    def test_full_pipeline_produce_consume_roundtrip(
        self, e2e_producer, kafka_bootstrap_servers, producer_config, pipeline_env,
    ):
        """Full round-trip: serialize -> produce -> consume -> deserialize -> verify."""
        result = _build_decorated_result()
        topic = producer_config.topic

        # Serialize via the producer's serializer
        avro_bytes = e2e_producer.serializer.to_avro_bytes(
            result, schema_name=e2e_producer.serializer.schema_name,
        )
        assert isinstance(avro_bytes, bytes)
        assert len(avro_bytes) > 0

        # Produce
        e2e_producer.producer.send(topic, avro_bytes).get(timeout=10)
        e2e_producer.producer.flush()

        # Consume
        consumer = _make_consumer(kafka_bootstrap_servers, topic)
        messages = [msg.value for msg in consumer]
        consumer.close()

        assert avro_bytes in messages

        # Deserialize the consumed bytes
        consumed_bytes = messages[messages.index(avro_bytes)]

        # The bytes include the Confluent wire format header (magic + schema ID)
        # if the schema was registered. Verify the header is present.
        if e2e_producer.registry.id_cache.get(e2e_producer.serializer.schema_name):
            assert consumed_bytes[0:1] == b'\x00', 'Missing Confluent magic byte'
            schema_id = int.from_bytes(consumed_bytes[1:5], byteorder='big')
            assert schema_id > 0, 'Schema ID should be positive'

            # Deserialize using the interface
            deserialized = e2e_producer.serializer.from_avro_bytes(consumed_bytes)

            # Verify key fields match
            assert deserialized.basis_set == result.basis_set
            assert deserialized.molecule_id == result.molecule_id
            assert float(deserialized.total_time) == pytest.approx(float(result.total_time))
            assert float(deserialized.vqe_time) == pytest.approx(float(result.vqe_time))
            assert float(deserialized.hamiltonian_time) == pytest.approx(float(result.hamiltonian_time))
            assert float(deserialized.mapping_time) == pytest.approx(float(result.mapping_time))

            # Verify nested VQEResult fields
            assert float(deserialized.vqe_result.minimum) == pytest.approx(float(result.vqe_result.minimum))
            assert len(deserialized.vqe_result.iteration_list) == len(result.vqe_result.iteration_list)
            np.testing.assert_allclose(
                deserialized.vqe_result.optimal_parameters,
                result.vqe_result.optimal_parameters,
                atol=1e-10,
            )

            # Verify VQEProcess iteration
            orig_proc = result.vqe_result.iteration_list[0]
            deser_proc = deserialized.vqe_result.iteration_list[0]
            assert deser_proc.iteration == orig_proc.iteration
            assert float(deser_proc.result) == pytest.approx(float(orig_proc.result))

            # Verify molecule info
            assert list(deserialized.molecule.symbols) == list(result.molecule.symbols)
            assert deserialized.molecule.charge == result.molecule.charge
            assert deserialized.molecule.multiplicity == result.molecule.multiplicity

    def test_schema_auto_registration_on_produce(
        self, e2e_producer, pipeline_env,
    ):
        """Producing a result should auto-register the schema in the real registry."""
        result = _build_decorated_result()
        sr_url = pipeline_env['schema_registry_url']

        # Serialize (triggers schema registration via the interface property)
        e2e_producer.serializer.to_avro_bytes(
            result, schema_name=e2e_producer.serializer.schema_name,
        )

        # Query the Schema Registry REST API directly to verify registration
        schema_name = e2e_producer.serializer.schema_name
        resp = requests.get(
            f'{sr_url}/subjects/{schema_name}-value/versions/latest',
            timeout=10,
        )
        assert resp.status_code == 200, (
            f'Schema {schema_name} was not auto-registered. '
            f'Response: {resp.status_code} {resp.text}'
        )

        body = resp.json()
        assert 'schema' in body
        assert 'id' in body
        assert isinstance(body['id'], int)

        # The registered schema should be a valid VQEDecoratedResult schema
        registered_schema = json.loads(body['schema'])
        assert registered_schema['name'] == 'VQEDecoratedResult'
        assert registered_schema['type'] == 'record'

        field_names = [f['name'] for f in registered_schema['fields']]
        assert 'vqe_result' in field_names
        assert 'molecule' in field_names
        assert 'basis_set' in field_names

    def test_multiple_observations_batch(
        self, e2e_producer, kafka_bootstrap_servers, pipeline_env,
    ):
        """Send multiple different VQEDecoratedResults, consume all, verify each."""
        topic = 'test-batch-observations'
        test_configs = [
            {'molecule_symbols': ['H', 'H'], 'basis_set': 'sto-3g', 'molecule_id': 0},
            {'molecule_symbols': ['Li', 'H'], 'basis_set': '6-31g', 'molecule_id': 1},
            {'molecule_symbols': ['H', 'H'], 'basis_set': 'cc-pvdz', 'molecule_id': 2, 'num_iterations': 3},
            {'molecule_symbols': ['O', 'H', 'H'], 'basis_set': 'sto-3g', 'molecule_id': 3},
        ]

        sent_bytes = []
        for config in test_configs:
            result = _build_decorated_result(**config)
            avro_bytes = e2e_producer.serializer.to_avro_bytes(
                result, schema_name=e2e_producer.serializer.schema_name,
            )
            e2e_producer.producer.send(topic, avro_bytes).get(timeout=10)
            sent_bytes.append(avro_bytes)

        e2e_producer.producer.flush()

        # Consume all messages
        consumer = _make_consumer(kafka_bootstrap_servers, topic)
        received = [msg.value for msg in consumer]
        consumer.close()

        assert len(received) == len(test_configs), (
            f'Expected {len(test_configs)} messages, got {len(received)}'
        )

        # Verify each message can be deserialized if the Confluent header is present
        schema_name = e2e_producer.serializer.schema_name
        if e2e_producer.registry.id_cache.get(schema_name):
            for i, (config, consumed) in enumerate(zip(test_configs, received)):
                deserialized = e2e_producer.serializer.from_avro_bytes(consumed)
                assert deserialized.basis_set == config['basis_set'], (
                    f'Message {i}: basis_set mismatch'
                )
                assert deserialized.molecule_id == config['molecule_id'], (
                    f'Message {i}: molecule_id mismatch'
                )
                assert list(deserialized.molecule.symbols) == config['molecule_symbols'], (
                    f'Message {i}: molecule symbols mismatch'
                )
                expected_iters = config.get('num_iterations', 1)
                assert len(deserialized.vqe_result.iteration_list) == expected_iters, (
                    f'Message {i}: iteration count mismatch'
                )

        # Verify all sent bytes were received
        for b in sent_bytes:
            assert b in received

    def test_topic_suffix_from_result(
        self, producer_config, pipeline_env,
    ):
        """Verify _update_topic() appends the correct schema suffix."""
        producer = VQEKafkaProducer(producer_config)
        original_topic = producer.config.topic

        result = _build_decorated_result(
            molecule_symbols=['H', 'H'],
            basis_set='sto-3g',
            molecule_id=0,
            num_iterations=1,
        )

        # Call _update_topic which modifies config.topic and serializer.schema_name
        producer._update_topic(result)

        expected_suffix = result.get_schema_suffix()
        assert expected_suffix == '_mol0_HH_it1_bs_sto_3g_bk_aer_simulator'

        # Topic should have the suffix appended
        assert producer.config.topic.endswith(expected_suffix)
        assert producer.serializer.schema_name.endswith(expected_suffix)
        assert producer.serializer.result_interface.schema_name.endswith(expected_suffix)

        # Calling _update_topic again with a different result should replace the suffix
        result2 = _build_decorated_result(
            molecule_symbols=['Li', 'H'],
            basis_set='6-31g',
            molecule_id=1,
            num_iterations=2,
        )
        producer._update_topic(result2)

        expected_suffix2 = result2.get_schema_suffix()
        assert producer.config.topic.endswith(expected_suffix2)
        assert not producer.config.topic.endswith(expected_suffix)

        producer.close()

    def test_avro_schema_compatibility(
        self, pipeline_env,
    ):
        """Register a schema, then verify a backward-compatible evolved version works."""
        sr_url = pipeline_env['schema_registry_url']
        subject = 'test_compat_check'

        # Register base schema (VQEProcess-like)
        base_schema = {
            'type': 'record',
            'name': 'VQEProcessCompat',
            'fields': [
                {'name': 'iteration', 'type': 'int'},
                {'name': 'parameters', 'type': {'type': 'array', 'items': 'double'}},
                {'name': 'result', 'type': 'double'},
                {'name': 'std', 'type': 'double'},
            ],
        }

        resp = requests.post(
            f'{sr_url}/subjects/{subject}-value/versions',
            headers={'Content-Type': 'application/vnd.schemaregistry.v1+json'},
            json={'schema': json.dumps(base_schema)},
            timeout=10,
        )
        assert resp.status_code == 200, f'Failed to register base schema: {resp.text}'
        base_id = resp.json()['id']
        assert isinstance(base_id, int)

        # Register backward-compatible evolved schema (add nullable field with default)
        evolved_schema = {
            'type': 'record',
            'name': 'VQEProcessCompat',
            'fields': [
                {'name': 'iteration', 'type': 'int'},
                {'name': 'parameters', 'type': {'type': 'array', 'items': 'double'}},
                {'name': 'result', 'type': 'double'},
                {'name': 'std', 'type': 'double'},
                {'name': 'convergence_delta', 'type': ['null', 'double'], 'default': None},
            ],
        }

        resp2 = requests.post(
            f'{sr_url}/subjects/{subject}-value/versions',
            headers={'Content-Type': 'application/vnd.schemaregistry.v1+json'},
            json={'schema': json.dumps(evolved_schema)},
            timeout=10,
        )
        assert resp2.status_code == 200, (
            f'Evolved schema should be compatible: {resp2.text}'
        )
        evolved_id = resp2.json()['id']
        assert evolved_id != base_id, 'Evolved schema should get a different ID'

        # Verify both versions exist
        versions_resp = requests.get(
            f'{sr_url}/subjects/{subject}-value/versions',
            timeout=10,
        )
        assert versions_resp.status_code == 200
        versions = versions_resp.json()
        assert len(versions) == 2, f'Expected 2 versions, got {versions}'

        # Verify latest is the evolved schema
        latest_resp = requests.get(
            f'{sr_url}/subjects/{subject}-value/versions/latest',
            timeout=10,
        )
        latest_schema = json.loads(latest_resp.json()['schema'])
        field_names = [f['name'] for f in latest_schema['fields']]
        assert 'convergence_delta' in field_names

    def test_pipeline_with_performance_data(
        self, e2e_producer, kafka_bootstrap_servers, pipeline_env,
    ):
        """Test round-trip with VQEDecoratedResult containing performance data."""
        perf_start = {
            'system': {
                'cpu': {'percent': 25.0, 'count': 8},
                'memory': {'total': 16_000_000_000, 'used': 8_000_000_000, 'percent': 50.0},
            },
            'container_type': 'docker',
            'gpu': [
                {'utilization_gpu': 10.0, 'utilization_memory': 20.0, 'power_draw': 50.0},
            ],
        }
        perf_end = {
            'system': {
                'cpu': {'percent': 75.0, 'count': 8},
                'memory': {'total': 16_000_000_000, 'used': 12_000_000_000, 'percent': 75.0},
            },
            'container_type': 'docker',
            'gpu': [
                {'utilization_gpu': 90.0, 'utilization_memory': 80.0, 'power_draw': 200.0},
            ],
        }

        result = _build_decorated_result(
            performance_start=perf_start,
            performance_end=perf_end,
        )

        # Verify performance delta calculation works
        delta = result.get_performance_delta()
        assert delta['cpu_usage_delta'] == pytest.approx(50.0)
        assert delta['container_type'] == 'docker'
        assert 'gpu_deltas' in delta

        topic = 'test-perf-data'

        # Serialize and produce
        avro_bytes = e2e_producer.serializer.to_avro_bytes(
            result, schema_name=e2e_producer.serializer.schema_name,
        )
        e2e_producer.producer.send(topic, avro_bytes).get(timeout=10)
        e2e_producer.producer.flush()

        # Consume
        consumer = _make_consumer(kafka_bootstrap_servers, topic)
        messages = [msg.value for msg in consumer]
        consumer.close()

        assert avro_bytes in messages

        # Deserialize and verify performance data survives the round trip
        schema_name = e2e_producer.serializer.schema_name
        if e2e_producer.registry.id_cache.get(schema_name):
            consumed = messages[messages.index(avro_bytes)]
            deserialized = e2e_producer.serializer.from_avro_bytes(consumed)

            # Performance data is serialized as JSON strings in Avro
            assert deserialized.performance_start is not None
            assert deserialized.performance_end is not None
            assert deserialized.performance_start['container_type'] == 'docker'
            assert deserialized.performance_end['system']['cpu']['percent'] == 75.0

            # Verify the deserialized result also computes the correct delta
            deser_delta = deserialized.get_performance_delta()
            assert deser_delta['cpu_usage_delta'] == pytest.approx(50.0)
            assert deser_delta['container_type'] == 'docker'
