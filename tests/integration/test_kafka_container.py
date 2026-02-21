"""Testcontainers-based Kafka integration tests.

Tests VQEKafkaProducer against a real Kafka broker running in Docker
via testcontainers. Schema Registry interactions are mocked since those
are covered separately in test_schema_registry_container.py.
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

docker = pytest.importorskip('docker', reason='Docker SDK not available')
tc_kafka = pytest.importorskip(
    'testcontainers.kafka', reason='testcontainers[kafka] not installed'
)

from kafka import KafkaConsumer, KafkaProducer
from testcontainers.kafka import KafkaContainer

from quantum_pipeline.configs.module.producer import ProducerConfig
from quantum_pipeline.configs.module.security import SecurityConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def kafka_container():
    """Start a single Kafka container shared across the entire test session."""
    with KafkaContainer() as kafka:
        yield kafka


@pytest.fixture
def bootstrap_server(kafka_container):
    """Return the broker address for the running Kafka container."""
    return kafka_container.get_bootstrap_server()


@pytest.fixture
def producer_config(bootstrap_server):
    """Build a ProducerConfig pointing at the containerised broker."""
    return ProducerConfig(
        servers=bootstrap_server,
        topic='test-vqe-topic',
        security=SecurityConfig.get_default(),
    )


@pytest.fixture
def mock_schema_registry():
    """Provide a mock SchemaRegistry so tests don't need a real registry."""
    registry = MagicMock()
    registry.id_cache = {}
    registry.schema_cache = {}
    registry.is_schema_registry_available.return_value = False
    registry.registry_schema_existence = {}
    # Force serializer interfaces to use their fallback inline schemas
    registry.get_schema.side_effect = FileNotFoundError('mocked: no schema')
    registry.save_schema = MagicMock()  # no-op
    return registry


def _make_consumer(bootstrap_server, topic, **kwargs):
    """Helper to create a KafkaConsumer with sensible test defaults."""
    return KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_server,
        auto_offset_reset='earliest',
        consumer_timeout_ms=10_000,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Helpers: realistic VQE data builders
# ---------------------------------------------------------------------------

def _build_vqe_decorated_result():
    """Build a realistic VQEDecoratedResult for integration tests."""
    from qiskit.circuit import QuantumCircuit
    from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
    from qiskit_nature.units import DistanceUnit

    from quantum_pipeline.structures.vqe_observation import (
        VQEDecoratedResult,
        VQEInitialData,
        VQEProcess,
        VQEResult,
    )

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    initial_data = VQEInitialData(
        backend='aer_simulator',
        num_qubits=2,
        hamiltonian=np.array([('ZZ', complex(0.5, 0.0)), ('XI', complex(-0.3, 0.1))], dtype=object),
        num_parameters=4,
        initial_parameters=np.array([0.1, 0.2, 0.3, 0.4]),
        noise_backend='none',
        optimizer='COBYLA',
        ansatz=qc,
        ansatz_reps=2,
        default_shots=1024,
    )

    iteration_list = [
        VQEProcess(iteration=1, parameters=np.array([0.1, 0.2, 0.3, 0.4]), result=np.float64(-0.5), std=np.float64(0.01)),
    ]

    vqe_result = VQEResult(
        initial_data=initial_data,
        iteration_list=iteration_list,
        minimum=np.float64(-0.7),
        optimal_parameters=np.array([0.15, 0.25, 0.35, 0.45]),
        maxcv=None,
        minimization_time=np.float64(1.23),
    )

    molecule = MoleculeInfo(
        symbols=['H', 'H'],
        coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.735]],
        multiplicity=1,
        charge=0,
        units=DistanceUnit.ANGSTROM,
        masses=None,
    )

    return VQEDecoratedResult(
        vqe_result=vqe_result,
        molecule=molecule,
        basis_set='sto-3g',
        hamiltonian_time=np.float64(0.1),
        mapping_time=np.float64(0.2),
        vqe_time=np.float64(0.9),
        total_time=np.float64(1.2),
        molecule_id=0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestKafkaContainerIntegration:
    """Integration tests using a real Kafka broker via testcontainers."""

    def test_producer_connects_to_real_kafka(self, producer_config, mock_schema_registry):
        """VQEKafkaProducer should initialise without error against a live broker."""
        with patch('quantum_pipeline.stream.kafka_interface.SchemaRegistry', return_value=mock_schema_registry):
            from quantum_pipeline.stream.kafka_interface import VQEKafkaProducer

            producer = VQEKafkaProducer(producer_config)
            assert producer.producer is not None
            producer.close()

    def test_produce_and_consume_message(self, bootstrap_server):
        """Raw bytes produced to a topic should be consumable."""
        topic = 'test-raw-bytes'
        payload = b'hello-quantum-pipeline'

        # Produce
        kp = KafkaProducer(bootstrap_servers=bootstrap_server)
        kp.send(topic, payload).get(timeout=10)
        kp.flush()
        kp.close()

        # Consume
        consumer = _make_consumer(bootstrap_server, topic)
        messages = [msg.value for msg in consumer]
        consumer.close()

        assert payload in messages

    def test_produce_vqe_result(self, producer_config, bootstrap_server, mock_schema_registry):
        """Serialize a VQEDecoratedResult to Avro, produce it, consume and verify."""
        result = _build_vqe_decorated_result()

        with patch('quantum_pipeline.stream.kafka_interface.SchemaRegistry', return_value=mock_schema_registry):
            from quantum_pipeline.stream.kafka_interface import VQEKafkaProducer

            producer = VQEKafkaProducer(producer_config)

            # Serialize using the producer's serializer (no registry header since id_cache is empty)
            avro_bytes = producer.serializer.to_avro_bytes(result)
            assert isinstance(avro_bytes, bytes)
            assert len(avro_bytes) > 0

            # Produce the serialized bytes through the underlying kafka-python producer
            expected_topic = producer_config.topic
            producer.producer.send(expected_topic, avro_bytes).get(timeout=10)
            producer.producer.flush()

            # Consume and verify
            consumer = _make_consumer(bootstrap_server, expected_topic)
            messages = [msg.value for msg in consumer]
            consumer.close()

            assert avro_bytes in messages
            producer.close()

    def test_multiple_messages(self, bootstrap_server):
        """Multiple messages should be received in order."""
        topic = 'test-multi-msg'
        payloads = [f'msg-{i}'.encode() for i in range(5)]

        kp = KafkaProducer(bootstrap_servers=bootstrap_server)
        for p in payloads:
            kp.send(topic, p).get(timeout=10)
        kp.flush()
        kp.close()

        consumer = _make_consumer(bootstrap_server, topic)
        received = [msg.value for msg in consumer]
        consumer.close()

        assert received == payloads

    def test_context_manager_with_real_kafka(self, producer_config, mock_schema_registry):
        """VQEKafkaProducer should work correctly as a context manager."""
        with patch('quantum_pipeline.stream.kafka_interface.SchemaRegistry', return_value=mock_schema_registry):
            from quantum_pipeline.stream.kafka_interface import VQEKafkaProducer

            with VQEKafkaProducer(producer_config) as producer:
                assert producer.producer is not None
                # send a trivial message to confirm the producer is functional
                producer.producer.send(producer_config.topic, b'ctx-mgr-test').get(timeout=10)
                producer.producer.flush()

            # After exiting the context, the underlying producer should be closed
            # (kafka-python sets _closed to True)
            assert producer.producer._closed

    def test_topic_creation(self, bootstrap_server):
        """Producing to a new topic should auto-create it on the broker."""
        topic = 'test-auto-create-topic'
        payload = b'auto-create'

        kp = KafkaProducer(bootstrap_servers=bootstrap_server)
        kp.send(topic, payload).get(timeout=10)
        kp.flush()
        kp.close()

        consumer = _make_consumer(bootstrap_server, topic)
        messages = [msg.value for msg in consumer]
        consumer.close()

        assert payload in messages

    def test_producer_close_cleanup(self, producer_config, mock_schema_registry):
        """Calling close() should cleanly shut down without errors."""
        with patch('quantum_pipeline.stream.kafka_interface.SchemaRegistry', return_value=mock_schema_registry):
            from quantum_pipeline.stream.kafka_interface import VQEKafkaProducer

            producer = VQEKafkaProducer(producer_config)
            assert producer.producer is not None

            # close() should not raise
            producer.close()
            assert producer.producer._closed

            # calling close() again should also be safe (producer is already None-guarded)
            producer.close()
