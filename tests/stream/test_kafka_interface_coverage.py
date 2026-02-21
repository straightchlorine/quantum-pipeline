"""Additional coverage tests for kafka_interface.py.

Covers error-handling paths, serialization failures, retry logic,
context-manager protocol, and the _update_topic helper that the
existing test_kafka_interface.py does not exercise deeply.
"""

from unittest.mock import Mock, patch, MagicMock

import pytest
from kafka.errors import KafkaError, NoBrokersAvailable

from quantum_pipeline.configs.module.producer import ProducerConfig
from quantum_pipeline.configs.module.security import SecurityConfig
from quantum_pipeline.stream.kafka_interface import KafkaProducerError, VQEKafkaProducer
from quantum_pipeline.structures.vqe_observation import VQEDecoratedResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_config():
    """Minimal mock ProducerConfig for unit tests."""
    cfg = Mock(spec=ProducerConfig)
    cfg.servers = ["localhost:9092"]
    cfg.topic = "test_topic"
    cfg.retries = 3
    cfg.retry_delay = 0          # no real sleep in tests
    cfg.kafka_retries = 3
    cfg.acks = "all"
    cfg.timeout = 10

    cfg.security = Mock(spec=SecurityConfig)
    cfg.security.ssl = False
    cfg.security.sasl_ssl = False
    cfg.security.ssl_check_hostname = True
    return cfg


@pytest.fixture
def producer(mock_config):
    """Return VQEKafkaProducer with all externals mocked out."""
    with (
        patch(
            "quantum_pipeline.stream.kafka_interface.KafkaSecurity",
            autospec=True,
        ) as mock_sec,
        patch(
            "quantum_pipeline.stream.kafka_interface.SchemaRegistry",
            autospec=True,
        ),
        patch(
            "quantum_pipeline.stream.kafka_interface.VQEDecoratedResultInterface",
            autospec=True,
        ),
        patch(
            "quantum_pipeline.stream.kafka_interface.KafkaProducer",
            autospec=True,
        ),
    ):
        mock_sec.return_value.build_security_config.return_value = {}
        p = VQEKafkaProducer(mock_config)
        # Replace the real KafkaProducer with a fresh Mock for assertions
        p.producer = Mock()
        yield p


@pytest.fixture
def mock_result():
    """Mock VQEDecoratedResult."""
    r = Mock(spec=VQEDecoratedResult)
    r.get_schema_suffix.return_value = "_mol_H2_sto3g"
    return r


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInitialization:
    """Test producer initialization error paths."""

    def test_no_brokers_raises_custom_error(self, mock_config):
        with (
            patch(
                "quantum_pipeline.stream.kafka_interface.KafkaSecurity",
                autospec=True,
            ) as mock_sec,
            patch(
                "quantum_pipeline.stream.kafka_interface.SchemaRegistry",
                autospec=True,
            ),
            patch(
                "quantum_pipeline.stream.kafka_interface.VQEDecoratedResultInterface",
                autospec=True,
            ),
            patch(
                "quantum_pipeline.stream.kafka_interface.KafkaProducer",
                side_effect=NoBrokersAvailable(),
            ),
        ):
            mock_sec.return_value.build_security_config.return_value = {}
            with pytest.raises(KafkaProducerError):
                VQEKafkaProducer(mock_config)

    def test_generic_init_error_raises_custom_error(self, mock_config):
        with (
            patch(
                "quantum_pipeline.stream.kafka_interface.KafkaSecurity",
                autospec=True,
            ) as mock_sec,
            patch(
                "quantum_pipeline.stream.kafka_interface.SchemaRegistry",
                autospec=True,
            ),
            patch(
                "quantum_pipeline.stream.kafka_interface.VQEDecoratedResultInterface",
                autospec=True,
            ),
            patch(
                "quantum_pipeline.stream.kafka_interface.KafkaProducer",
                side_effect=RuntimeError("cfg boom"),
            ),
        ):
            mock_sec.return_value.build_security_config.return_value = {}
            with pytest.raises(KafkaProducerError):
                VQEKafkaProducer(mock_config)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    """Test _serialize_result paths."""

    def test_serialize_delegates_to_serializer(self, producer, mock_result):
        producer.serializer.to_avro_bytes.return_value = b"\x00\x01"
        producer.serializer.schema_name = "test_schema"
        result = producer._serialize_result(mock_result)
        assert result == b"\x00\x01"
        producer.serializer.to_avro_bytes.assert_called_once()

    def test_serialize_failure_raises_producer_error(self, producer, mock_result):
        producer.serializer.schema_name = "test_schema"
        producer.serializer.to_avro_bytes.side_effect = Exception("avro boom")
        with pytest.raises(KafkaProducerError):
            producer._serialize_result(mock_result)


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

class TestSendWithRetry:
    """Test _send_with_retry paths."""

    def _make_metadata(self, topic="t", partition=0, offset=1):
        m = Mock()
        m.topic = topic
        m.partition = partition
        m.offset = offset
        return m

    def test_send_succeeds_first_try(self, producer):
        future = Mock()
        future.get.return_value = self._make_metadata()
        producer.producer.send.return_value = future

        assert producer._send_with_retry(b"data") is True
        producer.producer.send.assert_called_once()

    @patch("quantum_pipeline.stream.kafka_interface.sleep")
    def test_send_retries_on_kafka_error(self, mock_sleep, producer):
        """Should retry on KafkaError and succeed on 2nd attempt."""
        good_future = Mock()
        good_future.get.return_value = self._make_metadata()

        bad_future = Mock()
        bad_future.get.side_effect = KafkaError("transient")

        producer.producer.send.side_effect = [bad_future, good_future]
        # The first call raises via future.get, second succeeds
        # Actually, send returns the future, then .get() raises
        # We need send to return the future objects
        producer.producer.send.side_effect = None
        producer.producer.send.return_value = bad_future

        # First attempt fails, should retry
        # On second attempt, return good future
        call_count = 0
        def send_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return bad_future
            return good_future
        producer.producer.send.side_effect = send_side_effect

        assert producer._send_with_retry(b"data") is True
        assert call_count == 2
        mock_sleep.assert_called_once()

    @patch("quantum_pipeline.stream.kafka_interface.sleep")
    def test_send_exhausts_retries_raises(self, mock_sleep, producer):
        """After exhausting retries, should raise KafkaProducerError."""
        bad_future = Mock()
        bad_future.get.side_effect = KafkaError("persistent failure")
        producer.producer.send.return_value = bad_future

        with pytest.raises(KafkaProducerError):
            producer._send_with_retry(b"data")

    def test_unexpected_error_raises_immediately(self, producer):
        """Non-KafkaError should raise immediately without retry."""
        bad_future = Mock()
        bad_future.get.side_effect = RuntimeError("unexpected")
        producer.producer.send.return_value = bad_future

        with pytest.raises(KafkaProducerError):
            producer._send_with_retry(b"data")
        # Should only attempt once
        assert producer.producer.send.call_count == 1


# ---------------------------------------------------------------------------
# send_result orchestration
# ---------------------------------------------------------------------------

class TestSendResult:
    """Test the public send_result method."""

    def test_producer_not_initialized_raises(self, producer, mock_result):
        # _update_topic runs before the None check, so wire up schema_name attrs
        producer.serializer.schema_name = "schema"
        producer.serializer.result_interface = Mock()
        producer.serializer.result_interface.schema_name = "schema"
        producer.producer = None
        with pytest.raises(KafkaProducerError):
            producer.send_result(mock_result)

    def test_unexpected_error_wrapped(self, producer, mock_result):
        """Non-KafkaProducerError from _send_and_flush should be wrapped."""
        producer.serializer.schema_name = "schema"
        producer.serializer.result_interface = Mock()
        producer.serializer.result_interface.schema_name = "schema"

        with patch.object(
            producer, "_send_and_flush", side_effect=TypeError("bad type")
        ):
            with pytest.raises(KafkaProducerError):
                producer.send_result(mock_result)

    def test_kafka_producer_error_propagated(self, producer, mock_result):
        """KafkaProducerError from _send_and_flush should propagate as-is."""
        producer.serializer.schema_name = "schema"
        producer.serializer.result_interface = Mock()
        producer.serializer.result_interface.schema_name = "schema"

        with patch.object(
            producer,
            "_send_and_flush",
            side_effect=KafkaProducerError("inner fail"),
        ):
            with pytest.raises(KafkaProducerError, match="inner fail"):
                producer.send_result(mock_result)


# ---------------------------------------------------------------------------
# _update_topic
# ---------------------------------------------------------------------------

class TestUpdateTopic:
    """Test topic/schema suffix updating."""

    def test_suffix_appended_to_topic_and_schema(self, producer, mock_result):
        producer.config.topic = "vqe_results"
        producer.serializer.schema_name = "vqe_results"
        producer.serializer.result_interface = Mock()
        producer.serializer.result_interface.schema_name = "vqe_results"

        producer._update_topic(mock_result)

        assert producer.config.topic == "vqe_results_mol_H2_sto3g"
        assert producer.serializer.schema_name == "vqe_results_mol_H2_sto3g"

    def test_existing_suffix_replaced(self, producer, mock_result):
        """If topic already has a _mol suffix it should be replaced."""
        producer.config.topic = "vqe_results_mol_old"
        producer.serializer.schema_name = "vqe_results_mol_old"
        producer.serializer.result_interface = Mock()
        producer.serializer.result_interface.schema_name = "vqe_results_mol_old"

        producer._update_topic(mock_result)

        assert producer.config.topic == "vqe_results_mol_H2_sto3g"
        assert producer.serializer.schema_name == "vqe_results_mol_H2_sto3g"


# ---------------------------------------------------------------------------
# Close / context manager
# ---------------------------------------------------------------------------

class TestCloseAndContextManager:
    """Test close() and __enter__/__exit__."""

    def test_close_when_no_producer(self, producer):
        """close() with producer=None should be a no-op."""
        producer.producer = None
        producer.close()  # should not raise

    def test_close_error_raises(self, producer):
        producer.producer.close.side_effect = Exception("socket error")
        with pytest.raises(KafkaProducerError):
            producer.close()

    def test_context_manager_calls_close(self, producer):
        with patch.object(producer, "close") as mock_close:
            with producer:
                pass
            mock_close.assert_called_once()

    def test_enter_returns_self(self, producer):
        assert producer.__enter__() is producer
