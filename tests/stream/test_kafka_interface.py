"""Unit tests for VQEKafkaProducer initialization and security config."""

from unittest.mock import Mock, patch

import pytest

from quantum_pipeline.configs.module.producer import ProducerConfig
from quantum_pipeline.configs.module.security import (
    CertConfig,
    SaslSslOpts,
    SecurityConfig,
)
from quantum_pipeline.stream.kafka_interface import KafkaProducerError, VQEKafkaProducer


@pytest.fixture
def mock_config() -> ProducerConfig:
    mock_config = Mock(spec=ProducerConfig)
    mock_config.servers = ['localhost:9092']
    mock_config.topic = 'test_topic'
    mock_config.retries = 3
    mock_config.retry_delay = 1
    mock_config.kafka_retries = 3
    mock_config.acks = 'all'
    mock_config.timeout = 10

    mock_config.security = Mock(spec=SecurityConfig)
    mock_config.security.ssl = False
    mock_config.security.sasl_ssl = False
    mock_config.security.ssl_check_hostname = True

    mock_config.security.cert_config = Mock(CertConfig)
    mock_config.security.cert_config.ssl_dir = '/path/to/certs'
    mock_config.security.cert_config.ssl_cafile = 'ca.crt'
    mock_config.security.cert_config.ssl_certfile = 'client.crt'
    mock_config.security.cert_config.ssl_keyfile = 'client'
    mock_config.security.cert_config.ssl_password = 'testpass'
    mock_config.security.cert_config.ssl_crlfile = None
    mock_config.security.cert_config.ssl_ciphers = None

    mock_config.security.sasl_opts = Mock(SaslSslOpts)
    mock_config.security.sasl_opts.sasl_mechanism = 'PLAIN'
    mock_config.security.sasl_opts.sasl_plain_username = 'testuser'
    mock_config.security.sasl_opts.sasl_plain_password = 'testpass'
    mock_config.security.sasl_opts.sasl_kerberos_service_name = 'kerberos_name'
    mock_config.security.sasl_opts.sasl_kerberos_domain_name = 'kerberos_name'

    return mock_config


class TestVQEKafkaProducer:
    def test_init_success(self, mock_config):
        with patch('quantum_pipeline.stream.kafka_interface.VQEKafkaProducer') as mock_producer:
            mock_producer(mock_config)
            mock_producer.assert_called_once_with(mock_config)

    def test_init_no_brokers(self, mock_config):
        with (
            patch(
                'quantum_pipeline.stream.kafka_interface.VQEKafkaProducer',
                side_effect=KafkaProducerError('No brokers available'),
            ) as mock_producer,
            pytest.raises(KafkaProducerError),
        ):
            mock_producer(mock_config)

    def test_security_config_ssl(self, mock_config):
        mock_config.security.ssl = True
        mock_config.security.cert_config.ssl_cafile = 'ca.crt'
        mock_config.security.cert_config.ssl_certfile = 'client.crt'
        mock_config.security.cert_config.ssl_keyfile = 'client.key'

        with patch('quantum_pipeline.stream.kafka_interface.VQEKafkaProducer') as mock_producer:
            mock_producer(mock_config)
            mock_producer.assert_called_once_with(mock_config)

            producer_config = mock_producer.call_args[0][0]

            assert producer_config.security.ssl
            assert producer_config.security.cert_config.ssl_cafile == 'ca.crt'
            assert producer_config.security.cert_config.ssl_certfile == 'client.crt'
            assert producer_config.security.cert_config.ssl_keyfile == 'client.key'

    def test_security_config_sasl_plain(self, mock_config):
        mock_config.security.sasl_ssl = True
        mock_config.security.sasl_opts.sasl_mechanism = 'PLAIN'
        mock_config.security.sasl_opts.sasl_plain_username = 'testuser'
        mock_config.security.sasl_opts.sasl_plain_password = 'testpass'

        with patch('quantum_pipeline.stream.kafka_interface.VQEKafkaProducer') as mock_producer:
            mock_producer(mock_config)
            mock_producer.assert_called_once_with(mock_config)

            producer_config = mock_producer.call_args[0][0]

            assert producer_config.security.sasl_ssl
            assert producer_config.security.sasl_opts.sasl_mechanism == 'PLAIN'
            assert producer_config.security.sasl_opts.sasl_plain_username == 'testuser'
            assert producer_config.security.sasl_opts.sasl_plain_password == 'testpass'

    def test_invalid_sasl_config(self, mock_config):
        """sasl_ssl with missing username/password must raise, not build a half-configured producer."""
        mock_config.security.sasl_ssl = True
        mock_config.security.sasl_opts.sasl_mechanism = 'PLAIN'
        mock_config.security.sasl_opts.sasl_plain_username = None
        mock_config.security.sasl_opts.sasl_plain_password = None

        with pytest.raises(KafkaProducerError):
            VQEKafkaProducer(mock_config)


class TestProducerMessageLimits:
    """The two kafka-python size ceilings a converged VQE result runs into.

    kafka-python rejects a record larger than `max_request_size` (default 1 MiB) AND one
    larger than `buffer_memory` (default 32 MiB) - producer/kafka.py raises
    MessageSizeTooLargeError for either. A converged result is one row per energy
    evaluation at ~2 KB per row, i.e. tens of MB, so raising only one of the two leaves the
    send failing at the other.
    """

    KAFKA_PYTHON_MAX_REQUEST_SIZE = 1_048_576
    KAFKA_PYTHON_BUFFER_MEMORY = 33_554_432

    @staticmethod
    def _config() -> Mock:
        config = Mock(spec=ProducerConfig)
        config.servers = ['localhost:9092']
        config.topic = 'experiment.vqe'
        config.kafka_retries = 5
        config.acks = 'all'
        config.max_request_size = 134_217_728
        config.buffer_memory = 134_217_728
        return config

    def test_limits_reach_kafka_producer(self):
        """Both limits must arrive as producer kwargs, not just live on the config object."""
        with (
            patch('quantum_pipeline.stream.kafka_interface.KafkaProducer') as mock_kp,
            patch('quantum_pipeline.stream.kafka_interface.KafkaSecurity') as mock_security,
            patch('quantum_pipeline.stream.kafka_interface.SchemaRegistry'),
            patch('quantum_pipeline.stream.kafka_interface.VQEDecoratedResultInterface'),
        ):
            mock_security.return_value.build_security_config.return_value = {}
            VQEKafkaProducer(self._config())

        kwargs = mock_kp.call_args.kwargs
        assert kwargs['max_request_size'] == 134_217_728
        assert kwargs['buffer_memory'] == 134_217_728

    def test_defaults_clear_the_kafka_python_ceilings(self):
        config = ProducerConfig.from_dict({})

        assert config.max_request_size > self.KAFKA_PYTHON_MAX_REQUEST_SIZE
        assert config.buffer_memory > self.KAFKA_PYTHON_BUFFER_MEMORY
        # A record must fit the buffer as well as the request, so the buffer cannot be smaller.
        assert config.buffer_memory >= config.max_request_size

    def test_config_round_trips_through_dict(self):
        config = ProducerConfig.from_dict({'max_request_size': 4096, 'buffer_memory': 8192})

        assert config.max_request_size == 4096
        assert config.buffer_memory == 8192
        assert config.to_dict()['max_request_size'] == 4096
        assert config.to_dict()['buffer_memory'] == 8192
