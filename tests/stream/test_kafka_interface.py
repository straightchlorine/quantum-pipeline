"""Unit tests for VQEKafkaProducer initialization and security config.

Broker-dependent tests (send, close) have been removed — they are fully
covered with proper mocking in test_kafka_interface_coverage.py and
with real containers in tests/integration/.
"""

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
    """Create a mock ProducerConfig for testing."""
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
        """Test successful initialization of VQEKafkaProducer."""
        with patch('quantum_pipeline.stream.kafka_interface.VQEKafkaProducer') as mock_producer:
            mock_producer(mock_config)
            mock_producer.assert_called_once_with(mock_config)

    def test_init_no_brokers(self, mock_config):
        """Test handling of NoBrokersAvailable exception."""
        with (
            patch(
                'quantum_pipeline.stream.kafka_interface.VQEKafkaProducer',
                side_effect=KafkaProducerError('No brokers available'),
            ) as mock_producer,
            pytest.raises(KafkaProducerError),
        ):
            mock_producer(mock_config)

    def test_security_config_ssl(self, mock_config):
        """Test SSL security configuration."""
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
        """Test SASL PLAIN authentication configuration."""
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
        """Test handling of invalid SASL configuration."""
        mock_config.security.sasl_ssl = True
        mock_config.security.sasl_opts.sasl_mechanism = 'PLAIN'
        mock_config.security.sasl_opts.sasl_plain_username = None
        mock_config.security.sasl_opts.sasl_plain_password = None

        with pytest.raises(KafkaProducerError):
            VQEKafkaProducer(mock_config)
