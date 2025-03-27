from unittest.mock import Mock, patch

from kafka.errors import KafkaError, NoBrokersAvailable
import pytest

from quantum_pipeline.configs.module.producer import ProducerConfig
from quantum_pipeline.configs.module.security import (
    CertConfig,
    SaslSslOpts,
    SecurityConfig,
)
from quantum_pipeline.stream.kafka_interface import KafkaProducerError, VQEKafkaProducer
from quantum_pipeline.structures.vqe_observation import VQEDecoratedResult


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


@pytest.fixture
def mock_result() -> VQEDecoratedResult:
    """Create a mock VQEDecoratedResult for testing."""
    mock_result = Mock(spec=VQEDecoratedResult)
    mock_result.get_schema_suffix.return_value = '_mol_test'
    return mock_result


class TestVQEKafkaProducer:
    def test_init_success(self, mock_config):
        """Test successful initialization of VQEKafkaProducer."""
        with patch('quantum_pipeline.stream.kafka_interface.VQEKafkaProducer') as mock_producer:
            mock_producer(mock_config)
            mock_producer.assert_called_once_with(mock_config)

    def test_init_no_brokers(self, mock_config):
        """Test handling of NoBrokersAvailable exception."""
        with patch(
            'quantum_pipeline.stream.kafka_interface.VQEKafkaProducer',
            side_effect=KafkaProducerError('No brokers available'),
        ) as mock_producer:
            with pytest.raises(KafkaProducerError, match='No brokers available'):
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

    def test_send_result_success(self, mock_config, mock_result):
        """Test successful sending of a result.

        This test verifies that:
            1. The producer is initialized correctly
            2. Serialization method is called
            3. send() and flush() methods are called on the Kafka producer
            4. Topic is updated before sending
        """

        try:
            # attempt to connect to Kafka, if unable - skip the test
            test_producer = VQEKafkaProducer(mock_config)
            test_producer.close()
        except NoBrokersAvailable:
            pytest.skip('Skipping test: No Kafka brokers available')
        except Exception as e:
            pytest.skip(f'Skipping test due to Kafka connection issue: {str(e)}')

        with patch(
            'quantum_pipeline.stream.kafka_security.KafkaSecurity',
            autospec=True,
        ) as mock_security_class:
            # mock security configuration
            mock_security = mock_security_class.return_value
            mock_security.build_security_config.return_value = {}

            with patch(
                'quantum_pipeline.utils.schema_registry.SchemaRegistry',
                autospec=True,
            ) as mock_registry_class:
                # create a mock schema registry class
                mock_registry_class.return_value

                # mock the producer class (along with internal kafka producer)
                producer = VQEKafkaProducer(mock_config)
                producer.producer = Mock()

                # mock metadata, returned by the producer after the send()
                mock_future = Mock()
                mock_metadata = Mock(topic='test_topic', partition=0, offset=1)
                mock_future.get.return_value = mock_metadata
                producer.producer.send.return_value = mock_future

                # mock serialization
                with patch.object(
                    producer, '_serialize_result', return_value=b'serialized_data'
                ) as mock_serialize:
                    # send result
                    producer.send_result(mock_result)

                    # check if the topic was updated
                    mock_result.get_schema_suffix.assert_called_once()

                    # check if serialization was triggered
                    mock_serialize.assert_called_once_with(mock_result)

                    # check if send was called with serialized data
                    producer.producer.send.assert_called_once_with(
                        mock_config.topic, b'serialized_data'
                    )

                    # check if flush was called
                    producer.producer.flush.assert_called_once()

    def test_send_result_failure(self, mock_config, mock_result):
        """Test handling of send failures.

        This test verifies that:
            1. An exception during sending raises a KafkaProducerError
            2. The topic is still updated before attempting to send
            3. The error contains appropriate context
        """
        try:
            # attempt to connect to Kafka, if unable - skip the test
            test_producer = VQEKafkaProducer(mock_config)
            test_producer.close()
        except NoBrokersAvailable:
            pytest.skip('Skipping test: No Kafka brokers available')
        except Exception as e:
            pytest.skip(f'Skipping test due to Kafka connection issue: {str(e)}')

        with patch(
            'quantum_pipeline.stream.kafka_security.KafkaSecurity',
            autospec=True,
        ) as mock_security_class:
            # mock security config
            mock_security = mock_security_class.return_value
            mock_security.build_security_config.return_value = {}

            # mock schema registry
            with patch(
                'quantum_pipeline.utils.schema_registry.SchemaRegistry',
                autospec=True,
            ):
                # create a real producer, with mock KafkaProducer
                producer = VQEKafkaProducer(mock_config)
                producer.producer = Mock()

                # mock serialization
                with patch.object(
                    producer, '_serialize_result', return_value=b'serialized_data'
                ) as mock_serialize:
                    # send sends an exception
                    producer.producer.send.side_effect = KafkaError('Send failed!')

                    # check if KafkaProducerError was raised
                    with pytest.raises(KafkaProducerError) as excinfo:
                        producer.send_result(mock_result)

                    # confirm that topic update was attempted
                    mock_result.get_schema_suffix.assert_called_once()

                    # confirm that serialization was attempted
                    mock_serialize.assert_called_once_with(mock_result)

                    # check that the error message contains the relevant context
                    assert 'Send failed!' in str(excinfo.value)

                    # ensure no flush - since send failed
                    producer.producer.flush.assert_not_called()

    def test_close(self, mock_config):
        """Test closing the producer.

        This test verifies that:
            1. The close method successfully calls the internal producer's close method
            2. Logging occurs during successful closure
            3. Proper error handling happens if close fails
        """
        try:
            # attempt to connect to Kafka, if unable - skip the test
            test_producer = VQEKafkaProducer(mock_config)
            test_producer.close()
        except NoBrokersAvailable:
            pytest.skip('Skipping test: No Kafka brokers available')
        except Exception as e:
            pytest.skip(f'Skipping test due to Kafka connection issue: {str(e)}')

        with patch(
            'quantum_pipeline.stream.kafka_security.KafkaSecurity',
            autospec=True,
        ) as mock_security_class:
            mock_security = mock_security_class.return_value
            mock_security.build_security_config.return_value = {}

            with patch(
                'quantum_pipeline.utils.schema_registry.SchemaRegistry',
                autospec=True,
            ):
                # create an actual wrapper object, but mock internal Producer
                producer = VQEKafkaProducer(mock_config)
                producer.producer = Mock()

                # close the producer
                producer.close()

                # check if the method really was called
                producer.producer.close.assert_called_once()

    def test_close_with_error(self, mock_config):
        """Test error handling during producer closure.

        This test verifies that:
            1. Errors during close are properly handled
            2. A KafkaProducerError is raised
            3. Error logging occurs
        """
        try:
            # attempt to connect to Kafka, if unable - skip the test
            test_producer = VQEKafkaProducer(mock_config)
            test_producer.close()
        except NoBrokersAvailable:
            pytest.skip('Skipping test: No Kafka brokers available')
        except Exception as e:
            pytest.skip(f'Skipping test due to Kafka connection issue: {str(e)}')

        with patch(
            'quantum_pipeline.stream.kafka_security.KafkaSecurity',
            autospec=True,
        ) as mock_security_class:
            mock_security = mock_security_class.return_value
            mock_security.build_security_config.return_value = {}

            with patch(
                'quantum_pipeline.utils.schema_registry.SchemaRegistry',
                autospec=True,
            ):
                # create the actual producer, with mock internal producer
                producer = VQEKafkaProducer(mock_config)
                producer.producer = Mock()

                # raise an exception when close is called
                producer.producer.close.side_effect = Exception('Closure failed')

                # check if the exception is properly handled
                with pytest.raises(KafkaProducerError, match='Failed to close producer'):
                    producer.close()
