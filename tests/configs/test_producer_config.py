from typing import Any, Dict

import pytest

from quantum_pipeline.configs.module.producer import ProducerConfig
from quantum_pipeline.configs.module.security import SecurityConfig


@pytest.fixture
def sample_producer_defaults() -> Dict[str, Any]:
    """Fixture to provide sample default producer configurations."""
    return {
        'kafka': {
            'servers': 'localhost:9092',
            'topic': 'vqe_decorated_result',
            'retries': 3,
            'retry_delay': 2,
            'internal_retries': 0,
            'acks': 'all',
            'timeout': 10,
            'security': {
                'ssl': False,
                'sasl_ssl': False,
                'ssl_check_hostname': False,
                'certs': {
                    'cafile': '/path/to/default/cafile',
                    'certfile': '/path/to/default/certfile',
                    'keyfile': '/path/to/default/keyfile',
                },
            },
        }
    }


@pytest.fixture
def sample_security_config():
    """Fixture to provide a sample SecurityConfig."""
    return SecurityConfig.get_default()


def test_producer_config_init(sample_security_config):
    """Test initialization of ProducerConfig with all parameters."""
    producer_config = ProducerConfig(
        servers='broker1:9092,broker2:9092',
        topic='test-topic',
        security=sample_security_config,
        retries=5,
        retry_delay=3,
        kafka_retries=7,
        acks='1',
        timeout=15,
    )

    assert producer_config.servers == 'broker1:9092,broker2:9092'
    assert producer_config.topic == 'test-topic'
    assert producer_config.security == sample_security_config
    assert producer_config.retries == 5
    assert producer_config.retry_delay == 3
    assert producer_config.kafka_retries == 7
    assert producer_config.acks == '1'
    assert producer_config.timeout == 15


def test_producer_config_to_dict(sample_security_config):
    """Test conversion of ProducerConfig to dictionary."""
    producer_config = ProducerConfig(
        servers='broker1:9092,broker2:9092',
        topic='test-topic',
        security=sample_security_config,
        retries=5,
        retry_delay=3,
        kafka_retries=7,
        acks='1',
        timeout=15,
    )

    config_dict = producer_config.to_dict()

    assert config_dict['servers'] == 'broker1:9092,broker2:9092'
    assert config_dict['topic'] == 'test-topic'
    assert isinstance(config_dict['security'], dict)
    assert config_dict['retries'] == 5
    assert config_dict['retry_delay'] == 3
    assert config_dict['kafka_retries'] == 7
    assert config_dict['acks'] == '1'
    assert config_dict['timeout'] == 15


def test_producer_config_from_dict():
    """Test creating ProducerConfig from a dictionary."""
    config_dict = {
        'servers': 'broker1:9092,broker2:9092',
        'topic': 'test-topic',
        'security': {
            'ssl': True,
            'sasl_ssl': False,
            'ssl_check_hostname': True,
            'cert_config': {
                'ssl_dir': '/path/to/ssl',
                'cafile': '/path/to/cafile',
                'certfile': '/path/to/certfile',
                'keyfile': '/path/to/keyfile',
            },
        },
        'retries': 5,
        'retry_delay': 3,
        'kafka_retries': 7,
        'acks': '1',
        'timeout': 15,
    }

    producer_config = ProducerConfig.from_dict(config_dict)

    assert producer_config.servers == 'broker1:9092,broker2:9092'
    assert producer_config.topic == 'test-topic'
    assert producer_config.security.ssl is True
    assert producer_config.security.ssl_check_hostname is True
    assert producer_config.retries == 5
    assert producer_config.retry_delay == 3
    assert producer_config.kafka_retries == 7
    assert producer_config.acks == '1'
    assert producer_config.timeout == 15


def test_producer_config_from_empty_dict(monkeypatch, sample_producer_defaults):
    """Test creating ProducerConfig from an empty dictionary."""
    monkeypatch.setattr(
        'quantum_pipeline.configs.defaults.DEFAULTS',
        sample_producer_defaults,
    )

    producer_config = ProducerConfig.from_dict({})

    assert producer_config.servers == sample_producer_defaults['kafka']['servers']
    assert producer_config.topic == sample_producer_defaults['kafka']['topic']
    assert producer_config.retries == sample_producer_defaults['kafka']['retries']
    assert producer_config.retry_delay == sample_producer_defaults['kafka']['retry_delay']
    assert producer_config.kafka_retries == sample_producer_defaults['kafka']['internal_retries']
    assert producer_config.acks == sample_producer_defaults['kafka']['acks']
    assert producer_config.timeout == sample_producer_defaults['kafka']['timeout']
    assert isinstance(producer_config.security, SecurityConfig)


def test_producer_config_with_partial_dict(sample_producer_defaults):
    """Test creating ProducerConfig with a partial dictionary."""
    config_dict = {
        'servers': 'broker1:9092,broker2:9092',
        'topic': 'test-topic',
    }

    producer_config = ProducerConfig.from_dict(config_dict)

    assert producer_config.servers == 'broker1:9092,broker2:9092'
    assert producer_config.topic == 'test-topic'
    assert producer_config.retries == sample_producer_defaults['kafka']['retries']
    assert producer_config.retry_delay == sample_producer_defaults['kafka']['retry_delay']
    assert producer_config.kafka_retries == sample_producer_defaults['kafka']['internal_retries']
    assert producer_config.acks == sample_producer_defaults['kafka']['acks']
    assert producer_config.timeout == sample_producer_defaults['kafka']['timeout']
    assert isinstance(producer_config.security, SecurityConfig)
