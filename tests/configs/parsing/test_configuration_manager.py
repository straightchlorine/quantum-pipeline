from argparse import Namespace
import json
from unittest.mock import patch

import pytest

from quantum_pipeline.configs.module.backend import BackendConfig
from quantum_pipeline.configs.module.producer import ProducerConfig
from quantum_pipeline.configs.parsing.configuration_manager import ConfigurationManager


@pytest.fixture
def config_manager():
    """Fixture to provide a ConfigurationManager."""
    return ConfigurationManager()


@pytest.fixture
def sample_args():
    """Fixture to provide sample command line arguments."""
    args = Namespace()

    # Kafka related args
    args.servers = 'localhost:9092'
    args.topic = 'test-topic'
    args.ssl = False
    args.sasl_ssl = False
    args.disable_ssl_check_hostname = False
    args.ssl_dir = '/path/to/ssl'
    args.ssl_cafile = None
    args.ssl_certfile = None
    args.ssl_keyfile = None
    args.ssl_password = None
    args.ssl_crlfile = None
    args.ssl_ciphers = None
    args.sasl_mechanism = None
    args.sasl_plain_username = None
    args.sasl_plain_password = None
    args.sasl_kerberos_service_name = None
    args.sasl_kerberos_domain_name = None
    args.retries = 3
    args.retry_delay = 2
    args.internal_retries = 5
    args.acks = 'all'
    args.timeout = 10

    # Backend related args
    args.ibm = False
    args.min_qubits = 5
    args.optimization_level = 1
    args.gpu = False
    args.simulation_method = 'statevector'

    # Other args
    args.file = 'test_file.json'
    args.basis = 'STO-3G'
    args.optimizer = 'COBYLA'
    args.dump = False
    args.load = None

    return args


def test_create_kafka_config(config_manager, sample_args):
    """Test creating Kafka configuration from arguments."""
    kafka_config = config_manager.create_kafka_config(sample_args)

    assert isinstance(kafka_config, ProducerConfig)
    assert kafka_config.servers == 'localhost:9092'
    assert kafka_config.topic == 'test-topic'
    assert kafka_config.security.ssl is False
    assert kafka_config.security.sasl_ssl is False
    assert kafka_config.security.ssl_check_hostname is False
    assert kafka_config.retries == 3
    assert kafka_config.retry_delay == 2
    assert kafka_config.kafka_retries == 5
    assert kafka_config.acks == 'all'
    assert kafka_config.timeout == 10


def test_create_backend_config(config_manager, sample_args):
    """Test creating Backend configuration from arguments."""
    backend_config = config_manager.create_backend_config(sample_args)

    assert isinstance(backend_config, BackendConfig)
    assert backend_config.local is False
    assert backend_config.min_num_qubits == 5
    assert backend_config.optimization_level == 1
    assert backend_config.gpu is False
    assert backend_config.simulation_method == 'statevector'


@patch('quantum_pipeline.configs.settings')
@patch('json.dump')
@patch('builtins.open')
def test_dump_configuration(mock_open, mock_json_dump, mock_settings, config_manager, sample_args):
    """Test dumping configuration to a file."""
    sample_args.dump = True
    mock_settings.RUN_CONFIGS = '/mock/path'

    # test config
    config = {
        'backend_config': config_manager.create_backend_config(sample_args),
        'kafka_config': config_manager.create_kafka_config(sample_args),
        'some_other_config': 'value',
    }

    # call the method
    config_manager.dump(sample_args, config)

    # check if the file was open for writing and if the
    # dump() method was called
    mock_open.assert_called_once()
    mock_json_dump.assert_called_once()


@patch('builtins.open')
def test_load_configuration(mock_open, config_manager):
    """Test loading configuration from a file."""
    # mock configuration
    mock_file_content = json.dumps(
        {
            'backend_config': {
                'local': True,
                'gpu': False,
                'optimization_level': 1,
                'min_num_qubits': 5,
                'simulation_method': 'statevector',
            },
            'kafka_config': {
                'servers': 'localhost:9092',
                'topic': 'test-topic',
                'security': {'ssl': False, 'sasl_ssl': False, 'ssl_check_hostname': False},
                'retries': 3,
                'retry_delay': 2,
                'kafka_retries': 5,
                'acks': 'all',
                'timeout': 10,
            },
            'test_cfg': 'works',
        }
    )

    mock_open.return_value.__enter__.return_value.read.return_value = mock_file_content

    # call the method on the mock config
    config = config_manager.load('dummy_path.json')

    # check if the file was open for reading
    mock_open.assert_called_once_with('dummy_path.json')

    assert isinstance(config['backend_config'], BackendConfig)
    assert isinstance(config['kafka_config'], ProducerConfig)
    assert config['test_cfg'] == 'works'


@patch.object(ConfigurationManager, 'load')
@patch.object(ConfigurationManager, 'dump')
def test_get_config_with_load(mock_dump, mock_load, config_manager, sample_args):
    """Test get_config method with load argument."""
    sample_args.load = 'config.json'
    sample_args.dump = False

    config_manager.get_config(sample_args)

    mock_load.assert_called_once_with('config.json')
    mock_dump.assert_not_called()


@patch.object(ConfigurationManager, 'load')
@patch.object(ConfigurationManager, 'dump')
def test_get_config_with_dump(mock_dump, mock_load, config_manager, sample_args):
    """Test get_config method with dump argument."""
    sample_args.load = None
    sample_args.dump = True

    config = config_manager.get_config(sample_args)

    mock_load.assert_not_called()
    mock_dump.assert_called_once()
    assert isinstance(config['kafka_config'], ProducerConfig)
    assert isinstance(config['backend_config'], BackendConfig)


def test_get_config_regular(config_manager, sample_args):
    """Test get_config method with normal arguments."""
    sample_args.load = None
    sample_args.dump = False

    config = config_manager.get_config(sample_args)

    assert isinstance(config['kafka_config'], ProducerConfig)
    assert isinstance(config['backend_config'], BackendConfig)
    assert config['servers'] == 'localhost:9092'
    assert config['topic'] == 'test-topic'
    assert config['basis'] == 'STO-3G'
    assert config['optimizer'] == 'COBYLA'
