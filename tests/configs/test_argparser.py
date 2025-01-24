from unittest.mock import mock_open, patch

import pytest

from quantum_pipeline.configs.parsing.argparser import QuantumPipelineArgParser
from quantum_pipeline.configs.parsing.configuration_manager import ConfigurationManager


@pytest.fixture
def argparser():
    return QuantumPipelineArgParser()


def test_required_arguments(argparser):
    """Test that the required argument --file is enforced."""
    with pytest.raises(SystemExit):
        argparser.parser.parse_args([])


def test_minimum_arguments(argparser):
    """Test that the minimum required arguments can be parsed."""
    args = argparser.parser.parse_args(['--file', 'molecule.json'])
    assert args.file == 'molecule.json'


def test_basis_argument(argparser):
    """Test the basis set argument."""
    args = argparser.parser.parse_args(['--file', 'molecule.json', '--basis', 'sto-3g'])
    assert args.basis == 'sto-3g'


def test_ansatz_reps_argument(argparser):
    """Test the ansatz reps argument."""
    args = argparser.parser.parse_args(['--file', 'molecule.json', '--ansatz-reps', '3'])
    assert args.ansatz_reps == '3'


def test_local_backend_flag(argparser):
    """Test the --local flag for using a local backend."""
    args = argparser.parser.parse_args(['--file', 'molecule.json', '--ibm'])
    assert args.ibm is False


def test_kafka_arguments(argparser):
    """Test Kafka-related arguments."""
    args = argparser.parser.parse_args(
        [
            '--file',
            'molecule.json',
            '--kafka',
            '--servers',
            'localhost',
            '--topic',
            'quantum',
            '--retries',
            '5',
            '--internal-retries',
            '3',
            '--acks',
            'all',
            '--timeout',
            '10',
        ]
    )
    assert args.kafka is True
    assert args.servers == 'localhost'
    assert args.topic == 'quantum'
    assert args.retries == '5'
    assert args.internal_retries == 3
    assert args.acks == 'all'
    assert args.timeout == 10


def test_vqe_parameters(argparser):
    """Test VQE-specific arguments."""
    args = argparser.parser.parse_args(
        [
            '--file',
            'molecule.json',
            '--max-iterations',
            '100',
            '--threshold',
            '0.01',
            '--optimizer',
            'COBYLA',
        ]
    )
    assert args.max_iterations == 100
    assert args.threshold == 0.01
    assert args.optimizer == 'COBYLA'


def test_output_and_logging(argparser):
    """Test output and logging arguments."""
    args = argparser.parser.parse_args(
        ['--file', 'molecule.json', '--output-dir', '/tmp/output', '--log-level', 'DEBUG']
    )
    assert args.output_dir == '/tmp/output'
    assert args.log_level == 'DEBUG'


def test_invalid_optimizer(argparser):
    """Test passing an invalid optimizer."""
    with pytest.raises(SystemExit):
        argparser.parser.parse_args(['--file', 'molecule.json', '--optimizer', 'INVALID'])


@pytest.mark.parametrize(
    'args,expected',
    [
        (['--file', 'molecule.json', '--shots', '1024'], '1024'),
        (['--file', 'molecule.json', '--shots', '-1'], SystemExit),  # Invalid case
    ],
)
def test_shots_argument(argparser, args, expected):
    """Test the --shots argument with valid and invalid inputs."""
    if expected is SystemExit:
        with pytest.raises(SystemExit):
            argparser.parser.parse_args(args)
    else:
        parsed_args = argparser.parser.parse_args(args)
        assert parsed_args.shots == expected


@patch('builtins.open', new_callable=mock_open)
def test_dump_configuration(mock_file, argparser):
    """Test that configuration is correctly dumped to a JSON file."""
    args = argparser.parser.parse_args(['--file', 'molecule.json', '--dump'])
    cfg_manager = ConfigurationManager()
    cfg_manager.get_config(args)
    path = cfg_manager.config_path
    mock_file.assert_called_once_with(path, 'w')


def test_load_configuration(argparser):
    args = argparser.parser.parse_args(['--file', 'molecule.json', '--dump'])
    cfg_manager = ConfigurationManager()
    dump_config = cfg_manager.get_config(args)

    args = argparser.parser.parse_args(
        [
            '--file',
            'molecule.json',
            '--load',
            cfg_manager.config_path.as_posix(),
        ]
    )
    load_config = cfg_manager.get_config(args)

    # ensure the dump and load configurations are the same
    dump_config['dump'], load_config['dump'] = None, None
    dump_config['load'], load_config['load'] = None, None

    assert dump_config == load_config
