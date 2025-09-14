from unittest.mock import mock_open, patch
import os
import pytest

from quantum_pipeline.configs.parsing.argparser import QuantumPipelineArgParser
from quantum_pipeline.configs.parsing.configuration_manager import ConfigurationManager


@pytest.fixture
def argparser():
    return QuantumPipelineArgParser()


def test_required_arguments(argparser):
    """Test that the required argument --file is enforced."""
    with pytest.raises(SystemExit):
        args = argparser.parser.parse_args([])
        argparser._validate_args(args)


def test_ssl_basic_configuration_with_mock_dir(argparser, monkeypatch):
    """
    Test SSL configuration by mocking directory existence.

    Uses monkeypatch to mock os.path.isdir to always return True.
    """
    # ensures isdir always returns True
    monkeypatch.setattr(os.path, 'isdir', lambda path: True)

    args = argparser.parser.parse_args(['--file', 'molecule.json', '--kafka', '--ssl'])
    argparser._validate_args(args)

    assert args.ssl is True
    assert args.ssl_dir == './secrets/'


def test_ssl_with_password(argparser):
    """Test SSL configuration with password."""
    args = argparser.parser.parse_args(
        ['--file', 'molecule.json', '--ssl', '--ssl-password', 'secret']
    )
    assert args.ssl_password == 'secret'


def test_ssl_dir_conflict(argparser):
    """Test conflict between ssl_dir and individual SSL files."""
    with pytest.raises(SystemExit):
        args = argparser.parser.parse_args(
            ['--file', 'molecule.json', '--ssl', '--ssl-dir', './certs/', '--ssl-cafile', 'ca.pem']
        )
        argparser._validate_args(args)


def test_ssl_missing_cafile(argparser):
    """Test missing CA file when not using ssl_dir."""
    with pytest.raises(SystemExit):
        args = argparser.parser.parse_args(
            [
                '--file',
                'molecule.json',
                '--ssl',
                '--ssl-dir',
                '',
                '--ssl-certfile',
                'cert.pem',
            ]
        )
        argparser._validate_args(args)


def test_sasl_plain_valid(argparser):
    """Test valid SASL PLAIN configuration."""
    args = argparser.parser.parse_args(
        [
            '--file',
            'molecule.json',
            '--kafka',
            '--sasl-mechanism',
            'PLAIN',
            '--sasl-plain-username',
            'user',
            '--sasl-plain-password',
            'pass',
        ]
    )
    argparser._validate_args(args)
    assert args.sasl_mechanism == 'PLAIN'
    assert args.sasl_plain_username == 'user'
    assert args.sasl_plain_password == 'pass'


def test_sasl_missing_credentials(argparser):
    """Test missing SASL PLAIN credentials."""
    with pytest.raises(SystemExit):
        args = argparser.parser.parse_args(
            ['--file', 'molecule.json', '--kafka', '--sasl-ssl', '--sasl-mechanism', 'PLAIN']
        )
        argparser._validate_args(args)


def test_sasl_conflicting_options(argparser):
    """Test conflicting SASL mechanism options."""
    with pytest.raises(SystemExit):
        args = argparser.parser.parse_args(
            [
                '--file',
                'molecule.json',
                '--kafka',
                '--sasl-ssl' '--sasl-mechanism',
                'PLAIN',
                '--sasl-kerberos-domain-name',
                'domain',
            ]
        )
        argparser._validate_args(args)


def test_ssl_without_flag(argparser):
    """Test SSL options without --ssl flag."""
    with pytest.raises(SystemExit):
        args = argparser.parser.parse_args(
            ['--file', 'molecule.json', '--kafka', '--ssl-cafile', 'ca.pem']
        )
        argparser._validate_args(args)


def test_minimum_arguments(argparser):
    """Test that the minimum required arguments can be parsed."""
    args = argparser.parser.parse_args(['--file', 'molecule.json'])
    assert args.file == 'molecule.json'


def test_basis_argument(argparser):
    """Test the basis set argument."""
    args = argparser.parser.parse_args(['--file', 'molecule.json', '--basis', 'sto3g'])
    assert args.basis == 'sto3g'


def test_gpu_argument(argparser):
    """Test the gpu enable argument."""
    args = argparser.parser.parse_args(['--file', 'molecule.json', '--gpu'])
    assert args.gpu


def test_ssl_argument(argparser):
    """Test the gpu enable argument."""
    args = argparser.parser.parse_args(
        ['--file', 'molecule.json', '--ssl', '--ssl-password', 'password']
    )
    assert args.ssl
    assert args.ssl_password == 'password'


def test_noise_argument(argparser):
    """Test the noise enable argument."""
    args = argparser.parser.parse_args(['--file', 'molecule.json', '--noise', 'ibm_brisbane'])
    assert args.noise == 'ibm_brisbane'


def test_noise_disable_argument(argparser):
    """Test the noise enable argument."""
    args = argparser.parser.parse_args(['--file', 'molecule.json'])
    assert args.noise is None


def test_ansatz_reps_argument(argparser):
    """Test the ansatz reps argument."""
    args = argparser.parser.parse_args(['--file', 'molecule.json', '--ansatz-reps', '3'])
    assert args.ansatz_reps == '3'


def test_local_backend_flag(argparser):
    """Test the --local flag for using a local backend."""
    args = argparser.parser.parse_args(['--file', 'molecule.json', '--ibm'])
    assert not args.ibm


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


def test_invalid_basis_set(argparser):
    """Test passing an invalid basis set."""
    with pytest.raises(SystemExit):
        args = argparser.parser.parse_args(['--file', 'molecule.json', '--basis', 'INVALID_BASIS'])
        argparser._validate_args(args)


def test_simulation_method(argparser):
    """Test simulation method selection."""
    args = argparser.parser.parse_args(
        ['--file', 'molecule.json', '--simulation-method', 'statevector']
    )
    assert args.simulation_method == 'statevector'


@pytest.mark.parametrize('invalid_method', ['', 'unknown_method'])
def test_invalid_simulation_method(argparser, invalid_method):
    """Test invalid simulation method."""
    with pytest.raises(SystemExit):
        argparser.parser.parse_args(
            ['--file', 'molecule.json', '--simulation-method', invalid_method]
        )


def test_sasl_ssl_without_mechanism(argparser):
    """Test SASL SSL without specifying mechanism."""
    with pytest.raises(SystemExit):
        args = argparser.parser.parse_args(
            [
                '--file',
                'molecule.json',
                '--kafka',
                '--sasl-ssl',
                '--sasl-plain-username',
                'user',
                '--sasl-plain-password',
                'pass',
            ]
        )
        argparser._validate_args(args)


def test_kerberos_options(argparser):
    """Test Kerberos-specific SASL options."""
    args = argparser.parser.parse_args(
        [
            '--file',
            'molecule.json',
            '--kafka',
            '--sasl-ssl',
            '--sasl-mechanism',
            'GSSAPI',
            '--sasl-kerberos-service-name',
            'custom_service',
            '--sasl-kerberos-domain-name',
            'example.com',
        ]
    )
    assert args.sasl_kerberos_service_name == 'custom_service'
    assert args.sasl_kerberos_domain_name == 'example.com'


@pytest.mark.parametrize('log_level', ['DEBUG', 'INFO', 'WARNING', 'ERROR'])
def test_valid_log_levels(argparser, log_level):
    """Test all valid log levels."""
    args = argparser.parser.parse_args(['--file', 'molecule.json', '--log-level', log_level])
    assert args.log_level == log_level


def test_conflicting_config_options(argparser):
    """Test conflicting configuration options."""
    with pytest.raises(SystemExit):
        argparser.parser.parse_args(['--dump', '--load', 'config.json', '--file', 'molecule.json'])
