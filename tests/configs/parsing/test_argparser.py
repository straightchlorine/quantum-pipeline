import os
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
                '--sasl-ssl--sasl-mechanism',
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


def test_max_iterations_default(argparser):
    """Test that max_iterations uses default when not specified."""
    args = argparser.parser.parse_args(['--file', 'molecule.json'])
    assert args.max_iterations == 100  # Should use DEFAULTS['max_iterations']


def test_max_iterations_explicit(argparser):
    """Test that max_iterations uses explicit value when provided."""
    args = argparser.parser.parse_args(['--file', 'molecule.json', '--max-iterations', '3'])
    assert args.max_iterations == 3


class TestArgparserEdgeCases:
    """Test edge cases and input validation for argparser."""

    def test_max_iterations_zero(self, argparser):
        """Test max_iterations with zero value."""
        args = argparser.parser.parse_args(['--file', 'molecule.json', '--max-iterations', '0'])
        assert args.max_iterations == 0

    def test_max_iterations_negative(self, argparser):
        """Test max_iterations with negative value."""
        args = argparser.parser.parse_args(['--file', 'molecule.json', '--max-iterations', '-5'])
        assert args.max_iterations == -5

    def test_max_iterations_very_large(self, argparser):
        """Test max_iterations with very large value."""
        args = argparser.parser.parse_args(
            ['--file', 'molecule.json', '--max-iterations', '999999']
        )
        assert args.max_iterations == 999999

    def test_max_iterations_string_number(self, argparser):
        """Test max_iterations with string that converts to int."""
        args = argparser.parser.parse_args(['--file', 'molecule.json', '--max-iterations', '42'])
        assert args.max_iterations == 42
        assert isinstance(args.max_iterations, int)

    def test_max_iterations_invalid_string(self, argparser):
        """Test max_iterations with invalid string."""
        with pytest.raises(SystemExit):
            argparser.parser.parse_args(['--file', 'molecule.json', '--max-iterations', 'invalid'])

    def test_max_iterations_float_string(self, argparser):
        """Test max_iterations with float string."""
        with pytest.raises(SystemExit):
            argparser.parser.parse_args(['--file', 'molecule.json', '--max-iterations', '3.14'])

    def test_threshold_zero(self, argparser):
        """Test convergence threshold with zero value."""
        args = argparser.parser.parse_args(['--file', 'molecule.json', '--threshold', '0'])
        assert args.threshold == 0.0

    def test_threshold_negative(self, argparser):
        """Test convergence threshold with negative value."""
        # Note: argparse treats negative numbers as options, so we need to use = syntax
        args = argparser.parser.parse_args(['--file', 'molecule.json', '--threshold=-1e-6'])
        assert args.threshold == -1e-6

    def test_threshold_very_small(self, argparser):
        """Test convergence threshold with very small value."""
        args = argparser.parser.parse_args(['--file', 'molecule.json', '--threshold', '1e-20'])
        assert args.threshold == 1e-20

    def test_threshold_scientific_notation(self, argparser):
        """Test convergence threshold with scientific notation."""
        args = argparser.parser.parse_args(['--file', 'molecule.json', '--threshold', '1.5e-7'])
        assert args.threshold == 1.5e-7

    def test_threshold_invalid_string(self, argparser):
        """Test convergence threshold with invalid string."""
        with pytest.raises(SystemExit):
            argparser.parser.parse_args(['--file', 'molecule.json', '--threshold', 'invalid'])

    def test_convergence_flag_behavior(self, argparser):
        """Test convergence flag behavior with and without threshold."""
        # With convergence flag
        args_with_flag = argparser.parser.parse_args(['--file', 'molecule.json', '--convergence'])
        assert args_with_flag.convergence is True

        # Without convergence flag
        args_without_flag = argparser.parser.parse_args(['--file', 'molecule.json'])
        assert args_without_flag.convergence is False

    def test_multiple_flags_combination(self, argparser):
        """Test various combinations of max_iterations and convergence flags."""
        # Both specified
        args = argparser.parser.parse_args(
            [
                '--file',
                'molecule.json',
                '--max-iterations',
                '10',
                '--convergence',
                '--threshold',
                '1e-6',
            ]
        )
        assert args.max_iterations == 10
        assert args.convergence is True
        assert args.threshold == 1e-6

        # Only max_iterations
        args = argparser.parser.parse_args(['--file', 'molecule.json', '--max-iterations', '5'])
        assert args.max_iterations == 5
        assert args.convergence is False

        # Only convergence
        args = argparser.parser.parse_args(
            ['--file', 'molecule.json', '--convergence', '--threshold', '1e-8']
        )
        assert args.max_iterations == 100  # Default value
        assert args.convergence is True
        assert args.threshold == 1e-8

    def test_optimizer_valid_choices(self, argparser):
        """Test optimizer with all valid choices."""
        valid_optimizers = ['COBYLA', 'L-BFGS-B', 'COBYQA']  # Use actual supported optimizers

        for optimizer in valid_optimizers:
            args = argparser.parser.parse_args(
                ['--file', 'molecule.json', '--optimizer', optimizer]
            )
            assert args.optimizer == optimizer

    def test_optimizer_invalid_choice(self, argparser):
        """Test optimizer with invalid choice."""
        with pytest.raises(SystemExit):
            argparser.parser.parse_args(
                ['--file', 'molecule.json', '--optimizer', 'INVALID_OPTIMIZER']
            )

    def test_optimizer_case_sensitivity(self, argparser):
        """Test optimizer case sensitivity."""
        with pytest.raises(SystemExit):
            argparser.parser.parse_args(
                [
                    '--file',
                    'molecule.json',
                    '--optimizer',
                    'cobyla',  # lowercase should fail
                ]
            )

    def test_missing_required_file(self, argparser):
        """Test behavior when required file argument is missing."""
        with pytest.raises(SystemExit):
            argparser.parser.parse_args(['--max-iterations', '10'])

    def test_empty_file_string(self, argparser):
        """Test behavior with empty file string."""
        args = argparser.parser.parse_args(['--file', ''])
        assert args.file == ''

    def test_file_with_spaces(self, argparser):
        """Test file path with spaces."""
        args = argparser.parser.parse_args(['--file', 'path with spaces/molecule.json'])
        assert args.file == 'path with spaces/molecule.json'

    def test_special_characters_in_file(self, argparser):
        """Test file path with special characters."""
        special_path = 'mol@cule#data$.json'
        args = argparser.parser.parse_args(['--file', special_path])
        assert args.file == special_path

    def test_max_iterations_boundary_values(self, argparser):
        """Test max_iterations with boundary values."""
        # Test with 1 (minimum meaningful value)
        args = argparser.parser.parse_args(['--file', 'molecule.json', '--max-iterations', '1'])
        assert args.max_iterations == 1

        # Test with maximum int value
        import sys

        max_int = str(sys.maxsize)
        args = argparser.parser.parse_args(
            ['--file', 'molecule.json', '--max-iterations', max_int]
        )
        assert args.max_iterations == sys.maxsize

    def test_threshold_boundary_values(self, argparser):
        """Test convergence threshold with boundary values."""

        # Test with very small positive value
        args = argparser.parser.parse_args(['--file', 'molecule.json', '--threshold', '1e-308'])
        assert args.threshold == 1e-308

        # Test with 1.0 (upper boundary for typical convergence)
        args = argparser.parser.parse_args(['--file', 'molecule.json', '--threshold', '1.0'])
        assert args.threshold == 1.0

    def test_default_values_preservation(self, argparser):
        """Test that default values are properly preserved."""
        args = argparser.parser.parse_args(['--file', 'molecule.json'])

        # Check defaults from DEFAULTS dict
        assert args.max_iterations == 100  # DEFAULTS['max_iterations']
        assert args.optimizer == 'L-BFGS-B'  # DEFAULTS['optimizer']
        assert args.convergence is False  # DEFAULTS['convergence_threshold_enable']
        assert args.threshold == 1e-6  # DEFAULTS['convergence_threshold']


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


class TestPerformanceMonitoringArguments:
    """Test performance monitoring related arguments."""

    def test_enable_performance_monitoring_flag(self, argparser):
        """Test --enable-performance-monitoring flag."""
        args = argparser.parser.parse_args(
            ['--file', 'molecule.json', '--enable-performance-monitoring']
        )
        assert args.enable_performance_monitoring is True

    def test_performance_monitoring_disabled_by_default(self, argparser):
        """Test that performance monitoring is disabled by default."""
        args = argparser.parser.parse_args(['--file', 'molecule.json'])
        assert hasattr(args, 'enable_performance_monitoring')
        assert args.enable_performance_monitoring is False

    def test_performance_interval_argument(self, argparser):
        """Test --performance-interval argument."""
        args = argparser.parser.parse_args(
            ['--file', 'molecule.json', '--performance-interval', '30']
        )
        assert args.performance_interval == 30

    def test_performance_interval_default(self, argparser):
        """Test default value for performance interval."""
        args = argparser.parser.parse_args(['--file', 'molecule.json'])
        assert args.performance_interval == 30

    def test_performance_pushgateway_argument(self, argparser):
        """Test --performance-pushgateway argument."""
        args = argparser.parser.parse_args(
            ['--file', 'molecule.json', '--performance-pushgateway', 'http://monit:9091']
        )
        assert args.performance_pushgateway == 'http://monit:9091'

    def test_performance_export_format_json(self, argparser):
        """Test --performance-export-format with json."""
        args = argparser.parser.parse_args(
            ['--file', 'molecule.json', '--performance-export-format', 'json']
        )
        assert args.performance_export_format == 'json'

    def test_performance_export_format_prometheus(self, argparser):
        """Test --performance-export-format with prometheus."""
        args = argparser.parser.parse_args(
            ['--file', 'molecule.json', '--performance-export-format', 'prometheus']
        )
        assert args.performance_export_format == 'prometheus'

    def test_performance_export_format_both(self, argparser):
        """Test --performance-export-format with both."""
        args = argparser.parser.parse_args(
            ['--file', 'molecule.json', '--performance-export-format', 'both']
        )
        assert args.performance_export_format == 'both'

    def test_performance_export_format_default(self, argparser):
        """Test default value for performance export format."""
        args = argparser.parser.parse_args(['--file', 'molecule.json'])
        assert args.performance_export_format == 'both'

    def test_invalid_performance_export_format(self, argparser):
        """Test invalid performance export format."""
        with pytest.raises(SystemExit):
            argparser.parser.parse_args(
                ['--file', 'molecule.json', '--performance-export-format', 'invalid']
            )

    def test_complete_performance_monitoring_configuration(self, argparser):
        """Test complete performance monitoring configuration."""
        args = argparser.parser.parse_args(
            [
                '--file',
                'molecule.json',
                '--enable-performance-monitoring',
                '--performance-interval',
                '10',
                '--performance-pushgateway',
                'http://monit:9091',
                '--performance-export-format',
                'both',
            ]
        )
        assert args.enable_performance_monitoring is True
        assert args.performance_interval == 10
        assert args.performance_pushgateway == 'http://monit:9091'
        assert args.performance_export_format == 'both'

    def test_performance_interval_zero(self, argparser):
        """Test performance interval with zero value."""
        args = argparser.parser.parse_args(
            ['--file', 'molecule.json', '--performance-interval', '0']
        )
        assert args.performance_interval == 0

    def test_performance_interval_large_value(self, argparser):
        """Test performance interval with large value."""
        args = argparser.parser.parse_args(
            ['--file', 'molecule.json', '--performance-interval', '3600']
        )
        assert args.performance_interval == 3600

    def test_performance_interval_invalid_string(self, argparser):
        """Test performance interval with invalid string."""
        with pytest.raises(SystemExit):
            argparser.parser.parse_args(
                ['--file', 'molecule.json', '--performance-interval', 'invalid']
            )

    def test_performance_pushgateway_url_formats(self, argparser):
        """Test various URL formats for pushgateway."""
        test_urls = [
            'http://localhost:9091',
            'http://monit:9091',
            'http://192.168.1.100:9091',
            'http://monitoring-server.example.com:9091',
        ]

        for url in test_urls:
            args = argparser.parser.parse_args(
                ['--file', 'molecule.json', '--performance-pushgateway', url]
            )
            assert args.performance_pushgateway == url

    @pytest.mark.parametrize('export_format', ['json', 'prometheus', 'both'])
    def test_all_valid_export_formats(self, argparser, export_format):
        """Test all valid export format options."""
        args = argparser.parser.parse_args(
            ['--file', 'molecule.json', '--performance-export-format', export_format]
        )
        assert args.performance_export_format == export_format
