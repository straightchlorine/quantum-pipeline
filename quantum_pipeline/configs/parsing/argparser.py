import argparse
import logging
import os
from typing import Any

from quantum_pipeline.configs import settings
from quantum_pipeline.configs.defaults import DEFAULTS
from quantum_pipeline.configs.parsing.configuration_manager import ConfigurationManager
from quantum_pipeline.utils.dir import ensureDirExists
from quantum_pipeline.utils.logger import get_logger


class QuantumPipelineArgParser:
    """Class for handling command line argument parsing for quantum pipeline."""

    def __init__(self):
        """Initialize the parser with default configuration."""
        self.logger = get_logger('Argparser')

        self.initialize_simulation_environment()
        self.parser = argparse.ArgumentParser(
            description='Quantum Circuit Simulation and Execution',
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        self._add_arguments()

    def initialize_simulation_environment(self):
        """Ensure required directories and configurations are in place."""
        ensureDirExists(settings.GEN_DIR)
        ensureDirExists(settings.GRAPH_DIR)
        ensureDirExists(settings.REPORT_DIR)
        ensureDirExists(settings.RUN_CONFIGS)

    def _create_optimizer_help_text(self) -> str:
        """Create detailed help text for optimizer options."""
        help_text = 'Available optimizers:\n'
        for optimizer, description in settings.SUPPORTED_OPTIMIZERS.items():
            help_text += f'|  {optimizer}: {description} |'
        return help_text

    def _create_simulation_method_help_text(self) -> str:
        """Create detailed help text for simulation methods."""
        help_text = 'Available simulation methods:\n'
        for method, description in settings.SIMULATION_METHODS.items():
            help_text += f'|  {method}: {description} |\n'
        return help_text.strip()

    def _add_arguments(self):
        """Add all argument groups and their arguments to the parser."""
        self._add_required_arguments()
        self._add_simulation_config()
        self._add_vqe_parameters()
        self._add_output_logging()
        self._add_backend_options()
        self._add_kafka_config()
        self._add_additional_features()
        self._add_security_options()

    def _add_required_arguments(self):
        """Add required arguments to the parser."""
        self.parser.add_argument(
            '-f', '--file', required=True, help='Path to molecule data file (JSON)'
        )

    def _add_simulation_config(self):
        """Add simulation configuration arguments."""
        sim_group = self.parser.add_argument_group('Simulation Configuration')
        sim_group.add_argument(
            '-b',
            '--basis',
            default=DEFAULTS['basis_set'],
            help='Basis set for the simulation',
            choices=settings.SUPPORTED_BASIS_SETS,
        )
        sim_group.add_argument(
            '-ar',
            '--ansatz-reps',
            default=DEFAULTS['ansatz_reps'],
            help='Amount of reps for the ansatz',
        )
        sim_group.add_argument(
            '--ibm',
            action='store_false',
            default=DEFAULTS['backend']['local'],
            help='Using IBM Quantum backend for simulation (otherwise local Aer simulator is used.)',
        )
        sim_group.add_argument(
            '--min-qubits',
            type=int,
            default=DEFAULTS['backend']['min_qubits'],
            help='Minimum number of qubits required for the backend',
        )

    def _add_kafka_config(self):
        """Add simulation configuration arguments."""
        kafka_group = self.parser.add_argument_group('Kafka Configuration')
        kafka_group.add_argument(
            '--kafka', action='store_true', help='Enable streaming results to Apache Kafka'
        )
        kafka_group.add_argument(
            '--servers',
            type=str,
            default=DEFAULTS['kafka']['servers'],
            help='Apache Kafka instance address',
        )
        kafka_group.add_argument(
            '--topic',
            type=str,
            default=DEFAULTS['kafka']['topic'],
            help='Name of the topic, with which message will be categorised with',
        )
        kafka_group.add_argument(
            '--retries',
            type=str,
            default=DEFAULTS['kafka']['retries'],
            help='Number of attempts to send message to kafka',
        )
        kafka_group.add_argument(
            '--retry-delay',
            type=str,
            default=DEFAULTS['kafka']['retry_delay'],
            help='Number of attempts to send message to kafka',
        )
        kafka_group.add_argument(
            '--internal-retries',
            type=int,
            default=DEFAULTS['kafka']['internal_retries'],
            help='Number of attempts kafka should attempt automatically (introduces risk of duplicate entries)',
        )
        kafka_group.add_argument(
            '--acks',
            default=DEFAULTS['kafka']['acks'],
            choices=['0', '1', 'all'],
            help='Number of acknowledgments producer requires to receive before a request is complete.',
        )
        kafka_group.add_argument(
            '--timeout',
            type=int,
            default=DEFAULTS['kafka']['timeout'],
            help='Number of seconds required for producer to consider request as failed',
        )

    def _add_vqe_parameters(self):
        """Add VQE-specific parameters."""
        vqe_group = self.parser.add_argument_group('VQE Parameters')
        vqe_group.add_argument(
            '--max-iterations',
            type=int,
            default=DEFAULTS['max_iterations'],
            help='Maximum number of VQE iterations',
        )

        vqe_group.add_argument(
            '--convergence',
            action='store_true',
            default=DEFAULTS['convergence_threshold_enable'],
            help='Enable convergence threshold during minimization',
        )
        vqe_group.add_argument(
            '--threshold',
            type=float,
            default=DEFAULTS['convergence_threshold'],
            help='Set convergence threshold for VQE optimization',
        )
        vqe_group.add_argument(
            '--optimizer',
            choices=list(settings.SUPPORTED_OPTIMIZERS.keys()),
            default=DEFAULTS['optimizer'],
            help=self._create_optimizer_help_text(),
            metavar='OPTIMIZER',
        )

    def _add_output_logging(self):
        """Add output and logging related arguments."""
        output_group = self.parser.add_argument_group('Output and Logging')
        output_group.add_argument(
            '--output-dir', default=settings.GEN_DIR, help='Directory to store output files'
        )
        output_group.add_argument(
            '--log-level',
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
            default='INFO',
            help='Set the logging level',
        )

    def shots(self, shots):
        if int(shots) <= 0:
            raise argparse.ArgumentTypeError(f'shots:{shots} is not a valid number')
        return shots

    def _add_backend_options(self):
        """Add advanced backend configuration options."""
        backend_group = self.parser.add_argument_group('Advanced Backend Options')
        backend_group.add_argument(
            '--shots',
            type=self.shots,
            default=DEFAULTS['shots'],
            help='Number of shots for quantum circuit execution',
        )
        backend_group.add_argument(
            '--optimization-level',
            type=int,
            choices=[0, 1, 2, 3],
            default=DEFAULTS['backend']['optimization_level'],
            help='Circuit optimization level',
        )

    def ensure_dir(self, path):
        if os.path.isdir(path):
            return True
        else:
            raise argparse.ArgumentTypeError(f'readable_dir:{path} is not a valid path')

    def ensure_file(self, path):
        if os.path.isfile(path):
            return path
        else:
            raise argparse.ArgumentTypeError(f'readable_dir:{path} is not a valid path')

    def _add_additional_features(self):
        """Add additional feature arguments."""
        additional_group = self.parser.add_argument_group('Additional Features')
        additional_group.add_argument(
            '--report', action='store_true', help='Generate a PDF report after simulation'
        )
        additional_group.add_argument(
            '--dump',
            action='store_true',
            help='Dump configuration generated by the parameters into JSON file',
        )
        additional_group.add_argument(
            '--load',
            type=self.ensure_file,
            help='Path to a JSON configuration file to load parameters from',
        )
        additional_group.add_argument(
            '--gpu',
            action='store_true',
            help='Enable GPU acceleration.',
        )
        additional_group.add_argument(
            '--simulation-method',
            choices=list(settings.SIMULATION_METHODS.keys()),
            default=DEFAULTS['backend']['method'],
            help=self._create_simulation_method_help_text(),
            metavar='SIMULATION_METHOD',
        )
        additional_group.add_argument(
            '--noise',
            type=str,
            default=DEFAULTS['backend']['noise_backend'],
            help='Choose, which noise model to base the simulation on.',
        )

        # Performance Monitoring Options
        perf_group = self.parser.add_argument_group('Performance Monitoring')
        perf_group.add_argument(
            '--enable-performance-monitoring',
            action='store_true',
            help='Enable detailed performance and resource monitoring for thesis analysis',
        )
        perf_group.add_argument(
            '--performance-interval',
            type=int,
            default=30,
            help='Performance metrics collection interval in seconds (default: 30)',
        )
        perf_group.add_argument(
            '--performance-pushgateway',
            type=str,
            default='http://localhost:9091',
            help='Prometheus PushGateway URL for metrics export',
        )
        perf_group.add_argument(
            '--performance-export-format',
            choices=['json', 'prometheus', 'both'],
            default='both',
            help='Performance metrics export format',
        )

    def _add_security_options(self):
        security_group = self.parser.add_argument_group(
            'Security settings for Apache Kafka connection'
        )
        security_group.add_argument(
            '--ssl',
            action='store_true',
            help='Enable SSL for Kafka producer.',
        )
        security_group.add_argument(
            '--disable-ssl-check-hostname',
            action='store_false',
            help='Disable SSL check hostname. ONLY FOR TESTING.',
        )
        security_group.add_argument(
            '--sasl-ssl',
            action='store_true',
            help='Enable SASL_SSL connection',
        )
        security_group.add_argument(
            '--ssl-password',
            default=None,
            help='Password used for SSL connection.',
        )
        security_group.add_argument(
            '--ssl-dir',
            default=DEFAULTS['kafka']['security']['certs']['dir'],
            help='Set the directory with SSL keys.',
        )
        security_group.add_argument(
            '--ssl-cafile',
            default=None,
            help='Path to CA certificate file (excluded by --ssl-dir).',
        )
        security_group.add_argument(
            '--ssl-certfile',
            default=None,
            help='Path to SSL certificate file (excluded by --ssl-dir).',
        )
        security_group.add_argument(
            '--ssl-keyfile',
            default=None,
            help='Path to SSL key file (excluded by --ssl-dir).',
        )
        security_group.add_argument(
            '--ssl-crlfile',
            default=None,
            help='Path to SSL certificate revocation list file (excluded by --ssl-dir).',
        )
        security_group.add_argument(
            '--ssl-ciphers',
            default=None,
            help='SSL cipher suite to use.',
        )
        security_group.add_argument(
            '--sasl-mechanism',
            choices=['PLAIN', 'GSSAPI', 'SCRAM-SHA-256', 'SCRAM-SHA-512'],
            help='Authentication mechanism for SASL.',
        )
        security_group.add_argument(
            '--sasl-plain-username',
            default=None,
            help='Username for SASL PLAIN and SCRAM authentication.',
        )
        security_group.add_argument(
            '--sasl-plain-password',
            default=None,
            help='Password for SASL PLAIN and SCRAM authentication.',
        )
        security_group.add_argument(
            '--sasl-kerberos-service-name',
            default='kafka',
            help='Kerberos service name for GSSAPI SASL mechanism.',
        )
        security_group.add_argument(
            '--sasl-kerberos-domain-name',
            default=None,
            help='Kerberos domain name for GSSAPI SASL mechanism.',
        )

    def kafka_params_set(self, args: argparse.Namespace):
        if (
            args.servers != DEFAULTS['kafka']['servers']
            or args.topic != DEFAULTS['kafka']['topic']
            or args.retries != DEFAULTS['kafka']['retries']
            or args.retry_delay != DEFAULTS['kafka']['retry_delay']
            or args.internal_retries != DEFAULTS['kafka']['internal_retries']
            or args.acks != DEFAULTS['kafka']['acks']
            or args.timeout != DEFAULTS['kafka']['timeout']
        ):
            return True
        return False

    def _validate_args(self, args: argparse.Namespace) -> None:  # noqa: C901
        """Validate parsed arguments."""
        settings.LOG_LEVEL = getattr(logging, args.log_level)

        if args.dump and args.load:
            self.parser.error('--dump and --load cannot be used together.')

        if args.min_qubits is not None and args.ibm:
            self.parser.error('--min-qubits can only be used if --ibm is selected.')

        if args.convergence and args.threshold is None:
            self.parser.error('--threshold must be set if --convergence is enabled.')

        if self.kafka_params_set(args) and not args.kafka:
            self.parser.error('--kafka must be set for the options to take effect.')

        if args.sasl_mechanism and not args.kafka:
            self.parser.error('--kafka must be enabled when using SASL authentication')

        if args.ssl and not args.kafka:
            self.parser.error('--kafka must be enabled when using SSL authentication')

        if args.ssl:
            if args.ssl_dir is not None and self.ensure_dir(args.ssl_dir):
                individual_files = [
                    args.ssl_cafile,
                    args.ssl_certfile,
                    args.ssl_keyfile,
                    args.ssl_crlfile,
                ]
                if any(f is not None for f in individual_files):
                    self.parser.error(
                        'Cannot specify both --ssl-dir and individual SSL file options (--ssl-cafile, etc.)'
                    )
            else:
                required_files = [args.ssl_cafile, args.ssl_certfile, args.ssl_keyfile]
                if not all(required_files):
                    self.parser.error(
                        (
                            '--ssl-cafile, --ssl-certfile, and --ssl-keyfile '
                            'are required when --ssl is set '
                            'and --ssl-dir is not provided'
                        )
                    )
        elif not args.ssl and (args.ssl_cafile or args.ssl_certfile or args.ssl_keyfile):
            self.parser.error(
                '--ssl is required when specifying --ssl-cafile, --ssl-certfile, and --ssl-keyfile'
            )

        sasl_options_provided = (
            args.sasl_ssl is True
            and args.sasl_plain_username is not None
            or args.sasl_plain_password is not None
            or args.sasl_kerberos_service_name != 'kafka'
            or args.sasl_kerberos_domain_name is not None
        )

        if args.sasl_ssl:
            if sasl_options_provided and not args.sasl_mechanism:
                self.parser.error('--sasl-mechanism is required when SASL options are provided')

            if args.sasl_mechanism:
                if args.sasl_mechanism in ['PLAIN', 'SCRAM-SHA-256', 'SCRAM-SHA-512']:
                    if not (args.sasl_plain_username and args.sasl_plain_password):
                        self.parser.error(
                            f'--sasl-plain-username and --sasl-plain-password are required for {args.sasl_mechanism}'
                        )
                    if args.sasl_kerberos_domain_name is not None:
                        self.parser.error(f'Cannot use GSSAPI options with {args.sasl_mechanism}')
                elif args.sasl_mechanism == 'GSSAPI':
                    if (
                        args.sasl_plain_username is not None
                        or args.sasl_plain_password is not None
                    ):
                        self.parser.error('Cannot use PLAIN/SCRAM options with GSSAPI')

    def parse_args(self) -> argparse.Namespace:
        """Parse and validate command line arguments."""
        args = self.parser.parse_args()
        self._validate_args(args)
        return args

    def get_config(self) -> dict[str, Any]:
        """Get the configuration from the parsed arguments."""
        config_manager = ConfigurationManager()
        args = self.parse_args()
        return config_manager.get_config(args)
