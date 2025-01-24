import argparse
import logging
import os
from typing import Any, Dict

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

    def _add_arguments(self):
        """Add all argument groups and their arguments to the parser."""
        self._add_required_arguments()
        self._add_simulation_config()
        self._add_vqe_parameters()
        self._add_output_logging()
        self._add_backend_options()
        self._add_kafka_config()
        self._add_additional_features()

    def _add_required_arguments(self):
        """Add required arguments to the parser."""
        self.parser.add_argument(
            '-f', '--file', required=True, help='Path to molecule data file (JSON)'
        )

    def _add_simulation_config(self):
        """Add simulation configuration arguments."""
        sim_group = self.parser.add_argument_group('Simulation Configuration')
        sim_group.add_argument(
            '-b', '--basis', default=DEFAULTS['basis_set'], help='Basis set for the simulation'
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
            default=DEFAULTS['kafka']['retries'],
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

    def dir_path(self, path):
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
            type=self.dir_path,
            help='Path to a JSON configuration file to load parameters from',
        )

    def kafka_params_set(self, args: argparse.Namespace):
        if (
            args.servers != DEFAULTS['kafka']['servers']
            or args.topic != DEFAULTS['kafka']['topic']
            or args.retries != DEFAULTS['kafka']['retries']
            or args.internal_retries != DEFAULTS['kafka']['internal_retries']
            or args.acks != DEFAULTS['kafka']['acks']
            or args.timeout != DEFAULTS['kafka']['timeout']
        ):
            return True
        return False

    def _validate_args(self, args: argparse.Namespace) -> None:
        """Validate parsed arguments."""
        settings.LOG_LEVEL = getattr(logging, args.log_level)

        if args.dump and args.load:
            self.parser.error('--dump and --load cannot be used together.')

        if args.min_qubits is not None and not args.ibm_quantum:
            self.parser.error('--min-qubits can only be used if --ibm-quantum is selected.')

        if args.convergence and args.threshold is None:
            self.parser.error('--threshold must be set if --convergence is enabled.')

        if self.kafka_params_set(args) and not args.kafka:
            self.parser.error('--kafka must be set for the options to take effect.')

    def parse_args(self) -> argparse.Namespace:
        """Parse and validate command line arguments."""
        args = self.parser.parse_args()
        self._validate_args(args)
        return args

    def get_config(self) -> Dict[str, Any]:
        """Get the configuration from the parsed arguments."""
        config_manager = ConfigurationManager()
        args = self.parse_args()
        return config_manager.get_config(args)
