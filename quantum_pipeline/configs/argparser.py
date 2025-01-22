import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict

from qiskit_ibm_runtime import IBMBackend

from quantum_pipeline.configs import settings
from quantum_pipeline.configs.defaults import DEFAULTS
from quantum_pipeline.stream.kafka_interface import ProducerConfig
from quantum_pipeline.utils.dir import ensureDirExists
from quantum_pipeline.utils.logger import get_logger

SUPPORTED_OPTIMIZERS = {
    'Nelder-Mead': 'A simplex algorithm for unconstrained optimization.',
    'Powell': 'A directional set method for unconstrained optimization.',
    'CG': 'Non-linear conjugate gradient method for unconstrained optimization.',
    'BFGS': 'Quasi-Newton method using the Broyden–Fletcher–Goldfarb–Shanno algorithm.',
    'Newton-CG': "Newton's method with conjugate gradient for unconstrained optimization.",
    'L-BFGS-B': 'Limited-memory BFGS with box constraints.',
    'TNC': 'Truncated Newton method for bound-constrained optimization.',
    'COBYLA': 'Constrained optimization by linear approximations.',
    'COBYQA': 'Constrained optimization by quadratic approximations.',
    'SLSQP': 'Sequential Least Squares Programming for constrained optimization.',
    'trust-constr': 'Trust-region method for constrained optimization.',
    'dogleg': 'Dog-leg trust-region algorithm for unconstrained optimization.',
    'trust-ncg': 'Trust-region Newton conjugate gradient method.',
    'trust-exact': 'Exact trust-region optimization.',
    'trust-krylov': 'Trust-region method with Krylov subspace solver.',
    'custom': 'A user-provided callable object implementing the optimization method.',
}


@dataclass
class BackendConfig:
    """Dataclass for storing backend filter."""

    local: bool | None
    min_num_qubits: int | None
    filters: Callable[[IBMBackend], bool] | None

    def to_dict(self):
        """Convert the dataclass to a dictionary."""
        return {
            key: value
            for key, value in self.__dict__.items()
            if value is not None and key != 'local'
        }

    def toJSON(self) -> str:
        """Convert the configuration to a JSON string."""
        return json.dumps(self.to_dict())

    def local_backend(self):
        """Check if the backend is local."""
        return self.local

    @classmethod
    def _get_local_backend(cls):
        return cls(local=True, min_num_qubits=None, filters=None)


class QuantumPipelineArgParser:
    """Class for handling command line argument parsing for quantum pipeline."""

    def __init__(self):
        """Initialize the parser with default configuration."""
        self.initialize_simulation_environment()
        self.parser = argparse.ArgumentParser(
            description='Quantum Circuit Simulation and Execution',
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        self.logger = get_logger('Argparser')

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
        for optimizer, description in SUPPORTED_OPTIMIZERS.items():
            help_text += f'\n  {optimizer}: {description}'
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
            '--local',
            action='store_true',
            default=DEFAULTS['local'],
            help='Using local backend for simulation (otherwise IBM Quantum is used.)',
        )
        sim_group.add_argument(
            '--min-qubits', type=int, help='Minimum number of qubits required for the backend'
        )

    def _add_kafka_config(self):
        """Add simulation configuration arguments."""
        kafka_group = self.parser.add_argument_group('Kafka Configuration')
        kafka_group.add_argument(
            '--kafka', action='store_true', help='Enable streaming results to Apache Kafka'
        )
        kafka_group.add_argument(
            '--host',
            type=str,
            default=DEFAULTS['kafka']['server'],
            help='Apache Kafka instance address',
        )
        kafka_group.add_argument(
            '--topic',
            type=str,
            default=DEFAULTS['kafka']['topic'],
            help='Name of the topic, with which message will be categorised with',
        )
        kafka_group.add_argument(
            '-ret',
            '--retries',
            type=str,
            default=DEFAULTS['kafka']['retries'],
            help='Number of attempts to send message to kafka',
        )
        kafka_group.add_argument(
            '-iret',
            '--internal-retries',
            type=int,
            default=DEFAULTS['kafka']['internal_retries'],
            help='Number of attempts kafka should attempt automatically (introduces risk of duplicate entries)',
        )
        kafka_group.add_argument(
            '-a',
            '--acks',
            default=DEFAULTS['kafka']['acks'],
            choices=['0', '1', 'all'],
            help='Number of acknowledgments producer requires to receive before a request is complete.',
        )
        kafka_group.add_argument(
            '-t',
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
            choices=list(SUPPORTED_OPTIMIZERS.keys()),
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

    def _add_backend_options(self):
        """Add advanced backend configuration options."""
        backend_group = self.parser.add_argument_group('Advanced Backend Options')
        backend_group.add_argument(
            '--shots',
            type=int,
            default=DEFAULTS['shots'],
            help='Number of shots for quantum circuit execution',
        )
        backend_group.add_argument(
            '--optimization-level',
            type=int,
            choices=[0, 1, 2, 3],
            default=DEFAULTS['optimization_level'],
            help='Circuit optimization level',
        )

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

    def kafka_params_set(self, args: argparse.Namespace):
        if (
            args.host != DEFAULTS['kafka']['server']
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
        if args.min_qubits is not None and not args.ibm_quantum:
            self.parser.error('--min-qubits can only be used if --ibm-quantum is selected.')

        if args.convergence and args.threshold is None:
            self.parser.error('--threshold must be set if --convergence is enabled.')

        if self.kafka_params_set(args) and not args.kafka:
            self.parser.error('--kafka must be set for the options to take effect.')
        settings.LOG_LEVEL = getattr(logging, args.log_level)

    def parse_args(self) -> argparse.Namespace:
        """Parse and validate command line arguments."""
        args = self.parser.parse_args()
        self._validate_args(args)
        return args

    def create_backend_config(self, args: argparse.Namespace) -> BackendConfig:
        """Create BackendConfig from parsed arguments."""
        return BackendConfig(
            local=args.local,
            min_num_qubits=args.min_qubits,
            filters=None,
        )

    def create_kafka_config(self, args: argparse.Namespace) -> ProducerConfig:
        """Create ProducerConfig from parsed arguments"""
        return ProducerConfig(
            bootstrap_servers=args.host,
            topic=args.topic,
            retries=args.retries,
            kafka_retries=args.internal_retries,
            acks=args.acks,
            timeout=args.timeout,
        )

    def get_simulation_kwargs(self, args: argparse.Namespace, dump=False) -> Dict[str, Any]:
        """Convert parsed arguments dynamically into a dictionary."""
        kwargs = {key: value for key, value in vars(args).items() if key != 'local'}

        kwargs['kafka_config'] = self.create_kafka_config(args)
        kwargs['backend_config'] = self.create_backend_config(args)

        if args.dump:
            file_name = Path(args.file).stem
            basis_set = args.basis
            optimizer = args.optimizer
            backend_local = 'local' if args.local else 'api'
            current_date = datetime.now().strftime('%Y%m%d')

            file_path = (
                f'config-{file_name}-{basis_set}-{optimizer}-{backend_local}-{current_date}.json'
            )
            with open(file_path, 'w') as file:
                # TODO: fix this, so that configs are seriable
                json.dump(kwargs, file, indent=4)
            self.logger.info(f'Configuration saved to {file_path}')

        return kwargs
