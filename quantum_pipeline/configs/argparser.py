from dataclasses import dataclass
import logging
from typing import Any, Callable, Dict
import argparse

from qiskit_ibm_runtime import IBMBackend
from quantum_pipeline.configs import settings
from quantum_pipeline.configs.defaults import DEFAULTS
from quantum_pipeline.configs.settings import LOG_LEVEL
from quantum_pipeline.utils.dir import ensureDirExists

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

        self._add_arguments()

    def initialize_simulation_environment(self):
        """Ensure required directories and configurations are in place."""
        ensureDirExists(settings.GEN_DIR)
        ensureDirExists(settings.GRAPH_DIR)
        ensureDirExists(settings.REPORT_DIR)

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
            '--kafka', action='store_true', help='Connect to Apache Kafka for streaming data'
        )

    def _validate_args(self, args: argparse.Namespace) -> None:
        """Validate parsed arguments."""
        if args.min_qubits is not None and not args.ibm_quantum:
            self.parser.error('--min-qubits can only be used if --ibm-quantum is selected.')

        if args.convergence and args.threshold is None:
            self.parser.error('--threshold must be set if --convergence is enabled.')

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

    def get_simulation_kwargs(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Convert parsed arguments dynamically into a dictionary."""
        return {key: value for key, value in vars(args).items() if key != 'local'}
