"""
vqe_solver.py

This module contains a function to solve a quantum operator using the Variational
Quantum Eigensolver (VQE). The VQE algorithm combines quantum and classical
optimization to find the minimum eigenvalue of a Hamiltonian.
"""

from qiskit.circuit.library import TwoLocal
from qiskit.primitives import Estimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA, SPSA

from quantum_simulation.configs import settings
from quantum_simulation.report.report_generator import ReportGenerator
from quantum_simulation.utils.logger import get_logger

logger = get_logger('VQESolver')


class VQESolver:
    # Predefined optimizer configurations
    SUPPORTED_OPTIMIZERS = {
        'COBYLA': COBYLA(maxiter=100),
        'SPSA': SPSA(maxiter=50),
    }

    def __init__(
        self,
        qubit_op,
        ansatz_reps=None,
        optimizer_name=None,
        max_iterations=100,
        custom_optimizer=None,
        report_generator=None,
        report_name='vqe_report.pdf',
        report_description='VQE Results',
    ):
        self.report_name = report_name
        self.report_description = report_description

        self.report: ReportGenerator = report_generator
        # self.report.add_summary(self.report_description)

        self.qubit_op = qubit_op
        self.ansatz_reps = ansatz_reps
        self.optimizer_name = optimizer_name
        self.max_iterations = max_iterations
        self.custom_optimizer = custom_optimizer

    def solve(self):
        """
        Solves the given qubit operator using VQE and generates a report.

        Args:
            qubit_op: The operator to minimize (Qiskit's PauliSumOp or similar).
            ansatz_reps: (Optional) Number of repetitions in the variational ansatz.
            optimizer_name: (Optional) The optimizer to use ('COBYLA' or 'SPSA').
            max_iterations: (Optional) Maximum number of iterations for the optimizer.
            custom_optimizer: (Optional) A custom optimizer instance.
            report_name: (Optional) The name of the report file to generate.
            report_description: (Optional) Description for the generated report.

        Returns:
            float: The minimum eigenvalue found by VQE.

        Raises:
            ValueError: If an unsupported optimizer is passed and no custom optimizer is provided.
            RuntimeError: If VQE fails to converge or produces an invalid result.
        """
        # set to default if not provided
        ansatz_reps = self.ansatz_reps or settings.ANSATZ_REPS
        optimizer_name = self.optimizer_name or settings.OPTIMIZER

        predefined_optimizers = {
            'COBYLA': COBYLA(maxiter=self.max_iterations),
            'SPSA': SPSA(maxiter=self.max_iterations),
        }

        if self.custom_optimizer:
            optimizer = self.custom_optimizer
            logger.info('Using custom optimizer.')
        elif optimizer_name in predefined_optimizers:
            optimizer = predefined_optimizers[optimizer_name]
            logger.info(f'Using predefined optimizer: {optimizer_name}')
        else:
            supported = list(predefined_optimizers.keys())
            raise ValueError(
                f'Unsupported optimizer: {optimizer_name}. Supported: {supported}'
            )

        ansatz = TwoLocal(
            rotation_blocks='ry', entanglement_blocks='cz', reps=ansatz_reps
        )
        logger.debug(f'Created ansatz with {ansatz_reps} repetitions.')

        vqe = VQE(estimator=Estimator(), ansatz=ansatz, optimizer=optimizer)
        logger.info('VQE instance configured.')

        logger.info('Starting VQE computation...')
        try:
            result = vqe.compute_minimum_eigenvalue(self.qubit_op)
        except Exception as e:
            logger.error(f'VQE execution failed: {str(e)}')
            raise RuntimeError('VQE execution encountered an error.') from e

        if result.eigenvalue is None:
            logger.error('VQE did not converge to a valid result.')
            raise RuntimeError('VQE did not produce a valid eigenvalue.')

        min_energy = result.eigenvalue.real
        logger.info(f'VQE Converged. Minimum energy: {min_energy:.6f}')

        # add the metrics to the report
        try:
            self.report.add_insight(
                'Energy analysis', 'Results of VQE algorihtm:'
            )
            self.report.add_metrics(
                {
                    'Minimum Energy': str(round(min_energy, 4)) + ' Heartree',
                    'Optimizer': optimizer_name,
                    'Iterations': self.max_iterations,
                    'Ansatz Repetitions': ansatz_reps,
                }
            )
        except Exception as e:
            logger.error(f'Failed to generate report: {str(e)}')

        return min_energy
