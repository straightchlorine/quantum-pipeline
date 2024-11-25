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
from quantum_simulation.utils.logger import get_logger

logger = get_logger('VQESolver')

# Predefined optimizer configurations
SUPPORTED_OPTIMIZERS = {
    'COBYLA': COBYLA(maxiter=100),
    'SPSA': SPSA(maxiter=50),
}


def solve_vqe(qubit_op, ansatz_reps=None, optimizer=None):
    """
    Solves the given qubit operator using VQE.

    Args:
        qubit_op: The operator to minimize (Qiskit's PauliSumOp or similar).
        ansatz_reps: (Optional) Number of repetitions in the variational ansatz.
        optimizer: (Optional) The optimizer to use ('COBYLA' or 'SPSA').

    Returns:
        float: The minimum eigenvalue found by VQE.

    Raises:
        ValueError: If an unsupported optimizer is passed.
        RuntimeError: If VQE fails to converge.
    """
    # set defaults if not provided
    ansatz_reps = ansatz_reps or settings.ANSATZ_REPS
    optimizer = optimizer or settings.OPTIMIZER

    # validate optimizer
    if optimizer not in SUPPORTED_OPTIMIZERS:
        supported = list(SUPPORTED_OPTIMIZERS.keys())
        raise ValueError(
            f'Unsupported optimizer: {optimizer}. Supported: {supported}'
        )

    # create variational ansatz
    ansatz = TwoLocal(
        rotation_blocks='ry', entanglement_blocks='cz', reps=ansatz_reps
    )

    # configure VQE instance
    vqe = VQE(
        estimator=Estimator(),
        ansatz=ansatz,
        optimizer=SUPPORTED_OPTIMIZERS[optimizer],
    )

    # log the start of VQE execution
    logger.info(
        f'Starting VQE with optimizer={optimizer}, ansatz_reps={ansatz_reps}'
    )

    # execute VQE to compute the minimum eigenvalue
    result = vqe.compute_minimum_eigenvalue(qubit_op)

    # handle failure to converge
    if result.eigenvalue is None:
        logger.error('VQE failed to converge.')
        raise RuntimeError('VQE did not produce a valid eigenvalue.')

    # log and return the result
    logger.info(f'VQE Converged. Minimum energy: {result.eigenvalue.real:.6f}')

    return result.eigenvalue.real
