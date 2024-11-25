from qiskit.circuit.library import TwoLocal
from qiskit_algorithms import VQE
from qiskit.primitives import Estimator
from quantum_simulation.utils.logger import get_logger
from quantum_simulation.configs import settings

logger = get_logger("QuantumAtomicSim")


def solve_vqe(qubit_op, ansatz_reps=None, optimizer=None):
    """Solve the qubit operator using VQE."""
    from qiskit_algorithms.optimizers import COBYLA, SPSA

    optimizers = {"COBYLA": COBYLA(), "SPSA": SPSA()}
    optimizer = optimizer or settings.OPTIMIZER
    ansatz_reps = ansatz_reps or settings.ANSATZ_REPS

    if optimizer not in optimizers:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    ansatz = TwoLocal(rotation_blocks="ry", entanglement_blocks="cz", reps=ansatz_reps)
    vqe = VQE(estimator=Estimator(), ansatz=ansatz, optimizer=optimizers[optimizer])

    # start the computation
    logger.info(f"Starting VQE with optimizer={optimizer}, reps={ansatz_reps}")
    result = vqe.compute_minimum_eigenvalue(qubit_op)

    # validate the result
    if result.eigenvalue is None:
        logger.error("VQE failed to converge.")
        raise RuntimeError("VQE did not produce a valid eigenvalue.")

    # log and return the result
    logger.info(f"VQE Converged. Energy: {result.eigenvalue:.6f} Hartree")
    return result.eigenvalue.real
