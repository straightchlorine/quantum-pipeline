from qiskit.circuit.library import TwoLocal
from qiskit_algorithms import VQE
from qiskit.primitives import Estimator


def solve_vqe(qubit_op, ansatz_reps=2, optimizer="COBYLA"):
    """Solve the qubit operator using VQE."""
    from qiskit_algorithms.optimizers import COBYLA, SPSA

    optimizers = {"COBYLA": COBYLA(), "SPSA": SPSA()}
    if optimizer not in optimizers:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    ansatz = TwoLocal(rotation_blocks="ry", entanglement_blocks="cz", reps=ansatz_reps)
    vqe = VQE(estimator=Estimator(), ansatz=ansatz, optimizer=optimizers[optimizer])

    result = vqe.compute_minimum_eigenvalue(qubit_op)
    return result.eigenvalue.real
