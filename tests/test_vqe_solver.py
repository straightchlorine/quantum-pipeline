from quantum_simulation.solvers.vqe_solver import solve_vqe


def test_vqe_solver():
    # Mock a qubit operator (use a simple identity operator for testing)
    from qiskit.quantum_info import Pauli

    qubit_op = Pauli("2I")
    energy = solve_vqe(qubit_op)
    assert energy < 0  # Replace with an appropriate energy check
