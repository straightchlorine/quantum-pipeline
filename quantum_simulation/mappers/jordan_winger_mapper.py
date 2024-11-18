from qiskit_nature.second_q.mappers import JordanWignerMapper


def map_to_qubits(fermionic_op):
    """Map fermionic operator to qubit operator using Jordan-Wigner."""
    mapper = JordanWignerMapper()
    return mapper.map(fermionic_op)
