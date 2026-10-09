"""
jordan_wigner_mapper.py

Module defines a class for mapping fermionic operators to qubit operators
using the Jordan-Wigner transformation.
"""

from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.mappers import JordanWignerMapper as JordanWignerMapperQiskit

from quantum_pipeline.mappers.mapper import Mapper


class JordanWignerMapper(Mapper):
    def map(self, operator):
        """
        Maps a fermionic operator to a qubit operator.

        Args:
            operator: The fermionic operator to map (Qiskit's FermionicOp).

        Returns:
            Qubit operator (Qiskit's PauliSumOp).

        Raises:
            ValueError: If the input operator is invalid or None.
        """
        if operator is None:
            raise ValueError('The input operator must not be None.')

        # empty operator has no orbitals to map.
        if operator.register_length == 0:
            return SparsePauliOp([''], coeffs=[0j])

        # run the mapper
        return JordanWignerMapperQiskit().map(operator)

    def get_qiskit_mapper(self):
        return JordanWignerMapperQiskit()
