"""
jordan_wigner_mapper.py

This module defines a class for mapping fermionic operators to qubit operators
using the Jordan-Wigner transformation. This transformation is used to represent
fermionic systems on quantum computers.
"""

from qiskit_nature.second_q.mappers import JordanWignerMapper as JWM
from quantum_pipeline.mappers.mapper import Mapper


class JordanWignerMapper(Mapper):
    """
    A wrapper for the Jordan-Wigner mapper provided by Qiskit Nature.
    """

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

        # Perform the Jordan-Wigner mapping
        mapper = JWM()
        qubit_op = mapper.map(operator)
        return qubit_op
