from qiskit.quantum_info import SparsePauliOp
from collections import Counter


class HamiltonianFeatures:
    @staticmethod
    def extract_features(operator: SparsePauliOp):
        """Extract features from a Hamiltonian (SparsePauliOp).

        Args:
            operator (SparsePauliOp): The Hamiltonian as a PauliSumOp or
            SparsePauliOp.

        Returns:
            dict: A dictionary of Hamiltonian features.
        """
        pauli_counts = Counter(str(pauli) for pauli in operator.paulis)
        pauli_terms = [
            {'pauli': str(pauli), 'coefficient': float(coeff)}
            for pauli, coeff in zip(
                [str(pauli) for pauli in operator.paulis], operator.coeffs
            )
        ]

        features = {
            'num_pauli_terms': len(operator),
            'pauli_distribution': pauli_counts,
            'pauli_terms': pauli_terms,
        }
        return features
