from qiskit import QuantumCircuit
from collections import Counter


class CircuitFeatures:
    @staticmethod
    def extract_features(circuit: QuantumCircuit):
        """
        Extract features from a quantum circuit, including depth, gate counts,
        and entanglement.

        Args:
            circuit (QuantumCircuit): The quantum circuit.

        Returns:
            dict: A dictionary containing circuit features.
        """
        gate_counts = Counter([gate[0].name for gate in circuit.data])
        features = {
            'depth': circuit.depth(),
            'qubits': circuit.num_qubits,
            'ancillas': circuit.num_ancillas,
            'clbits': circuit.num_clbits,
            'gate_counts': gate_counts,
        }
        return features
