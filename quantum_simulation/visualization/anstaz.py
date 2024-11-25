"""
ansatz_viewer.py

This module visualizes the ansatz circuit used in quantum algorithms.
"""

import matplotlib.pyplot as plt


class AnsatzViewer:
    """
    A utility class for visualizing ansatz circuits.
    """

    @staticmethod
    def display_circuit(ansatz):
        """
        Displays the ansatz circuit diagram.

        Args:
            ansatz: Qiskit QuantumCircuit object.
        """
        ansatz.draw(output='mpl')
        plt.show()

    @staticmethod
    def save_circuit(ansatz, save_path):
        """
        Saves the ansatz circuit diagram.

        Args:
            ansatz: Qiskit QuantumCircuit object.
            save_path: Path to save the diagram.
        """
        fig = ansatz.draw(output='mpl')
        fig.savefig(save_path)
        print(f'Circuit diagram saved at {save_path}')
