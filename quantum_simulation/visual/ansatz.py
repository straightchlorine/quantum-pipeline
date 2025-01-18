"""
ansatz.py

This module visualizes the ansatz circuit used in quantum algorithms.
"""

import matplotlib.pyplot as plt
from qiskit.visualization import circuit_drawer
from quantum_simulation.configs import settings
from quantum_simulation.utils.dir import getGraphPath, savePlot


class AnsatzViewer:
    """
    A utility class for visualizing ansatz circuits.
    """

    @staticmethod
    def save_circuit(ansatz, symbols):
        """
        Saves the given ansatz circuit and its decomposed version as images.

        Args:
            ansatz (QuantumCircuit): Qiskit QuantumCircuit object.
            symbols (str): Additional symbols to append to the saved filename.
        """

        # save the ansatz circuit
        circuit_drawer(
            ansatz,
            output='mpl',
            filename=str(getGraphPath(settings.ANSATZ_PLOT_DIR, settings.ANSATZ, symbols)),
        )

        # save the decomposed ansatz circuit
        # TODO: bring back!!
        # circuit_drawer(
        #     ansatz.decompose(),
        #     output='mpl',
        #     filename=str(
        #         getGraphPath(
        #             settings.ANSATZ_DECOMPOSED_PLOT_DIR,
        #             settings.ANSATZ_DECOMPOSED,
        #             symbols,
        #         )
        #     ),
        # )
