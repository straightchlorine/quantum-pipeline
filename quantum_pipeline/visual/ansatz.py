"""
ansatz.py

This module visualizes the ansatz circuit used in quantum algorithms.
"""

from qiskit.visualization import circuit_drawer

from quantum_pipeline.configs import settings
from quantum_pipeline.utils.dir import getGraphPath
from quantum_pipeline.utils.logger import get_logger


class AnsatzViewer:
    """
    A utility class for visualizing ansatz circuits.
    """

    def __init__(self, ansatz, symbols):
        self.logger = get_logger(self.__class__.__name__)
        self.ansatz = ansatz
        self.symbols = symbols

    def save_circuit(self):
        """
        Saves the given ansatz circuit and its decomposed version as images.

        Args:
            ansatz (QuantumCircuit): Qiskit QuantumCircuit object.
            symbols (str): Additional symbols to append to the saved filename.
        """

        # save the ansatz circuit
        try:
            circuit_drawer(
                self.ansatz,
                output='mpl',
                filename=str(
                    getGraphPath(settings.ANSATZ_PLOT_DIR, settings.ANSATZ, self.symbols)
                ),
            )
        except Exception as e:
            self.logger.error(f'Unable to save ansatz: {e}')

        # save the decomposed ansatz circuit
        try:
            circuit_drawer(
                self.ansatz.decompose(),
                output='mpl',
                filename=str(
                    getGraphPath(
                        settings.ANSATZ_DECOMPOSED_PLOT_DIR,
                        settings.ANSATZ_DECOMPOSED,
                        self.symbols,
                    )
                ),
            )
        except Exception as e:
            self.logger.error(f'Unable to save decomposed ansatz: {e}')
