from qiskit.visualization import circuit_drawer

from quantum_pipeline.configs import settings
from quantum_pipeline.utils.dir import get_graph_path
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
        """Save the ansatz circuit and its decomposed version as images."""
        try:
            circuit_drawer(
                self.ansatz,
                output='mpl',
                filename=str(
                    get_graph_path(settings.ANSATZ_PLOT_DIR, settings.ANSATZ, self.symbols)
                ),
            )
        except Exception as e:
            self.logger.error(f'Unable to save ansatz: {e}')

        # saving the decomposed circuit
        try:
            circuit_drawer(
                self.ansatz.decompose(),
                output='mpl',
                filename=str(
                    get_graph_path(
                        settings.ANSATZ_DECOMPOSED_PLOT_DIR,
                        settings.ANSATZ_DECOMPOSED,
                        self.symbols,
                    )
                ),
            )
        except Exception as e:
            self.logger.error(f'Unable to save decomposed ansatz: {e}')
