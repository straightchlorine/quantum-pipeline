"""
state_visualizer.py

This module visualizes quantum state probabilities.
"""

import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram


class StateVisualizer:
    """
    A utility class for visualizing quantum state probabilities.
    """

    @staticmethod
    def plot_state_probabilities(statevector):
        """
        Plots the probabilities of quantum states from a state vector.

        Args:
            statevector: Qiskit Statevector object.
        """
        probabilities = statevector.probabilities_dict()
        plot_histogram(probabilities)
        plt.show()

    @staticmethod
    def save_state_probabilities(statevector, save_path):
        """
        Saves the histogram of state probabilities.

        Args:
            statevector: Qiskit Statevector object.
            save_path: Path to save the histogram.
        """
        probabilities = statevector.probabilities_dict()
        fig = plot_histogram(probabilities)
        fig.savefig(save_path)
        print(f'State probabilities histogram saved at {save_path}.')
