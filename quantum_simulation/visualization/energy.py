"""
energy_plotter.py

This module visualizes the energy convergence during optimization.
"""

import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


class EnergyPlotter:
    """
    A utility class to plot energy values from VQE runs.
    """

    def __init__(self):
        self.iterations = []
        self.energy_values = []

    def add_iteration(self, iteration, energy):
        """
        Adds an iteration's energy value for plotting.

        Args:
            iteration: The iteration number (int).
            energy: The energy value (float).
        """
        self.iterations.append(iteration)
        self.energy_values.append(energy)
        logger.info(f'Iteration {iteration}: Energy = {energy:.6f}')

    def plot_convergence(self, title='Energy Convergence', save_path=None):
        """
        Plots the energy convergence.

        Args:
            title: Title of the plot.
            save_path: Path to save the plot (optional).
        """
        plt.figure(figsize=(10, 6))
        plt.plot(
            self.iterations, self.energy_values, marker='o', label='Energy'
        )
        plt.xlabel('Iteration')
        plt.ylabel('Energy (a.u.)')
        plt.title(title)
        plt.legend()
        plt.grid()

        if save_path:
            plt.savefig(save_path)
            logger.info(f'Convergence plot saved at {save_path}.')
        else:
            plt.show()
