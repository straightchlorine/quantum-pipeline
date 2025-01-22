"""
energy_plotter.py

This module visualizes the energy convergence during optimization.
"""

import matplotlib.pyplot as plt
import logging

from quantum_pipeline.configs import settings
from quantum_pipeline.utils.dir import savePlot
from quantum_pipeline.structures.vqe_observation import VQEProcess

logger = logging.getLogger(__name__)


class EnergyPlotter:
    """
    A utility class to plot energy values from VQE runs.
    """

    def __init__(self, iterations: list[VQEProcess], symbols, max_points=100):
        """
        Initializes the EnergyPlotter.

        Args:
            max_points: Maximum number of points to display on the graph.
        """
        self.iterations = [iteration.iteration for iteration in iterations]
        self.energy_values = [iteration.result for iteration in iterations]
        self.stds = [iteration.std for iteration in iterations]
        self.symbols = symbols
        self.max_points = max_points

    def _filter_points(self):
        """
        Filters points to avoid clutter on the graph.

        Returns:
            Tuple of filtered iterations and energy values.
        """
        total_points = len(self.iterations)
        if total_points <= self.max_points:
            return self.iterations, self.energy_values

        # downsample points to a maximum of max_points
        step = max(1, total_points // self.max_points)
        filtered_iterations = self.iterations[::step]
        filtered_energy_values = self.energy_values[::step]

        # last point always included
        if self.iterations[-1] not in filtered_iterations:
            filtered_iterations.append(self.iterations[-1])
            filtered_energy_values.append(self.energy_values[-1])

        return filtered_iterations, filtered_energy_values

    def plot_convergence(self, title='Energy Convergence'):
        """
        Plots the energy convergence.

        Args:
            title: Title of the plot.
            save_path: Path to save the plot (optional).
        """
        iterations, energy_values = self._filter_points()

        plt.figure(figsize=(10, 6))
        # plt.plot(iterations, energy_values, marker='o', label='Energy')
        plt.errorbar(
            iterations,
            energy_values,
            yerr=self.stds,
            fmt='o-',
            label='Energy',
            capsize=5,
            capthick=1,
            elinewidth=1,
            markersize=4,
        )

        plt.xlabel('Iteration')
        plt.ylabel('Energy (a.u.)')
        plt.title(title)
        plt.legend()
        plt.grid()

        plot_path = savePlot(
            plt,
            settings.ENERGY_CONVERGENCE_PLOT_DIR,
            settings.ENERGY_CONVERGENCE_PLOT,
            self.symbols,
        )
        return plot_path
