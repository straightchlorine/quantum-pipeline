"""
operator_viewer.py

This module visualizes the coefficients of qubit operators.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from quantum_pipeline.configs import settings
from quantum_pipeline.utils.dir import savePlot

matplotlib.use('Agg')


class OperatorViewer:
    """Utility class to visualize qubit operator coefficients.

    That also included their real and imaginary parts, and their magnitudes
    """

    @staticmethod
    def plot_operator_coefficients(
        qubit_op,
        symbols,
        title='Operator Coefficients',
        threshold=1e-2,
        max_terms=50,
    ):
        """
        Visualizes the coefficients of the qubit operator as a bar plot,
        filtering out negligible terms and supporting dynamic subsampling.

        Args:
            qubit_op: Qiskit's PauliSumOp or similar operator.
            symbols: Identifier for the plot naming.
            title: Title of the plot.
            threshold: Minimum coefficient magnitude to display.
            max_terms: Maximum number of terms to display.

        Returns:
            str: Path to the saved plot.
        """
        # convert coefficients to a list
        terms = qubit_op
        operators = [term[0] for term in terms]
        coefficients = [term[1] for term in terms]

        # separate real and imaginary parts
        real_parts = [coeff.real for coeff in coefficients]
        imag_parts = [coeff.imag for coeff in coefficients]

        # filter out insignificant terms (smaller than threshold)
        significant_indices = [i for i, coeff in enumerate(coefficients) if abs(coeff) > threshold]
        filtered_real_parts = [real_parts[i] for i in significant_indices]
        filtered_imag_parts = [imag_parts[i] for i in significant_indices]
        filtered_operators = [operators[i] for i in significant_indices]

        # group up filtered out terms
        other_real = sum(
            real_parts[i] for i in range(len(real_parts)) if i not in significant_indices
        )
        other_imag = sum(
            imag_parts[i] for i in range(len(imag_parts)) if i not in significant_indices
        )

        if other_real or other_imag:
            filtered_real_parts.append(other_real)
            filtered_imag_parts.append(other_imag)
            filtered_operators.append('Other')

        # displayed terms must be limited to make the plot readable
        if len(filtered_operators) > max_terms:
            indices = np.linspace(0, len(filtered_operators) - 1, max_terms, dtype=int)
            filtered_real_parts = [filtered_real_parts[i] for i in indices]
            filtered_imag_parts = [filtered_imag_parts[i] for i in indices]
            filtered_operators = [filtered_operators[i] for i in indices]

        # create the figure
        fig, ax = plt.subplots(figsize=(14, 8))

        # bar plot
        indices = np.arange(len(filtered_operators))
        bar_width = 0.4
        ax.barh(
            indices - bar_width / 2,
            filtered_real_parts,
            bar_width,
            label='Real',
            color='blue',
            alpha=0.7,
        )
        ax.barh(
            indices + bar_width / 2,
            filtered_imag_parts,
            bar_width,
            label='Imaginary',
            color='red',
            alpha=0.7,
        )

        # set the title and labels
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Coefficient Value', fontsize=12)
        ax.set_ylabel('Term Index / Label', fontsize=12)
        ax.set_yticks(indices)
        ax.set_yticklabels(filtered_operators, fontsize=8)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        ax.legend(fontsize=10)

        # set layout and save the plot
        plt.tight_layout()
        plot_path = savePlot(
            plt,
            settings.OPERATOR_COEFFS_PLOT_DIR,
            settings.OPERATOR_COEFFS_PLOT,
            symbols,
        )

        plt.close(fig)
        return plot_path

    @staticmethod
    def plot_complex_coefficients_polar(
        qubit_op,
        symbols,
        title='Operator Coefficients (Polar)',
        threshold=1e-2,
        max_terms=50,
    ):
        """
        Visualizes the coefficients of the qubit operator in polar coordinates,
        filtering out negligible terms and supporting dynamic subsampling.

        Args:
            qubit_op: Qiskit's PauliSumOp or similar operator.
            symbols: Identifier for the plot naming.
            title: Title of the plot.
            threshold: Minimum coefficient magnitude to display.
            max_terms: Maximum number of terms to display.

        Returns:
            str: Path to the saved plot.
        """

        # terms and operators to lists
        terms = qubit_op
        operators = [term[0] for term in terms]
        coefficients = [term[1] for term in terms]

        # calculate magnitudes and phases
        magnitudes = np.abs(coefficients)
        phases = np.angle(coefficients)

        # filter out insignificant terms
        significant_indices = [i for i, mag in enumerate(magnitudes) if mag > threshold]
        filtered_magnitudes = [magnitudes[i] for i in significant_indices]
        filtered_phases = [phases[i] for i in significant_indices]
        filtered_operators = [operators[i] for i in significant_indices]

        # group up the insignificant terms
        other_magnitude = sum(
            magnitudes[i] for i in range(len(magnitudes)) if i not in significant_indices
        )
        insignificant_phases = [
            phases[i] for i in range(len(phases)) if i not in significant_indices
        ]
        other_magnitude = sum(
            magnitudes[i] for i in range(len(magnitudes)) if i not in significant_indices
        )

        # no phase equivalent to 0
        other_phase = np.mean(insignificant_phases) if insignificant_phases else 0

        if other_magnitude > 0:
            filtered_magnitudes.append(other_magnitude)
            filtered_phases.append(other_phase)
            filtered_operators.append('Other')

        # limit the displayed terms
        if len(filtered_operators) > max_terms:
            indices = np.linspace(0, len(filtered_operators) - 1, max_terms, dtype=int)
            filtered_magnitudes = [filtered_magnitudes[i] for i in indices]
            filtered_phases = [filtered_phases[i] for i in indices]
            filtered_operators = [filtered_operators[i] for i in indices]

        # Create the figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # magnitude plot
        indices = np.arange(len(filtered_operators))
        ax1.bar(indices, filtered_magnitudes, color='green', alpha=0.7)
        ax1.set_title(f'{title} - Magnitude', fontsize=14)
        ax1.set_xlabel('Term Index', fontsize=12)
        ax1.set_ylabel('Coefficient Magnitude', fontsize=12)
        ax1.set_xticks(indices)
        ax1.set_xticklabels(filtered_operators, rotation=90, fontsize=8)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

        # phase plot
        ax2.bar(indices, filtered_phases, color='purple', alpha=0.7)
        ax2.set_title(f'{title} - Phase', fontsize=14)
        ax2.set_xlabel('Term Index', fontsize=12)
        ax2.set_ylabel('Coefficient Phase (radians)', fontsize=12)
        ax2.set_xticks(indices)
        ax2.set_xticklabels(filtered_operators, rotation=90, fontsize=8)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)

        # adjust layout and save
        plt.tight_layout()
        plot_path = savePlot(
            plt,
            settings.OPERATOR_CPLX_COEFFS_PLOT_DIR,
            settings.OPERATOR_CPLX_COEFFS_PLOT,
            symbols,
        )

        plt.close(fig)
        return plot_path
