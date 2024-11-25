"""
operator_viewer.py

This module visualizes the coefficients of qubit operators.
"""

import matplotlib.pyplot as plt
import numpy as np


class OperatorViewer:
    """
    A utility class to visualize qubit operator coefficients.
    """

    @staticmethod
    def plot_operator_coefficients(qubit_op, title='Operator Coefficients'):
        """
        Visualizes the coefficients of the qubit operator as a heatmap.

        Args:
            qubit_op: Qiskit's PauliSumOp or similar operator.
            title: Title of the heatmap.
        """
        # Convert coefficients to a 1D array for visualization
        operators = [term[0] for term in qubit_op.to_list()]
        coefficients = [term[1] for term in qubit_op.to_list()]

        # Create the heatmap
        plt.figure(figsize=(12, 6))
        plt.bar(operators, coefficients)
        plt.xlabel('Term Index')
        plt.ylabel('Coefficient (Real Part)')
        plt.title(title)
        plt.grid()
        plt.show()
