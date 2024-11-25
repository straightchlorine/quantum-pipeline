"""
basis_sets.py

This module manages and validates basis sets for molecular simulations.
"""

SUPPORTED_BASIS_SETS = {'sto3g', '6-31g', 'cc-pvdz'}


def validate_basis_set(basis_set):
    """
    Validates that the provided basis set is supported.

    Args:
        basis_set: The name of the basis set to validate.

    Raises:
        ValueError: If the basis set is unsupported.
    """
    if basis_set not in SUPPORTED_BASIS_SETS:
        raise ValueError(
            f'Unsupported basis set: {basis_set}. Supported options: {SUPPORTED_BASIS_SETS}'
        )
