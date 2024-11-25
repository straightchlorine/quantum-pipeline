"""
molecule_loader.py

This module contains utilities to load and validate molecular information
used in quantum simulations.
"""

import json
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.units import DistanceUnit


def validate_molecule_data(data: dict):
    """Validate data parsed from the JSON file.

    Args:
        data: The data parsed from the JSON file.

    Raises:
        ValueError: If the data is missing required fields.
    """
    required_fields = {'symbols', 'coords'}
    for molecule in data:
        if not required_fields.issubset(molecule.keys()):
            raise ValueError(f'Missing required fields in: {molecule}')


def load_molecule(file_path: str):
    """Load molecule data from a file and return MoleculeInfo objects.

    Args:
        file_path: The path to the file containing the molecule data.

    Returns:
        A list of MoleculeInfo objects.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
        validate_molecule_data(data)

        molecules = []
        for mol in data:
            molecules.append(
                MoleculeInfo(
                    symbols=mol['symbols'],
                    coords=mol['coords'],
                    multiplicity=mol.get('multiplicity', 1),
                    charge=mol.get('charge', 0),
                    units=DistanceUnit[mol.get('units', 'angstrom').upper()],
                    masses=mol.get('masses', None),
                )
            )
        return molecules
