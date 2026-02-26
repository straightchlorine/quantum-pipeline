import json
import os
import tempfile
import unittest

from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.units import DistanceUnit

from quantum_pipeline.drivers.molecule_loader import load_molecule, validate_molecule_data


class TestMoleculeLoader(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test method."""
        self.temp_dir = tempfile.mkdtemp()

    def create_temp_json_file(self, data):
        """Helper method to create a temporary JSON file with given data."""
        temp_file_path = os.path.join(self.temp_dir, 'molecules.json')
        with open(temp_file_path, 'w') as f:
            json.dump(data, f)
        return temp_file_path

    def test_validate_molecule_data_valid_input(self):
        """Test validate_molecule_data with a valid input."""
        valid_data = [
            {'symbols': ['H', 'H'], 'coords': [[0, 0, 0], [1, 1, 1]]},
            {'symbols': ['O'], 'coords': [[0, 0, 0]], 'multiplicity': 1, 'charge': 0},
        ]
        try:
            validate_molecule_data(valid_data)
        except ValueError:
            self.fail('validate_molecule_data raised ValueError unexpectedly!')

    def test_validate_molecule_data_missing_symbols(self):
        """Test validate_molecule_data when symbols are missing."""
        invalid_data = [{'coords': [[0, 0, 0]]}]
        with self.assertRaises(ValueError, msg='Missing symbols should raise ValueError'):
            validate_molecule_data(invalid_data)

    def test_validate_molecule_data_missing_coords(self):
        """Test validate_molecule_data when coordinates are missing."""
        invalid_data = [{'symbols': ['H']}]
        with self.assertRaises(ValueError, msg='Missing coords should raise ValueError'):
            validate_molecule_data(invalid_data)

    def test_load_molecule_complete_data(self):
        """Test load_molecule with complete molecular data."""
        complete_data = [
            {
                'symbols': ['H', 'H'],
                'coords': [[0, 0, 0], [1, 1, 1]],
                'multiplicity': 1,
                'charge': 0,
                'units': 'angstrom',
            }
        ]
        temp_file = self.create_temp_json_file(complete_data)

        molecules = load_molecule(temp_file)

        self.assertEqual(len(molecules), 1)
        molecule = molecules[0]

        self.assertIsInstance(molecule, MoleculeInfo)
        self.assertEqual(molecule.symbols, ['H', 'H'])
        self.assertEqual(molecule.multiplicity, 1)
        self.assertEqual(molecule.charge, 0)
        self.assertEqual(molecule.units, DistanceUnit.ANGSTROM)

    def test_load_molecule_default_values(self):
        """Test load_molecule with minimal data to ensure default values."""
        minimal_data = [{'symbols': ['O'], 'coords': [[0, 0, 0]]}]
        temp_file = self.create_temp_json_file(minimal_data)

        molecules = load_molecule(temp_file)

        self.assertEqual(len(molecules), 1)
        molecule = molecules[0]

        self.assertEqual(molecule.multiplicity, 1)
        self.assertEqual(molecule.charge, 0)
        self.assertEqual(molecule.units, DistanceUnit.ANGSTROM)
        self.assertIsNone(molecule.masses)

    def test_load_molecule_non_default_units(self):
        """Test load_molecule with non-default distance units."""
        data_with_units = [
            {'symbols': ['H', 'Cl'], 'coords': [[0, 0, 0], [2, 2, 2]], 'units': 'bohr'}
        ]
        temp_file = self.create_temp_json_file(data_with_units)

        molecules = load_molecule(temp_file)

        self.assertEqual(molecules[0].units, DistanceUnit.BOHR)

    def test_load_molecule_file_not_found(self):
        """Test load_molecule with a non-existent file."""
        with self.assertRaises(FileNotFoundError):
            load_molecule('/path/to/nonexistent/file.json')

    def test_load_molecule_invalid_json(self):
        """Test load_molecule with an invalid JSON file."""
        invalid_json_file = os.path.join(self.temp_dir, 'invalid.json')
        with open(invalid_json_file, 'w') as f:
            f.write('This is not a valid JSON')

        with self.assertRaises(json.JSONDecodeError):
            load_molecule(invalid_json_file)
