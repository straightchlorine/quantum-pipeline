import json
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.units import DistanceUnit


def validate_molecule_data(data: dict):
    """Validate data parsed from the JSON file."""
    required_fields = {"symbols", "coords"}
    for molecule in data:
        if not required_fields.issubset(molecule.keys()):
            raise ValueError(f"Missing required fields in: {molecule}")


def load_molecule(file_path: str):
    """Load molecule data from a file and return MoleculeInfo objects."""
    with open(file_path, "r") as file:
        data = json.load(file)
        validate_molecule_data(data)

        molecules = []
        for mol in data:
            molecules.append(
                MoleculeInfo(
                    symbols=mol["symbols"],
                    coords=mol["coords"],
                    multiplicity=mol.get("multiplicity", 1),
                    charge=mol.get("charge", 0),
                    units=DistanceUnit[mol.get("units", "angstrom").upper()],
                    masses=mol.get("masses", None),
                )
            )
        return molecules
