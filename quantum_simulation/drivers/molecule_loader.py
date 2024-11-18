import json
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.units import DistanceUnit


def load_molecule(file_path: str):
    """Load molecule data from a JSON file and return MoleculeInfo objects."""
    with open(file_path, "r") as file:
        data = json.load(file)
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
