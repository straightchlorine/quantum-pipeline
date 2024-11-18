from quantum_simulation.drivers.molecule_loader import load_molecule


def test_load_molecule():
    data = load_molecule("data/molecules.json")
    assert isinstance(data, list)
    assert "name" in data[0]
