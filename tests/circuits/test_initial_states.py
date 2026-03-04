"""Tests for Hartree-Fock initial state circuit building."""

import pytest
from qiskit.circuit import QuantumCircuit

from quantum_pipeline.circuits import HFData, build_hf_initial_state
from quantum_pipeline.mappers import JordanWignerMapper


@pytest.fixture
def mapper():
    return JordanWignerMapper()


class TestHFData:
    """Tests for the HFData dataclass."""

    def test_creation(self):
        hf = HFData(num_particles=(1, 1), num_spatial_orbitals=2)
        assert hf.num_particles == (1, 1)
        assert hf.num_spatial_orbitals == 2
        assert hf.reference_energy is None

    def test_with_reference_energy(self):
        hf = HFData(num_particles=(2, 2), num_spatial_orbitals=6, reference_energy=-7.8633)
        assert hf.reference_energy == -7.8633


class TestBuildHFInitialState:
    """Tests for build_hf_initial_state using qiskit-nature HartreeFock circuit."""

    def test_returns_quantum_circuit(self, mapper):
        hf_data = HFData(num_particles=(1, 1), num_spatial_orbitals=2)
        circuit = build_hf_initial_state(hf_data, mapper)
        assert isinstance(circuit, QuantumCircuit)

    def test_h2_circuit_num_qubits(self, mapper):
        """H2: 2 spatial orbitals -> 4 qubits under JW."""
        hf_data = HFData(num_particles=(1, 1), num_spatial_orbitals=2)
        circuit = build_hf_initial_state(hf_data, mapper)
        assert circuit.num_qubits == 4

    def test_lih_circuit_num_qubits(self, mapper):
        """LiH: 6 spatial orbitals -> 12 qubits under JW."""
        hf_data = HFData(num_particles=(2, 2), num_spatial_orbitals=6)
        circuit = build_hf_initial_state(hf_data, mapper)
        assert circuit.num_qubits == 12

    def test_circuit_deterministic(self, mapper):
        """Calling twice with same data produces identical circuits."""
        hf_data = HFData(num_particles=(1, 1), num_spatial_orbitals=2)
        circuit_a = build_hf_initial_state(hf_data, mapper)
        circuit_b = build_hf_initial_state(hf_data, mapper)
        assert circuit_a == circuit_b

    @pytest.mark.parametrize(
        'num_particles, num_spatial_orbitals, expected_qubits',
        [
            ((1, 1), 2, 4),
            ((2, 2), 6, 12),
            ((3, 3), 5, 10),
        ],
    )
    def test_qubit_count_matches_jw_mapping(
        self, mapper, num_particles, num_spatial_orbitals, expected_qubits
    ):
        """Qubit count should be 2 * num_spatial_orbitals (JW mapping)."""
        hf_data = HFData(
            num_particles=num_particles, num_spatial_orbitals=num_spatial_orbitals
        )
        circuit = build_hf_initial_state(hf_data, mapper)
        assert circuit.num_qubits == expected_qubits
