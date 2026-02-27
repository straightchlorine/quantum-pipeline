"""Tests for Hartree-Fock based parameter initialization."""

import numpy as np
import pytest

from quantum_pipeline.solvers.hf_init import HFData, compute_hf_initial_parameters


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


class TestComputeHFInitialParameters:
    """Tests for compute_hf_initial_parameters."""

    @pytest.mark.parametrize(
        'n_qubits, num_particles, ansatz_reps, n_occupied',
        [
            (4, (1, 1), 2, 2),    # H2: 4 qubits, 2 electrons
            (12, (3, 3), 2, 6),   # LiH: 12 qubits, 6 electrons
            (10, (5, 5), 3, 10),  # NH3-like: 10 qubits, all occupied
        ],
    )
    def test_occupied_qubits_get_pi(self, n_qubits, num_particles, ansatz_reps, n_occupied):
        """Verify occupied qubits get Ry=π in layer 0."""
        hf_data = HFData(num_particles=num_particles, num_spatial_orbitals=n_qubits // 2)
        params = compute_hf_initial_parameters(n_qubits, hf_data, ansatz_reps)

        # Layer 0, Ry block: first n_occupied params should be π
        for i in range(n_occupied):
            assert params[i] == pytest.approx(np.pi), (
                f'Occupied qubit {i} should have Ry=π'
            )

    @pytest.mark.parametrize(
        'n_qubits, num_particles, ansatz_reps',
        [
            (4, (1, 1), 2),
            (12, (3, 3), 2),
        ],
    )
    def test_virtual_qubits_are_zero(self, n_qubits, num_particles, ansatz_reps):
        """Verify virtual (unoccupied) qubits get Ry=0 in layer 0."""
        hf_data = HFData(num_particles=num_particles, num_spatial_orbitals=n_qubits // 2)
        n_occupied = sum(num_particles)
        params = compute_hf_initial_parameters(n_qubits, hf_data, ansatz_reps)

        # Layer 0, Ry block: virtual qubits should be 0
        for i in range(n_occupied, n_qubits):
            assert params[i] == 0.0, f'Virtual qubit {i} should have Ry=0'

    def test_rz_and_upper_layers_are_zero(self):
        """All Rz params and all upper layer params should be 0."""
        n_qubits = 4
        ansatz_reps = 2
        hf_data = HFData(num_particles=(1, 1), num_spatial_orbitals=2)
        params = compute_hf_initial_parameters(n_qubits, hf_data, ansatz_reps)

        # Layer 0, Rz block (indices n_qubits to 2*n_qubits)
        rz_block = params[n_qubits:2 * n_qubits]
        np.testing.assert_array_equal(rz_block, 0.0)

        # All subsequent layers (indices >= 2*n_qubits)
        upper_layers = params[2 * n_qubits:]
        np.testing.assert_array_equal(upper_layers, 0.0)

    def test_total_parameter_count(self):
        """Verify total number of parameters matches EfficientSU2 layout."""
        n_qubits = 4
        ansatz_reps = 3
        hf_data = HFData(num_particles=(1, 1), num_spatial_orbitals=2)
        params = compute_hf_initial_parameters(n_qubits, hf_data, ansatz_reps)

        expected = n_qubits * 2 * (ansatz_reps + 1)
        assert len(params) == expected

    def test_determinism(self):
        """HF init should be deterministic (no randomness)."""
        hf_data = HFData(num_particles=(1, 1), num_spatial_orbitals=2)
        params_a = compute_hf_initial_parameters(4, hf_data, 2)
        params_b = compute_hf_initial_parameters(4, hf_data, 2)
        np.testing.assert_array_equal(params_a, params_b)

    def test_raises_when_occupied_exceeds_qubits(self):
        """Should raise ValueError when n_occupied > n_qubits."""
        hf_data = HFData(num_particles=(3, 3), num_spatial_orbitals=3)
        with pytest.raises(ValueError, match='exceeds'):
            compute_hf_initial_parameters(4, hf_data, 2)

    def test_h2_specific(self):
        """H2 molecule: 4 qubits, 2 electrons, 2 reps."""
        hf_data = HFData(num_particles=(1, 1), num_spatial_orbitals=2)
        params = compute_hf_initial_parameters(4, hf_data, 2)

        # 4 qubits * 2 gates * 3 layers = 24 params
        assert len(params) == 24

        # First 2 occupied get π, rest are 0
        expected_ry_layer0 = [np.pi, np.pi, 0.0, 0.0]
        np.testing.assert_array_almost_equal(params[:4], expected_ry_layer0)

        # Everything else is zero
        assert np.all(params[4:] == 0.0)

    def test_zero_occupied(self):
        """Edge case: no occupied orbitals."""
        hf_data = HFData(num_particles=(0, 0), num_spatial_orbitals=2)
        params = compute_hf_initial_parameters(4, hf_data, 2)
        np.testing.assert_array_equal(params, 0.0)
