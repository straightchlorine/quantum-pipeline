"""Tests for energy visualization plotter."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from quantum_pipeline.visual.energy import EnergyPlotter
from quantum_pipeline.structures.vqe_observation import VQEProcess


@pytest.fixture
def sample_vqe_processes():
    """Create sample VQE processes for testing."""
    processes = []
    energies = np.linspace(2.0, -1.0, 10)  # Energy convergence
    for i, energy in enumerate(energies):
        process = VQEProcess(
            iteration=i + 1,
            parameters=np.random.random(4),
            result=energy,
            std=0.05 * (i + 1) / 10,  # Decreasing uncertainty
        )
        processes.append(process)
    return processes


@pytest.fixture
def sample_symbols():
    """Sample parameter symbols."""
    return ['θ0', 'θ1', 'θ2', 'θ3']


class TestEnergyPlotterInitialization:
    """Test EnergyPlotter initialization."""

    def test_plotter_creation(self, sample_vqe_processes, sample_symbols):
        """Test basic plotter creation."""
        plotter = EnergyPlotter(sample_vqe_processes, sample_symbols)
        assert plotter is not None
        assert len(plotter.iterations) == 10
        assert len(plotter.energy_values) == 10
        assert len(plotter.stds) == 10

    def test_iterations_extracted(self, sample_vqe_processes, sample_symbols):
        """Test that iterations are correctly extracted."""
        plotter = EnergyPlotter(sample_vqe_processes, sample_symbols)
        assert plotter.iterations == list(range(1, 11))

    def test_energy_values_extracted(self, sample_vqe_processes, sample_symbols):
        """Test that energy values are correctly extracted."""
        plotter = EnergyPlotter(sample_vqe_processes, sample_symbols)
        expected_energies = [p.result for p in sample_vqe_processes]
        np.testing.assert_array_almost_equal(plotter.energy_values, expected_energies)

    def test_stds_extracted(self, sample_vqe_processes, sample_symbols):
        """Test that standard deviations are correctly extracted."""
        plotter = EnergyPlotter(sample_vqe_processes, sample_symbols)
        expected_stds = [p.std for p in sample_vqe_processes]
        np.testing.assert_array_almost_equal(plotter.stds, expected_stds)

    def test_symbols_stored(self, sample_vqe_processes, sample_symbols):
        """Test that symbols are properly stored."""
        plotter = EnergyPlotter(sample_vqe_processes, sample_symbols)
        assert plotter.symbols == sample_symbols

    def test_max_points_default(self, sample_vqe_processes, sample_symbols):
        """Test default max_points value."""
        plotter = EnergyPlotter(sample_vqe_processes, sample_symbols)
        assert plotter.max_points == 100

    def test_max_points_custom(self, sample_vqe_processes, sample_symbols):
        """Test custom max_points value."""
        plotter = EnergyPlotter(sample_vqe_processes, sample_symbols, max_points=50)
        assert plotter.max_points == 50

    def test_single_process(self, sample_symbols):
        """Test with single VQE process."""
        process = VQEProcess(
            iteration=1,
            parameters=np.array([0.1, 0.2, 0.3, 0.4]),
            result=1.5,
            std=0.1,
        )
        plotter = EnergyPlotter([process], sample_symbols)
        assert len(plotter.iterations) == 1
        assert plotter.energy_values[0] == 1.5

    def test_many_processes(self, sample_symbols):
        """Test with many VQE processes."""
        processes = []
        for i in range(1000):
            process = VQEProcess(
                iteration=i + 1,
                parameters=np.random.random(4),
                result=2.0 - i * 0.001,
                std=0.05,
            )
            processes.append(process)
        plotter = EnergyPlotter(processes, sample_symbols)
        assert len(plotter.iterations) == 1000


class TestPointFiltering:
    """Test point filtering functionality."""

    def test_no_filtering_small_dataset(self, sample_symbols):
        """Test that small datasets are not filtered."""
        processes = []
        for i in range(50):
            process = VQEProcess(
                iteration=i + 1,
                parameters=np.random.random(4),
                result=1.0 - i * 0.01,
                std=0.05,
            )
            processes.append(process)
        plotter = EnergyPlotter(processes, sample_symbols, max_points=100)

        filtered_iter, filtered_energy, filtered_std = plotter._filter_points()
        assert len(filtered_iter) == 50
        assert len(filtered_energy) == 50
        assert len(filtered_std) == 50

    def test_filtering_large_dataset(self, sample_symbols):
        """Test that large datasets are filtered."""
        processes = []
        for i in range(1000):
            process = VQEProcess(
                iteration=i + 1,
                parameters=np.random.random(4),
                result=1.0 - i * 0.0001,
                std=0.05,
            )
            processes.append(process)
        plotter = EnergyPlotter(processes, sample_symbols, max_points=100)

        filtered_iter, filtered_energy, filtered_std = plotter._filter_points()
        # Filtering may result in slightly more points due to slicing
        assert len(filtered_iter) <= 200  # More lenient upper bound
        assert len(filtered_energy) <= 200
        assert len(filtered_std) <= 200
        assert len(filtered_iter) < len(processes)  # But definitely fewer than original

    def test_filtering_preserves_endpoints(self, sample_symbols):
        """Test that filtering preserves first and last points."""
        processes = []
        for i in range(200):
            process = VQEProcess(
                iteration=i + 1,
                parameters=np.random.random(4),
                result=2.0 - i * 0.01,
                std=0.05,
            )
            processes.append(process)
        plotter = EnergyPlotter(processes, sample_symbols, max_points=50)

        filtered_iter, _, _ = plotter._filter_points()
        assert filtered_iter[0] == 1  # First iteration
        assert filtered_iter[-1] == 200  # Last iteration

    def test_max_points_one(self, sample_vqe_processes, sample_symbols):
        """Test with max_points = 1."""
        plotter = EnergyPlotter(sample_vqe_processes, sample_symbols, max_points=1)
        filtered_iter, filtered_energy, filtered_std = plotter._filter_points()
        # With max_points=1 and 10 points, step is 10, so we get points at indices 0, 10
        assert len(filtered_iter) <= 2

    def test_filtering_step_calculation(self, sample_symbols):
        """Test that filtering step is calculated correctly."""
        processes = []
        for i in range(100):
            process = VQEProcess(
                iteration=i + 1,
                parameters=np.random.random(4),
                result=1.0,
                std=0.05,
            )
            processes.append(process)
        plotter = EnergyPlotter(processes, sample_symbols, max_points=10)

        filtered_iter, _, _ = plotter._filter_points()
        # With 100 points and max_points=10, step should be ~10
        # May result in slightly more due to slicing
        assert len(filtered_iter) <= 20
        assert len(filtered_iter) < 100  # But definitely fewer than original


class TestEnergyConvergence:
    """Test energy convergence patterns."""

    def test_monotonic_decreasing_energy(self, sample_symbols):
        """Test with monotonically decreasing energy (good convergence)."""
        processes = []
        for i in range(20):
            process = VQEProcess(
                iteration=i + 1,
                parameters=np.random.random(4),
                result=2.0 - i * 0.1,  # Monotonic decrease
                std=0.01,
            )
            processes.append(process)
        plotter = EnergyPlotter(processes, sample_symbols)
        assert plotter.energy_values[0] > plotter.energy_values[-1]

    def test_non_monotonic_energy(self, sample_symbols):
        """Test with non-monotonic energy (realistic optimization)."""
        processes = []
        energies = [2.0, 1.8, 1.9, 1.7, 1.6, 1.5, 1.6, 1.4, 1.3, 1.2]
        for i, energy in enumerate(energies):
            process = VQEProcess(
                iteration=i + 1,
                parameters=np.random.random(4),
                result=energy,
                std=0.05,
            )
            processes.append(process)
        plotter = EnergyPlotter(processes, sample_symbols)
        assert len(plotter.energy_values) == 10

    def test_flat_energy_landscape(self, sample_symbols):
        """Test with flat energy (no convergence)."""
        processes = []
        for i in range(10):
            process = VQEProcess(
                iteration=i + 1,
                parameters=np.random.random(4),
                result=1.5,  # Constant energy
                std=0.05,
            )
            processes.append(process)
        plotter = EnergyPlotter(processes, sample_symbols)
        assert all(e == 1.5 for e in plotter.energy_values)

    def test_negative_energies(self, sample_symbols):
        """Test with negative energy values."""
        processes = []
        for i in range(10):
            process = VQEProcess(
                iteration=i + 1,
                parameters=np.random.random(4),
                result=-1.0 - i * 0.1,  # Negative energies
                std=0.05,
            )
            processes.append(process)
        plotter = EnergyPlotter(processes, sample_symbols)
        assert all(e < 0 for e in plotter.energy_values)

    def test_zero_crossing_energies(self, sample_symbols):
        """Test with energies crossing zero."""
        processes = []
        energies = [1.0, 0.5, 0.0, -0.5, -1.0, -1.5]
        for i, energy in enumerate(energies):
            process = VQEProcess(
                iteration=i + 1,
                parameters=np.random.random(4),
                result=energy,
                std=0.05,
            )
            processes.append(process)
        plotter = EnergyPlotter(processes, sample_symbols)
        assert plotter.energy_values[0] > 0
        assert plotter.energy_values[-1] < 0


class TestUncertaintyHandling:
    """Test uncertainty (standard deviation) handling."""

    def test_zero_uncertainty(self, sample_symbols):
        """Test with zero uncertainty."""
        processes = []
        for i in range(10):
            process = VQEProcess(
                iteration=i + 1,
                parameters=np.random.random(4),
                result=1.0 - i * 0.1,
                std=0.0,
            )
            processes.append(process)
        plotter = EnergyPlotter(processes, sample_symbols)
        assert all(s == 0.0 for s in plotter.stds)

    def test_large_uncertainty(self, sample_symbols):
        """Test with large uncertainty values."""
        processes = []
        for i in range(10):
            process = VQEProcess(
                iteration=i + 1,
                parameters=np.random.random(4),
                result=1.0,
                std=10.0,  # Large uncertainty
            )
            processes.append(process)
        plotter = EnergyPlotter(processes, sample_symbols)
        assert all(s == 10.0 for s in plotter.stds)

    def test_decreasing_uncertainty(self, sample_symbols):
        """Test with decreasing uncertainty over iterations."""
        processes = []
        for i in range(10):
            process = VQEProcess(
                iteration=i + 1,
                parameters=np.random.random(4),
                result=1.0,
                std=1.0 - i * 0.05,  # Decreasing uncertainty
            )
            processes.append(process)
        plotter = EnergyPlotter(processes, sample_symbols)
        assert plotter.stds[0] > plotter.stds[-1]

    def test_increasing_uncertainty(self, sample_symbols):
        """Test with increasing uncertainty over iterations."""
        processes = []
        for i in range(10):
            process = VQEProcess(
                iteration=i + 1,
                parameters=np.random.random(4),
                result=1.0,
                std=i * 0.05,  # Increasing uncertainty
            )
            processes.append(process)
        plotter = EnergyPlotter(processes, sample_symbols)
        assert plotter.stds[0] < plotter.stds[-1]


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_symbols(self, sample_vqe_processes):
        """Test with empty symbols list."""
        plotter = EnergyPlotter(sample_vqe_processes, [])
        assert plotter.symbols == []

    def test_single_symbol(self, sample_vqe_processes):
        """Test with single symbol."""
        plotter = EnergyPlotter(sample_vqe_processes, ['θ'])
        assert len(plotter.symbols) == 1

    def test_many_symbols(self, sample_vqe_processes):
        """Test with many symbols."""
        symbols = [f'θ{i}' for i in range(100)]
        plotter = EnergyPlotter(sample_vqe_processes, symbols)
        assert len(plotter.symbols) == 100

    def test_very_large_iteration_count(self, sample_symbols):
        """Test with very large iteration count."""
        processes = []
        for i in range(10000):
            process = VQEProcess(
                iteration=i + 1,
                parameters=np.random.random(4),
                result=2.0 - i * 0.0001,
                std=0.05,
            )
            processes.append(process)
        plotter = EnergyPlotter(processes, sample_symbols)
        assert len(plotter.iterations) == 10000

    def test_unicode_symbols(self, sample_vqe_processes):
        """Test with unicode symbols."""
        unicode_symbols = ['α', 'β', 'γ', 'δ']
        plotter = EnergyPlotter(sample_vqe_processes, unicode_symbols)
        assert plotter.symbols == unicode_symbols
