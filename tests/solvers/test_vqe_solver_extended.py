"""Extended tests for VQE solver covering additional scenarios."""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp
from quantum_pipeline.configs.module.backend import BackendConfig
from quantum_pipeline.solvers.vqe_solver import VQESolver


@pytest.fixture
def backend_config():
    """Create a BackendConfig for testing."""
    return BackendConfig(
        local=True,
        gpu=False,
        optimization_level=2,
        min_num_qubits=4,
        filters=None,
        simulation_method='statevector',
        gpu_opts=None,
        noise=None,
    )


@pytest.fixture
def hamiltonian_2q():
    """2-qubit test Hamiltonian."""
    return SparsePauliOp.from_list([('II', 0.1), ('IX', 0.2), ('XX', 0.3)])


@pytest.fixture
def hamiltonian_4q():
    """4-qubit test Hamiltonian."""
    return SparsePauliOp.from_list([
        ('IIII', 0.1),
        ('IIXI', 0.2),
        ('XIXI', 0.3),
        ('XXII', 0.15),
    ])


@pytest.fixture
def hamiltonian_8q():
    """8-qubit test Hamiltonian for larger molecules."""
    return SparsePauliOp.from_list([
        ('I' * 8, 0.1),
        ('X' + 'I' * 7, 0.2),
        ('X' * 4 + 'I' * 4, 0.3),
    ])


class TestVQESolverInitialization:
    """Test VQESolver initialization with various parameters."""

    def test_default_parameters(self, hamiltonian_4q, backend_config):
        """Test VQESolver with default parameters."""
        solver = VQESolver(
            qubit_op=hamiltonian_4q,
            backend_config=backend_config,
        )
        assert solver.max_iterations == 50
        assert solver.optimizer == 'COBYLA'
        assert solver.ansatz_reps == 3
        assert solver.default_shots == 1024
        assert solver.current_iter == 1

    def test_custom_iterations(self, hamiltonian_4q, backend_config):
        """Test VQESolver with custom max iterations."""
        for max_iter in [1, 5, 100, 1000]:
            solver = VQESolver(
                qubit_op=hamiltonian_4q,
                backend_config=backend_config,
                max_iterations=max_iter,
            )
            assert solver.max_iterations == max_iter
            assert solver.digits_iter == len(str(max_iter))

    def test_custom_optimizer(self, hamiltonian_4q, backend_config):
        """Test VQESolver with different optimizers."""
        for opt in ['COBYLA', 'SPSA', 'Powell', 'CG']:
            solver = VQESolver(
                qubit_op=hamiltonian_4q,
                backend_config=backend_config,
                optimizer=opt,
            )
            assert solver.optimizer == opt

    def test_custom_ansatz_reps(self, hamiltonian_4q, backend_config):
        """Test VQESolver with different ansatz repetitions."""
        for reps in [1, 2, 5, 10]:
            solver = VQESolver(
                qubit_op=hamiltonian_4q,
                backend_config=backend_config,
                ansatz_reps=reps,
            )
            assert solver.ansatz_reps == reps

    def test_custom_shots(self, hamiltonian_4q, backend_config):
        """Test VQESolver with different shot counts."""
        for shots in [100, 512, 1024, 4096, 8192]:
            solver = VQESolver(
                qubit_op=hamiltonian_4q,
                backend_config=backend_config,
                default_shots=shots,
            )
            assert solver.default_shots == shots

    def test_convergence_threshold_setting(self, hamiltonian_4q, backend_config):
        """Test VQESolver with convergence threshold."""
        solver = VQESolver(
            qubit_op=hamiltonian_4q,
            backend_config=backend_config,
            convergence_threshold=1e-6,
        )
        assert solver.convergence_threshold == 1e-6

    def test_optimization_level_setting(self, hamiltonian_4q, backend_config):
        """Test VQESolver with different optimization levels."""
        for level in [0, 1, 2, 3]:
            solver = VQESolver(
                qubit_op=hamiltonian_4q,
                backend_config=backend_config,
                optimization_level=level,
            )
            assert solver.optimization_level == level


class TestVQESolverComputeEnergy:
    """Test computeEnergy method with various scenarios."""

    @pytest.fixture
    def solver(self, hamiltonian_4q, backend_config):
        """Create a VQESolver for energy computation tests."""
        return VQESolver(
            qubit_op=hamiltonian_4q,
            backend_config=backend_config,
            max_iterations=10,
        )

    def test_single_energy_computation(self, solver):
        """Test computing single energy value."""
        mock_estimator = MagicMock()
        mock_result = MagicMock()
        mock_result.data.evs = [1.5]
        mock_result.data.stds = [0.1]
        mock_estimator.run.return_value.result.return_value = [mock_result]

        params = np.array([0.1, 0.2, 0.3, 0.4])
        with patch.object(solver.logger, 'debug'):
            energy = solver.computeEnergy(params, MagicMock(), MagicMock(), mock_estimator)

        assert energy == 1.5
        assert len(solver.vqe_process) == 1
        assert solver.current_iter == 2

    def test_sequential_energy_computations(self, solver):
        """Test multiple sequential energy computations."""
        mock_estimator = MagicMock()
        energies = [1.5, 1.4, 1.2, 1.0, 0.9]

        for energy_val in energies:
            mock_result = MagicMock()
            mock_result.data.evs = [energy_val]
            mock_result.data.stds = [0.05]
            mock_estimator.run.return_value.result.return_value = [mock_result]

            params = np.random.random(4)
            with patch.object(solver.logger, 'debug'):
                energy = solver.computeEnergy(params, MagicMock(), MagicMock(), mock_estimator)

            assert energy == energy_val

        assert len(solver.vqe_process) == len(energies)
        assert solver.current_iter == len(energies) + 1

    def test_energy_with_zero_std(self, solver):
        """Test energy computation with zero standard deviation."""
        mock_estimator = MagicMock()
        mock_result = MagicMock()
        mock_result.data.evs = [0.5]
        mock_result.data.stds = [0.0]
        mock_estimator.run.return_value.result.return_value = [mock_result]

        params = np.array([0.0] * 4)
        with patch.object(solver.logger, 'debug'):
            energy = solver.computeEnergy(params, MagicMock(), MagicMock(), mock_estimator)

        assert energy == 0.5
        assert solver.vqe_process[0].std == 0.0

    def test_energy_with_large_std(self, solver):
        """Test energy computation with large standard deviation."""
        mock_estimator = MagicMock()
        mock_result = MagicMock()
        mock_result.data.evs = [1.5]
        mock_result.data.stds = [0.5]
        mock_estimator.run.return_value.result.return_value = [mock_result]

        params = np.random.random(4)
        with patch.object(solver.logger, 'debug'):
            energy = solver.computeEnergy(params, MagicMock(), MagicMock(), mock_estimator)

        assert energy == 1.5
        assert solver.vqe_process[0].std == 0.5

    def test_energy_with_negative_value(self, solver):
        """Test energy computation with negative energy values."""
        mock_estimator = MagicMock()
        mock_result = MagicMock()
        mock_result.data.evs = [-1.5]
        mock_result.data.stds = [0.1]
        mock_estimator.run.return_value.result.return_value = [mock_result]

        params = np.array([0.1, 0.2, 0.3, 0.4])
        with patch.object(solver.logger, 'debug'):
            energy = solver.computeEnergy(params, MagicMock(), MagicMock(), mock_estimator)

        assert energy == -1.5

    def test_parameter_storage(self, solver):
        """Test that parameters are stored correctly."""
        mock_estimator = MagicMock()
        mock_result = MagicMock()
        mock_result.data.evs = [1.0]
        mock_result.data.stds = [0.1]
        mock_estimator.run.return_value.result.return_value = [mock_result]

        test_params = np.array([0.1, 0.2, 0.3, 0.4])
        with patch.object(solver.logger, 'debug'):
            solver.computeEnergy(test_params, MagicMock(), MagicMock(), mock_estimator)

        stored_params = solver.vqe_process[0].parameters
        np.testing.assert_array_almost_equal(stored_params, test_params)

    def test_iteration_counter_increment(self, solver):
        """Test that iteration counter increments correctly."""
        mock_estimator = MagicMock()
        mock_result = MagicMock()
        mock_result.data.evs = [1.0]
        mock_result.data.stds = [0.1]
        mock_estimator.run.return_value.result.return_value = [mock_result]

        assert solver.current_iter == 1
        for i in range(5):
            with patch.object(solver.logger, 'debug'):
                solver.computeEnergy(np.random.random(4), MagicMock(), MagicMock(), mock_estimator)
            assert solver.current_iter == i + 2

    def test_vqe_process_tracking(self, solver):
        """Test VQEProcess tracking throughout optimization."""
        mock_estimator = MagicMock()
        energies = [2.0, 1.5, 1.0]

        for idx, energy_val in enumerate(energies):
            mock_result = MagicMock()
            mock_result.data.evs = [energy_val]
            mock_result.data.stds = [0.05]
            mock_estimator.run.return_value.result.return_value = [mock_result]

            with patch.object(solver.logger, 'debug'):
                solver.computeEnergy(np.random.random(4), MagicMock(), MagicMock(), mock_estimator)

        assert len(solver.vqe_process) == 3
        for idx, process in enumerate(solver.vqe_process):
            assert process.iteration == idx + 1
            assert process.result == energies[idx]


class TestVQESolverDifferentHamiltonians:
    """Test VQESolver with different Hamiltonian sizes."""

    def test_2qubit_hamiltonian(self, hamiltonian_2q, backend_config):
        """Test VQESolver with 2-qubit Hamiltonian."""
        solver = VQESolver(
            qubit_op=hamiltonian_2q,
            backend_config=backend_config,
        )
        assert solver.qubit_op.num_qubits == 2

    def test_4qubit_hamiltonian(self, hamiltonian_4q, backend_config):
        """Test VQESolver with 4-qubit Hamiltonian."""
        solver = VQESolver(
            qubit_op=hamiltonian_4q,
            backend_config=backend_config,
        )
        assert solver.qubit_op.num_qubits == 4

    def test_8qubit_hamiltonian(self, hamiltonian_8q, backend_config):
        """Test VQESolver with 8-qubit Hamiltonian."""
        solver = VQESolver(
            qubit_op=hamiltonian_8q,
            backend_config=backend_config,
        )
        assert solver.qubit_op.num_qubits == 8

    def test_solver_scales_with_hamiltonian_size(self, backend_config):
        """Test that solver can handle various Hamiltonian sizes."""
        for num_qubits in [2, 4, 8, 12]:
            # Create a simple Hamiltonian with identity and X on first qubit
            terms = [(f'X' + 'I' * (num_qubits - 1), 0.5)]
            ham = SparsePauliOp.from_list(terms)

            solver = VQESolver(
                qubit_op=ham,
                backend_config=backend_config,
            )
            assert solver.qubit_op.num_qubits == num_qubits


class TestVQESolverEdgeCases:
    """Test VQESolver with edge cases and boundary conditions."""

    def test_single_iteration_solver(self, hamiltonian_4q, backend_config):
        """Test solver with only 1 iteration."""
        solver = VQESolver(
            qubit_op=hamiltonian_4q,
            backend_config=backend_config,
            max_iterations=1,
        )
        assert solver.max_iterations == 1
        assert solver.digits_iter == 1

    def test_very_large_iteration_count(self, hamiltonian_4q, backend_config):
        """Test solver with very large iteration count."""
        solver = VQESolver(
            qubit_op=hamiltonian_4q,
            backend_config=backend_config,
            max_iterations=100000,
        )
        assert solver.max_iterations == 100000
        assert solver.digits_iter == 6

    def test_zero_ansatz_reps(self, hamiltonian_4q, backend_config):
        """Test solver with zero ansatz repetitions (minimal circuit)."""
        solver = VQESolver(
            qubit_op=hamiltonian_4q,
            backend_config=backend_config,
            ansatz_reps=0,
        )
        assert solver.ansatz_reps == 0

    def test_very_high_ansatz_reps(self, hamiltonian_4q, backend_config):
        """Test solver with very deep ansatz."""
        solver = VQESolver(
            qubit_op=hamiltonian_4q,
            backend_config=backend_config,
            ansatz_reps=50,
        )
        assert solver.ansatz_reps == 50

    def test_very_small_threshold(self, hamiltonian_4q, backend_config):
        """Test solver with very small convergence threshold."""
        solver = VQESolver(
            qubit_op=hamiltonian_4q,
            backend_config=backend_config,
            convergence_threshold=1e-12,
        )
        assert solver.convergence_threshold == 1e-12

    def test_very_large_threshold(self, hamiltonian_4q, backend_config):
        """Test solver with large convergence threshold."""
        solver = VQESolver(
            qubit_op=hamiltonian_4q,
            backend_config=backend_config,
            convergence_threshold=10.0,
        )
        assert solver.convergence_threshold == 10.0


class TestVQESolverConfiguration:
    """Test VQESolver configuration propagation."""

    def test_backend_config_propagation(self, hamiltonian_4q, backend_config):
        """Test that backend config is properly stored."""
        solver = VQESolver(
            qubit_op=hamiltonian_4q,
            backend_config=backend_config,
        )
        assert solver.backend_config == backend_config
        assert solver.backend_config.local is True
        assert solver.backend_config.gpu is False

    def test_multiple_solver_instances(self, hamiltonian_4q, backend_config):
        """Test that multiple solver instances are independent."""
        solver1 = VQESolver(
            qubit_op=hamiltonian_4q,
            backend_config=backend_config,
            max_iterations=10,
        )
        solver2 = VQESolver(
            qubit_op=hamiltonian_4q,
            backend_config=backend_config,
            max_iterations=20,
        )

        assert solver1.max_iterations == 10
        assert solver2.max_iterations == 20
        assert solver1 is not solver2
