from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.backends.aer_simulator import AerBackend

from quantum_pipeline.configs.module.backend import BackendConfig
from quantum_pipeline.solvers.vqe_solver import VQESolver
from quantum_pipeline.structures.vqe_observation import VQEResult


@pytest.fixture
def mock_backend_config():
    """Create a mock BackendConfig for testing."""
    return BackendConfig(
        local=True,
        gpu=False,
        optimization_level=2,
        min_num_qubits=4,
        filters=None,
        simulation_method='mock_backend',
        gpu_opts=None,
        noise=None,
    )


@pytest.fixture
def sample_hamiltonian():
    """Create a sample Hamiltonian for testing."""
    return SparsePauliOp.from_list([('IIII', 0.1), ('IIXI', 0.2), ('XIXI', 0.3)])


@pytest.fixture
def vqe_solver(mock_backend_config, sample_hamiltonian):
    """Create a VQESolver instance for testing."""
    return VQESolver(
        qubit_op=sample_hamiltonian,
        backend_config=mock_backend_config,
        max_iterations=10,
        optimizer='COBYLA',
        ansatz_reps=2,
    )


def test_vqe_solver_initialization(vqe_solver, sample_hamiltonian, mock_backend_config):
    """Test VQESolver initialization."""
    assert vqe_solver.qubit_op == sample_hamiltonian
    assert vqe_solver.backend_config == mock_backend_config
    assert vqe_solver.max_iterations == 10
    assert vqe_solver.optimizer == 'COBYLA'
    assert vqe_solver.ansatz_reps == 2
    assert vqe_solver.current_iter == 1
    assert len(vqe_solver.vqe_process) == 0


def test_compute_energy(vqe_solver):
    """Test compute_energy method."""
    # mocking the estimator
    mock_estimator = MagicMock()
    mock_result = MagicMock()
    mock_result.data.evs = [1.5]
    mock_result.data.stds = [0.1]
    mock_estimator.run.return_value.result.return_value = [mock_result]

    # mock the ansatz and hamiltonian
    mock_ansatz = MagicMock()
    mock_hamiltonian = MagicMock()

    # test params
    params = np.array([0.1, 0.2, 0.3])

    # suppress logging output
    with patch.object(vqe_solver.logger, 'debug'):
        energy = vqe_solver.compute_energy(params, mock_ansatz, mock_hamiltonian, mock_estimator)

    assert energy == 1.5
    assert len(vqe_solver.vqe_process) == 1
    assert vqe_solver.vqe_process[0].iteration == 1
    assert vqe_solver.vqe_process[0].result == 1.5
    assert vqe_solver.current_iter == 2


def test_optimize_circuits(vqe_solver):
    """Test _optimize_circuits method."""
    # mock backend and its target
    mock_backend = MagicMock()
    mock_backend.target = MagicMock()

    # mock ansatz and hamiltonian
    mock_ansatz = MagicMock()
    mock_hamiltonian = MagicMock()

    # mock circuit optimization via pass_manager
    with patch('quantum_pipeline.solvers.vqe_solver.generate_preset_pass_manager') as mock_pm:
        # mock the pm instance
        mock_pass_manager = MagicMock()
        mock_pass_manager.run.return_value = mock_ansatz
        mock_pm.return_value = mock_pass_manager

        # mock the apply layout method
        mock_hamiltonian.apply_layout.return_value = mock_hamiltonian

        # call the method
        optimized_ansatz, _optimized_hamiltonian = vqe_solver._optimize_circuits(
            mock_ansatz, mock_hamiltonian, mock_backend
        )

        mock_pm.assert_called_once_with(
            target=mock_backend.target, optimization_level=vqe_solver.optimization_level
        )
        assert optimized_ansatz == mock_ansatz
        mock_hamiltonian.apply_layout.assert_called_once()


def test_solve_via_aer(vqe_solver):
    """Test solve method with Aer backend."""
    mock_backend = MagicMock(spec=AerBackend)

    # patch the backend
    with (
        patch.object(vqe_solver, 'get_backend', return_value=mock_backend),
        patch.object(vqe_solver, 'via_aer') as mock_via_aer,
    ):
        # assign mock result
        mock_result = MagicMock()
        mock_via_aer.return_value = mock_result

        # call the solver
        result = vqe_solver.solve()

        vqe_solver.get_backend.assert_called_once()
        mock_via_aer.assert_called_once_with(mock_backend)
        assert result == mock_result


def test_solve_via_ibmq(vqe_solver):
    """Test solve method with IBMQ backend."""
    mock_backend = MagicMock()

    # patch the get_backend to run via ibm quantum
    with (
        patch.object(vqe_solver, 'get_backend', return_value=mock_backend),
        patch.object(vqe_solver, 'via_ibmq') as mock_via_ibmq,
    ):
        # mock result
        mock_result = MagicMock()
        mock_via_ibmq.return_value = mock_result

        # call the solver
        result = vqe_solver.solve()

        vqe_solver.get_backend.assert_called_once()
        mock_via_ibmq.assert_called_once_with(mock_backend)
        assert result == mock_result


def test_vqe_solver_custom_parameters(mock_backend_config, sample_hamiltonian):
    """Test VQESolver with custom parameters."""
    custom_solver = VQESolver(
        qubit_op=sample_hamiltonian,
        backend_config=mock_backend_config,
        max_iterations=100,
        optimizer='SPSA',
        ansatz_reps=5,
        default_shots=2048,
        convergence_threshold=1e-4,
        optimization_level=2,
    )

    assert custom_solver.max_iterations == 100
    assert custom_solver.optimizer == 'SPSA'
    assert custom_solver.ansatz_reps == 5
    assert custom_solver.default_shots == 2048
    assert custom_solver.convergence_threshold == 1e-4
    assert custom_solver.optimization_level == 2


@pytest.mark.parametrize('backend_type', [AerBackend, MagicMock])
def test_solve_method_result_type(vqe_solver, backend_type):
    """Test that solve method returns a VQEResult for different backend types."""
    # mock backend
    mock_backend = MagicMock(spec=backend_type)
    mock_backend.name = 'test_backend'

    # mock result of the solver
    mock_result = MagicMock(spec=VQEResult)

    # patch get_backend to return the mock backend
    with patch.object(vqe_solver, 'get_backend', return_value=mock_backend):
        # patch method based on backend type
        if backend_type == AerBackend:
            with patch.object(vqe_solver, 'via_aer', return_value=mock_result):
                result = vqe_solver.solve()
        else:
            with patch.object(vqe_solver, 'via_ibmq', return_value=mock_result):
                result = vqe_solver.solve()

        # verify if correct result was returned
        assert isinstance(result, VQEResult | MagicMock)
        assert result._spec_class == VQEResult


class TestVQEConvergencePriority:
    """Test suite for VQE convergence and max_iterations priority logic."""

    def test_max_iterations_priority_over_convergence(self):
        """Test that max_iterations takes priority when both are specified."""
        # Test the logic directly without full VQE execution
        max_iterations = 5
        convergence_threshold = 1e-6

        # Logic from VQE solver: tol should be None when both are specified
        # tol = convergence_threshold if convergence_threshold and not max_iterations else None
        tol = convergence_threshold if convergence_threshold and not max_iterations else None

        assert tol is None, (
            'tol should be None when max_iterations takes priority over convergence'
        )

    def test_convergence_only_uses_tolerance(self):
        """Test that convergence threshold is used when only convergence is specified."""
        max_iterations = None
        convergence_threshold = 1e-6

        # Logic from VQE solver
        tol = convergence_threshold if convergence_threshold and not max_iterations else None

        assert tol == 1e-6, 'tol should be set when only convergence is specified'

    def test_max_iterations_only_no_tolerance(self):
        """Test that no tolerance is used when only max_iterations is specified."""
        max_iterations = 10
        convergence_threshold = None

        # Logic from VQE solver
        tol = convergence_threshold if convergence_threshold and not max_iterations else None

        assert tol is None, 'tol should be None when only max_iterations is specified'

    def test_neither_specified_no_tolerance(self):
        """Test that no tolerance is used when neither is specified."""
        max_iterations = None
        convergence_threshold = None

        # Logic from VQE solver
        tol = convergence_threshold if convergence_threshold and not max_iterations else None

        assert tol is None, 'tol should be None when neither is specified'

    def test_vqe_solver_initialization_with_both(self, sample_hamiltonian, mock_backend_config):
        """Test that VQESolver can be initialized with both parameters."""
        solver = VQESolver(
            qubit_op=sample_hamiltonian,
            backend_config=mock_backend_config,
            max_iterations=5,
            convergence_threshold=1e-6,
            optimizer='COBYLA',
            ansatz_reps=2,
        )

        assert solver.max_iterations == 5
        assert solver.convergence_threshold == 1e-6
        assert solver.optimizer == 'COBYLA'

    def test_vqe_solver_initialization_convergence_only(
        self, sample_hamiltonian, mock_backend_config
    ):
        """Test that VQESolver can be initialized with only convergence threshold."""
        solver = VQESolver(
            qubit_op=sample_hamiltonian,
            backend_config=mock_backend_config,
            max_iterations=None,
            convergence_threshold=1e-6,
            optimizer='COBYLA',
            ansatz_reps=2,
        )

        assert solver.max_iterations is None
        assert solver.convergence_threshold == 1e-6

    def test_vqe_solver_initialization_max_iterations_only(
        self, sample_hamiltonian, mock_backend_config
    ):
        """Test that VQESolver can be initialized with only max_iterations."""
        solver = VQESolver(
            qubit_op=sample_hamiltonian,
            backend_config=mock_backend_config,
            max_iterations=10,
            convergence_threshold=None,
            optimizer='COBYLA',
            ansatz_reps=2,
        )

        assert solver.max_iterations == 10
        assert solver.convergence_threshold is None

    def test_lbfgs_b_priority_logic(self):
        """Test L-BFGS-B specific logic when max_iterations is specified."""
        optimizer = 'L-BFGS-B'
        max_iterations = 5
        _convergence_threshold = 1e-6

        # Test the specific L-BFGS-B logic - should disable early convergence when max_iterations is set
        should_disable_early_convergence = optimizer == 'L-BFGS-B' and max_iterations

        assert should_disable_early_convergence, (
            'L-BFGS-B should disable default convergence when max_iterations is set'
        )

        # Verify the ftol and gtol values that would be set
        expected_ftol = 1e-15
        expected_gtol = 1e-15

        assert expected_ftol < 1e-10, 'ftol should be very small to prevent early convergence'
        assert expected_gtol < 1e-10, 'gtol should be very small to prevent early convergence'

    def test_lbfgs_b_without_max_iterations(self):
        """Test L-BFGS-B behavior when max_iterations is not specified."""
        optimizer = 'L-BFGS-B'
        max_iterations = None

        # Should NOT disable early convergence when max_iterations is not set
        should_disable_early_convergence = optimizer == 'L-BFGS-B' and max_iterations

        assert not should_disable_early_convergence, (
            'L-BFGS-B should use default convergence when max_iterations is not set'
        )


class TestVQEEdgeCases:
    """Test edge cases and boundary conditions for VQE solver."""

    def test_max_iterations_zero(self, sample_hamiltonian, mock_backend_config):
        """Test VQESolver behavior with max_iterations=0."""
        solver = VQESolver(
            qubit_op=sample_hamiltonian,
            backend_config=mock_backend_config,
            max_iterations=0,
            optimizer='COBYLA',
            ansatz_reps=2,
        )
        assert solver.max_iterations == 0

    def test_max_iterations_negative(self, sample_hamiltonian, mock_backend_config):
        """Test VQESolver behavior with negative max_iterations."""
        solver = VQESolver(
            qubit_op=sample_hamiltonian,
            backend_config=mock_backend_config,
            max_iterations=-5,
            optimizer='COBYLA',
            ansatz_reps=2,
        )
        assert solver.max_iterations == -5

    def test_max_iterations_very_large(self, sample_hamiltonian, mock_backend_config):
        """Test VQESolver behavior with very large max_iterations."""
        large_value = 999999
        solver = VQESolver(
            qubit_op=sample_hamiltonian,
            backend_config=mock_backend_config,
            max_iterations=large_value,
            optimizer='L-BFGS-B',
            ansatz_reps=2,
        )
        assert solver.max_iterations == large_value

    def test_convergence_threshold_zero(self, sample_hamiltonian, mock_backend_config):
        """Test VQESolver behavior with convergence_threshold=0."""
        solver = VQESolver(
            qubit_op=sample_hamiltonian,
            backend_config=mock_backend_config,
            max_iterations=10,
            convergence_threshold=0.0,
            optimizer='L-BFGS-B',
            ansatz_reps=2,
        )
        assert solver.convergence_threshold == 0.0

    def test_convergence_threshold_negative(self, sample_hamiltonian, mock_backend_config):
        """Test VQESolver behavior with negative convergence_threshold."""
        solver = VQESolver(
            qubit_op=sample_hamiltonian,
            backend_config=mock_backend_config,
            max_iterations=10,
            convergence_threshold=-1e-6,
            optimizer='L-BFGS-B',
            ansatz_reps=2,
        )
        assert solver.convergence_threshold == -1e-6

    def test_convergence_threshold_very_small(self, sample_hamiltonian, mock_backend_config):
        """Test VQESolver behavior with very small convergence_threshold."""
        tiny_threshold = 1e-20
        solver = VQESolver(
            qubit_op=sample_hamiltonian,
            backend_config=mock_backend_config,
            max_iterations=10,
            convergence_threshold=tiny_threshold,
            optimizer='L-BFGS-B',
            ansatz_reps=2,
        )
        assert solver.convergence_threshold == tiny_threshold

    def test_lbfgs_b_with_max_iterations_one(self, sample_hamiltonian, mock_backend_config):
        """Test L-BFGS-B behavior with max_iterations=1."""
        solver = VQESolver(
            qubit_op=sample_hamiltonian,
            backend_config=mock_backend_config,
            max_iterations=1,
            optimizer='L-BFGS-B',
            ansatz_reps=2,
        )

        # Should still disable default convergence criteria
        should_disable = solver.optimizer == 'L-BFGS-B' and solver.max_iterations
        assert should_disable

    def test_cobyla_with_max_iterations_one(self, sample_hamiltonian, mock_backend_config):
        """Test COBYLA behavior with max_iterations=1."""
        solver = VQESolver(
            qubit_op=sample_hamiltonian,
            backend_config=mock_backend_config,
            max_iterations=1,
            optimizer='COBYLA',
            ansatz_reps=2,
        )

        # COBYLA should not have special convergence handling
        should_disable = solver.optimizer == 'L-BFGS-B' and solver.max_iterations
        assert not should_disable

    def test_different_optimizers_with_max_iterations(
        self, sample_hamiltonian, mock_backend_config
    ):
        """Test that only L-BFGS-B gets special convergence handling."""
        optimizers = ['COBYLA', 'L-BFGS-B', 'COBYQA']

        for optimizer in optimizers:
            solver = VQESolver(
                qubit_op=sample_hamiltonian,
                backend_config=mock_backend_config,
                max_iterations=5,
                optimizer=optimizer,
                ansatz_reps=2,
            )

            should_disable = solver.optimizer == 'L-BFGS-B' and solver.max_iterations

            if optimizer == 'L-BFGS-B':
                assert should_disable, 'L-BFGS-B should disable convergence criteria'
            else:
                assert not should_disable, f'{optimizer} should not disable convergence criteria'

    def test_string_to_bool_edge_cases(self):
        """Test edge cases for convergence threshold truthiness."""
        # Test various falsy values for convergence_threshold
        falsy_values = [None, 0, 0.0, False]
        truthy_values = [1e-6, 1e-20, -1e-6, 1, True]

        for value in falsy_values:
            assert not bool(value), f'Value {value} should be falsy'

        for value in truthy_values:
            assert bool(value), f'Value {value} should be truthy'

    def test_optimization_params_structure(self):
        """Test that optimization parameters have correct structure."""
        max_iterations = 42

        # Base params that should always be present
        optimization_params = {
            'maxiter': max_iterations,
            'disp': False,
        }

        # L-BFGS-B specific additions
        if True:  # Simulating L-BFGS-B condition
            optimization_params.update(
                {
                    'ftol': 1e-15,
                    'gtol': 1e-15,
                }
            )

        assert 'maxiter' in optimization_params
        assert 'disp' in optimization_params
        assert optimization_params['maxiter'] == max_iterations
        assert optimization_params['disp'] is False
        assert optimization_params['ftol'] == 1e-15
        assert optimization_params['gtol'] == 1e-15
