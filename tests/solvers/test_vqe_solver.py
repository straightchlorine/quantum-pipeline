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
    """Test computeEnergy method."""
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
        energy = vqe_solver.computeEnergy(params, mock_ansatz, mock_hamiltonian, mock_estimator)

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
        optimized_ansatz, optimized_hamiltonian = vqe_solver._optimize_circuits(
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
        patch.object(vqe_solver, 'viaAer') as mock_via_aer,
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
        patch.object(vqe_solver, 'viaIBMQ') as mock_via_ibmq,
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
            with patch.object(vqe_solver, 'viaAer', return_value=mock_result):
                result = vqe_solver.solve()
        else:
            with patch.object(vqe_solver, 'viaIBMQ', return_value=mock_result):
                result = vqe_solver.solve()

        # verify if correct result was returned
        assert isinstance(result, VQEResult | MagicMock)
        assert result._spec_class == VQEResult
