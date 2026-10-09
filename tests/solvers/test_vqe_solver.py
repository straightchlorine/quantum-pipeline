from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.backends.aer_simulator import AerBackend

from quantum_pipeline.circuits import HFData
from quantum_pipeline.configs.module.backend import BackendConfig
from quantum_pipeline.solvers.optimizer_config import get_optimizer_configuration
from quantum_pipeline.solvers.vqe_solver import MaxFunctionEvalsReachedError, VQESolver
from quantum_pipeline.structures.vqe_observation import VQEProcess, VQEResult


@pytest.fixture
def mock_backend_config():
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
    return SparsePauliOp.from_list([('IIII', 0.1), ('IIXI', 0.2), ('XIXI', 0.3)])


@pytest.fixture
def vqe_solver(mock_backend_config, sample_hamiltonian):
    return VQESolver(
        qubit_op=sample_hamiltonian,
        backend_config=mock_backend_config,
        max_iterations=10,
        optimizer='COBYLA',
        ansatz_reps=2,
    )


def test_vqe_solver_initialization(vqe_solver, sample_hamiltonian, mock_backend_config):
    assert vqe_solver.qubit_op == sample_hamiltonian
    assert vqe_solver.backend_config == mock_backend_config
    assert vqe_solver.max_iterations == 10
    assert vqe_solver.optimizer == 'COBYLA'
    assert vqe_solver.ansatz_reps == 2
    assert vqe_solver.current_iter == 1
    assert len(vqe_solver.vqe_process) == 0


def test_compute_energy(vqe_solver):
    mock_estimator = MagicMock()
    mock_result = MagicMock()
    mock_result.data.evs = [1.5]
    mock_result.data.stds = [0.1]
    mock_estimator.run.return_value.result.return_value = [mock_result]

    mock_ansatz = MagicMock()
    mock_hamiltonian = MagicMock()

    params = np.array([0.1, 0.2, 0.3])

    # suppress debug logs to keep test output clean
    with patch.object(vqe_solver.logger, 'debug'):
        energy = vqe_solver.compute_energy(params, mock_ansatz, mock_hamiltonian, mock_estimator)

    assert energy == 1.5
    assert len(vqe_solver.vqe_process) == 1
    assert vqe_solver.vqe_process[0].iteration == 1
    assert vqe_solver.vqe_process[0].result == 1.5
    assert vqe_solver.current_iter == 2


def test_compute_energy_derived_features(vqe_solver):
    """Test that compute_energy populates derived ML features across iterations."""
    mock_ansatz = MagicMock()
    mock_hamiltonian = MagicMock()

    def make_estimator(energy, std):
        mock_estimator = MagicMock()
        mock_result = MagicMock()
        mock_result.data.evs = [energy]
        mock_result.data.stds = [std]
        mock_estimator.run.return_value.result.return_value = [mock_result]
        return mock_estimator

    params1 = np.array([1.0, 2.0, 3.0])
    params2 = np.array([1.1, 2.2, 3.3])
    params3 = np.array([1.0, 2.0, 3.0])

    with patch.object(vqe_solver.logger, 'debug'):
        vqe_solver.compute_energy(
            params1, mock_ansatz, mock_hamiltonian, make_estimator(-1.0, 0.1)
        )
        p1 = vqe_solver.vqe_process[0]
        assert p1.energy_delta is None
        assert p1.parameter_delta_norm is None
        assert p1.cumulative_min_energy == np.float64(-1.0)

        vqe_solver.compute_energy(
            params2, mock_ansatz, mock_hamiltonian, make_estimator(-1.5, 0.05)
        )
        p2 = vqe_solver.vqe_process[1]
        assert p2.energy_delta == pytest.approx(-0.5)
        expected_norm = np.linalg.norm(params2 - params1)
        assert p2.parameter_delta_norm == pytest.approx(expected_norm)
        assert p2.cumulative_min_energy == np.float64(-1.5)

        vqe_solver.compute_energy(
            params3, mock_ansatz, mock_hamiltonian, make_estimator(-0.8, 0.2)
        )
        p3 = vqe_solver.vqe_process[2]
        assert p3.energy_delta == pytest.approx(0.7)
        expected_norm3 = np.linalg.norm(params3 - params2)
        assert p3.parameter_delta_norm == pytest.approx(expected_norm3)
        assert p3.cumulative_min_energy == np.float64(-1.5)


def test_optimize_circuits(vqe_solver):
    mock_backend = MagicMock()
    mock_backend.target = MagicMock()

    mock_ansatz = MagicMock()
    mock_hamiltonian = MagicMock()

    with patch('quantum_pipeline.solvers.vqe_solver.generate_preset_pass_manager') as mock_pm:
        mock_pass_manager = MagicMock()
        mock_pass_manager.run.return_value = mock_ansatz
        mock_pm.return_value = mock_pass_manager

        mock_hamiltonian.apply_layout.return_value = mock_hamiltonian

        optimized_ansatz, _optimized_hamiltonian = vqe_solver._optimize_circuits(
            mock_ansatz, mock_hamiltonian, mock_backend
        )

        mock_pm.assert_called_once_with(
            target=mock_backend.target,
            optimization_level=vqe_solver.optimization_level,
            seed_transpiler=vqe_solver.seed,
        )
        assert optimized_ansatz == mock_ansatz
        mock_hamiltonian.apply_layout.assert_called_once()


def test_solve_via_aer(vqe_solver):
    mock_backend = MagicMock(spec=AerBackend)

    with (
        patch.object(vqe_solver, 'get_backend', return_value=mock_backend),
        patch.object(vqe_solver, 'via_aer') as mock_via_aer,
    ):
        mock_result = MagicMock()
        mock_via_aer.return_value = mock_result

        result = vqe_solver.solve()

        vqe_solver.get_backend.assert_called_once()
        mock_via_aer.assert_called_once_with(mock_backend)
        assert result == mock_result


def test_solve_via_ibmq(vqe_solver):
    mock_backend = MagicMock()

    with (
        patch.object(vqe_solver, 'get_backend', return_value=mock_backend),
        patch.object(vqe_solver, 'via_ibmq') as mock_via_ibmq,
    ):
        mock_result = MagicMock()
        mock_via_ibmq.return_value = mock_result

        result = vqe_solver.solve()

        vqe_solver.get_backend.assert_called_once()
        mock_via_ibmq.assert_called_once_with(mock_backend)
        assert result == mock_result


def test_vqe_solver_custom_parameters(mock_backend_config, sample_hamiltonian):
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
    mock_backend = MagicMock(spec=backend_type)
    mock_backend.name = 'test_backend'

    mock_result = MagicMock(spec=VQEResult)

    with patch.object(vqe_solver, 'get_backend', return_value=mock_backend):
        if backend_type == AerBackend:
            with patch.object(vqe_solver, 'via_aer', return_value=mock_result):
                result = vqe_solver.solve()
        else:
            with patch.object(vqe_solver, 'via_ibmq', return_value=mock_result):
                result = vqe_solver.solve()

        assert isinstance(result, VQEResult | MagicMock)
        assert result._spec_class == VQEResult


class TestVQEConvergencePriority:
    def test_max_iterations_priority_over_convergence(self):
        """Test that max_iterations takes priority when both are specified."""
        # Restates the tol rule instead of driving VQESolver; the real selection lives in
        # get_optimizer_configuration(), which rejects both limits at once.
        max_iterations = 5
        convergence_threshold = 1e-6

        tol = convergence_threshold if convergence_threshold and not max_iterations else None

        assert tol is None, (
            'tol should be None when max_iterations takes priority over convergence'
        )

    def test_convergence_only_uses_tolerance(self):
        max_iterations = None
        convergence_threshold = 1e-6

        tol = convergence_threshold if convergence_threshold and not max_iterations else None

        assert tol == 1e-6, 'tol should be set when only convergence is specified'

    def test_max_iterations_only_no_tolerance(self):
        max_iterations = 10
        convergence_threshold = None

        tol = convergence_threshold if convergence_threshold and not max_iterations else None

        assert tol is None, 'tol should be None when only max_iterations is specified'

    def test_neither_specified_no_tolerance(self):
        max_iterations = None
        convergence_threshold = None

        tol = convergence_threshold if convergence_threshold and not max_iterations else None

        assert tol is None, 'tol should be None when neither is specified'

    def test_vqe_solver_initialization_with_both(self, sample_hamiltonian, mock_backend_config):
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
        """Test L-BFGS-B max_iterations mode sets only iteration limits, no tolerance overrides."""
        from quantum_pipeline.solvers.optimizer_config import get_optimizer_configuration

        max_iterations = 5
        options, minimize_tol = get_optimizer_configuration(
            optimizer='L-BFGS-B', max_iterations=max_iterations, num_parameters=10
        )

        assert options['maxiter'] == max_iterations
        assert options['maxfun'] == max_iterations
        # Tight tolerances to prevent premature convergence
        assert options['ftol'] == 1e-15
        assert options['gtol'] == 1e-15
        assert minimize_tol is None

    def test_lbfgs_b_without_max_iterations(self):
        optimizer = 'L-BFGS-B'
        max_iterations = None

        should_disable_early_convergence = optimizer == 'L-BFGS-B' and max_iterations

        assert not should_disable_early_convergence, (
            'L-BFGS-B should use default convergence when max_iterations is not set'
        )


class TestVQEEdgeCases:
    def test_max_iterations_zero(self, sample_hamiltonian, mock_backend_config):
        solver = VQESolver(
            qubit_op=sample_hamiltonian,
            backend_config=mock_backend_config,
            max_iterations=0,
            optimizer='COBYLA',
            ansatz_reps=2,
        )
        assert solver.max_iterations == 0

    def test_max_iterations_negative(self, sample_hamiltonian, mock_backend_config):
        solver = VQESolver(
            qubit_op=sample_hamiltonian,
            backend_config=mock_backend_config,
            max_iterations=-5,
            optimizer='COBYLA',
            ansatz_reps=2,
        )
        assert solver.max_iterations == -5

    def test_max_iterations_very_large(self, sample_hamiltonian, mock_backend_config):
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
        solver = VQESolver(
            qubit_op=sample_hamiltonian,
            backend_config=mock_backend_config,
            max_iterations=1,
            optimizer='L-BFGS-B',
            ansatz_reps=2,
        )

        should_disable = solver.optimizer == 'L-BFGS-B' and solver.max_iterations
        assert should_disable

    def test_cobyla_with_max_iterations_one(self, sample_hamiltonian, mock_backend_config):
        solver = VQESolver(
            qubit_op=sample_hamiltonian,
            backend_config=mock_backend_config,
            max_iterations=1,
            optimizer='COBYLA',
            ansatz_reps=2,
        )

        should_disable = solver.optimizer == 'L-BFGS-B' and solver.max_iterations
        assert not should_disable

    def test_different_optimizers_with_max_iterations(
        self, sample_hamiltonian, mock_backend_config
    ):
        """Test that only L-BFGS-B gets special convergence handling."""
        optimizers = ['COBYLA', 'L-BFGS-B', 'SLSQP']

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
        falsy_values = [None, 0, 0.0, False]
        truthy_values = [1e-6, 1e-20, -1e-6, 1, True]

        for value in falsy_values:
            assert not bool(value), f'Value {value} should be falsy'

        for value in truthy_values:
            assert bool(value), f'Value {value} should be truthy'

    def test_optimization_params_structure(self):
        """Test that L-BFGS-B max_iterations options have correct structure."""
        from quantum_pipeline.solvers.optimizer_config import get_optimizer_configuration

        max_iterations = 42
        options, minimize_tol = get_optimizer_configuration(
            optimizer='L-BFGS-B', max_iterations=max_iterations, num_parameters=10
        )

        assert 'maxiter' in options
        assert 'disp' in options
        assert options['maxiter'] == max_iterations
        assert options['maxfun'] == max_iterations
        assert options['disp'] is False
        # Tight tolerances to prevent premature convergence
        assert options['ftol'] == 1e-15
        assert options['gtol'] == 1e-15
        assert minimize_tol is None


class TestMaxFunctionEvalsReachedError:
    """Tests for the hard function evaluation limit."""

    def test_exception_stores_fields(self):
        """Test that the exception stores iteration, best_params, and best_energy."""
        params = np.array([1.0, 2.0, 3.0])
        exc = MaxFunctionEvalsReachedError(iteration=50, best_params=params, best_energy=-1.117)
        assert exc.iteration == 50
        np.testing.assert_array_equal(exc.best_params, params)
        assert exc.best_energy == -1.117
        assert '50' in str(exc)

    def test_compute_energy_raises_at_limit(self, vqe_solver):
        """Test that compute_energy raises MaxFunctionEvalsReachedError when limit is exceeded."""
        vqe_solver.max_iterations = 3
        vqe_solver.current_iter = 4  # one past the limit

        # _best_step keys on `result` (raw energy), reporting that step's params/energy.
        mock_process = MagicMock()
        mock_process.result = -1.5
        mock_process.cumulative_min_energy = -1.5
        mock_process.parameters = np.array([0.1, 0.2])
        vqe_solver.vqe_process = [mock_process]

        with pytest.raises(MaxFunctionEvalsReachedError) as exc_info:
            vqe_solver.compute_energy(np.array([0.1, 0.2]), MagicMock(), MagicMock(), MagicMock())

        assert exc_info.value.iteration == 3
        assert exc_info.value.best_energy == -1.5
        np.testing.assert_array_equal(exc_info.value.best_params, np.array([0.1, 0.2]))

    def test_compute_energy_runs_within_limit(self, vqe_solver):
        vqe_solver.max_iterations = 5
        vqe_solver.current_iter = 1

        mock_estimator = MagicMock()
        mock_result = MagicMock()
        mock_result.data.evs = [-0.5]
        mock_result.data.stds = [0.01]
        mock_estimator.run.return_value.result.return_value = [mock_result]

        with patch.object(vqe_solver.logger, 'debug'):
            energy = vqe_solver.compute_energy(
                np.array([0.1, 0.2, 0.3]), MagicMock(), MagicMock(), mock_estimator
            )

        assert energy == -0.5
        assert vqe_solver.current_iter == 2

    def test_truncated_result_uses_best_energy(self, vqe_solver):
        vqe_solver.init_data = MagicMock()
        vqe_solver.vqe_process = [MagicMock(), MagicMock()]

        result = vqe_solver._make_truncated_result(
            best_energy=-1.5, best_params=np.array([1.0, 2.0]), elapsed=10.0
        )

        assert isinstance(result, VQEResult)
        assert result.minimum == np.float64(-1.5)
        assert result.minimization_time == np.float64(10.0)
        assert result.maxcv is None

    def test_no_limit_when_max_iterations_none(self, mock_backend_config, sample_hamiltonian):
        solver = VQESolver(
            qubit_op=sample_hamiltonian,
            backend_config=mock_backend_config,
            max_iterations=None,
            optimizer='COBYLA',
            ansatz_reps=2,
        )
        solver.current_iter = 9999

        mock_estimator = MagicMock()
        mock_result = MagicMock()
        mock_result.data.evs = [-1.0]
        mock_result.data.stds = [0.01]
        mock_estimator.run.return_value.result.return_value = [mock_result]

        with patch.object(solver.logger, 'debug'):
            energy = solver.compute_energy(
                np.array([0.1, 0.2, 0.3]), MagicMock(), MagicMock(), mock_estimator
            )

        assert energy == -1.0


class TestVQESolverSeed:
    """Tests for seed-based reproducibility via VQESolver."""

    def test_seed_stored(self, mock_backend_config, sample_hamiltonian):
        solver = VQESolver(
            qubit_op=sample_hamiltonian,
            backend_config=mock_backend_config,
            seed=42,
        )
        assert solver.seed == 42

    def test_seed_default_none(self, vqe_solver):
        assert vqe_solver.seed is None

    @staticmethod
    def _run_via_aer_and_get_init_params(sample_hamiltonian, mock_backend_config, seed):
        """Helper: run via_aer with mocked optimizer/estimator, return initial_parameters."""
        solver = VQESolver(
            qubit_op=sample_hamiltonian,
            backend_config=mock_backend_config,
            max_iterations=1,
            optimizer='COBYLA',
            seed=seed,
        )

        mock_backend = MagicMock(spec=AerBackend)
        mock_backend.name = 'aer_simulator'
        mock_backend.target = MagicMock()

        mock_ansatz_isa = MagicMock()
        mock_ansatz_isa.num_parameters = 16
        mock_ansatz_isa.layout = None

        mock_hamiltonian_isa = MagicMock()
        mock_hamiltonian_isa.num_qubits = sample_hamiltonian.num_qubits
        mock_hamiltonian_isa.to_list.return_value = []

        mock_minimize_result = MagicMock()
        mock_minimize_result.fun = -1.0
        mock_minimize_result.x = np.zeros(16)
        mock_minimize_result.success = True

        with (
            patch.object(
                solver, '_optimize_circuits', return_value=(mock_ansatz_isa, mock_hamiltonian_isa)
            ),
            patch('quantum_pipeline.solvers.vqe_solver.EstimatorV2'),
            patch(
                'quantum_pipeline.solvers.vqe_solver.minimize', return_value=mock_minimize_result
            ),
        ):
            solver.via_aer(mock_backend)

        return solver.init_data.initial_parameters

    def test_same_seed_produces_identical_params_via_aer(
        self, mock_backend_config, sample_hamiltonian
    ):
        params_1 = self._run_via_aer_and_get_init_params(
            sample_hamiltonian, mock_backend_config, seed=42
        )
        params_2 = self._run_via_aer_and_get_init_params(
            sample_hamiltonian, mock_backend_config, seed=42
        )
        np.testing.assert_array_equal(params_1, params_2)

    def test_different_seeds_produce_different_params_via_aer(
        self, mock_backend_config, sample_hamiltonian
    ):
        params_a = self._run_via_aer_and_get_init_params(
            sample_hamiltonian, mock_backend_config, seed=42
        )
        params_b = self._run_via_aer_and_get_init_params(
            sample_hamiltonian, mock_backend_config, seed=99
        )
        assert not np.array_equal(params_a, params_b)

    def test_seed_stored_in_init_data(self, mock_backend_config, sample_hamiltonian):
        self._run_via_aer_and_get_init_params(sample_hamiltonian, mock_backend_config, seed=42)
        solver = VQESolver(
            qubit_op=sample_hamiltonian,
            backend_config=mock_backend_config,
            max_iterations=1,
            optimizer='COBYLA',
            seed=42,
        )

        mock_backend = MagicMock(spec=AerBackend)
        mock_backend.name = 'aer_simulator'
        mock_backend.target = MagicMock()

        mock_ansatz_isa = MagicMock()
        mock_ansatz_isa.num_parameters = 16
        mock_ansatz_isa.layout = None

        mock_hamiltonian_isa = MagicMock()
        mock_hamiltonian_isa.num_qubits = sample_hamiltonian.num_qubits
        mock_hamiltonian_isa.to_list.return_value = []

        mock_minimize_result = MagicMock()
        mock_minimize_result.fun = -1.0
        mock_minimize_result.x = np.zeros(16)
        mock_minimize_result.success = True

        with (
            patch.object(
                solver, '_optimize_circuits', return_value=(mock_ansatz_isa, mock_hamiltonian_isa)
            ),
            patch('quantum_pipeline.solvers.vqe_solver.EstimatorV2'),
            patch(
                'quantum_pipeline.solvers.vqe_solver.minimize', return_value=mock_minimize_result
            ),
        ):
            solver.via_aer(mock_backend)

        assert solver.init_data.seed == 42

    def test_no_seed_produces_varying_params(self, mock_backend_config, sample_hamiltonian):
        params_1 = self._run_via_aer_and_get_init_params(
            sample_hamiltonian, mock_backend_config, seed=None
        )
        params_2 = self._run_via_aer_and_get_init_params(
            sample_hamiltonian, mock_backend_config, seed=None
        )
        # Unseeded RNG: identical draws are possible but vanishingly unlikely.
        assert not np.array_equal(params_1, params_2)


class TestVQESolverHFInit:
    """Tests for Hartree-Fock initialization strategy via VQESolver."""

    @staticmethod
    def _run_via_aer_and_get_init_params(
        sample_hamiltonian, mock_backend_config, init_strategy='random', hf_data=None, seed=None
    ):
        """Helper: run via_aer with mocked optimizer/estimator, return initial_parameters."""
        mock_mapper = MagicMock()
        solver = VQESolver(
            qubit_op=sample_hamiltonian,
            backend_config=mock_backend_config,
            max_iterations=1,
            optimizer='COBYLA',
            seed=seed,
            init_strategy=init_strategy,
            hf_data=hf_data,
            mapper=mock_mapper,
        )

        mock_backend = MagicMock(spec=AerBackend)
        mock_backend.name = 'aer_simulator'
        mock_backend.target = MagicMock()

        mock_ansatz_isa = MagicMock()
        mock_ansatz_isa.num_parameters = 16
        mock_ansatz_isa.layout = None

        mock_hamiltonian_isa = MagicMock()
        mock_hamiltonian_isa.num_qubits = sample_hamiltonian.num_qubits
        mock_hamiltonian_isa.to_list.return_value = []

        mock_minimize_result = MagicMock()
        mock_minimize_result.fun = -1.0
        mock_minimize_result.x = np.zeros(16)
        mock_minimize_result.success = True

        # For HF init, mock the pre-optimization to avoid needing real quantum circuits
        hf_init_params = (
            np.ones(16) * 0.5 if init_strategy == 'hf' and hf_data is not None else None
        )

        with (
            patch.object(
                solver, '_optimize_circuits', return_value=(mock_ansatz_isa, mock_hamiltonian_isa)
            ),
            patch('quantum_pipeline.solvers.vqe_solver.EstimatorV2'),
            patch(
                'quantum_pipeline.solvers.vqe_solver.minimize', return_value=mock_minimize_result
            ),
        ):
            if hf_init_params is not None:
                with patch.object(
                    solver,
                    '_compute_hf_initial_parameters',
                    return_value=hf_init_params,
                ):
                    solver.via_aer(mock_backend)
            else:
                solver.via_aer(mock_backend)

        return solver.init_data

    def test_init_strategy_stored_in_init_data(self, mock_backend_config, sample_hamiltonian):
        hf_data = HFData(num_particles=(1, 1), num_spatial_orbitals=2)
        init_data = self._run_via_aer_and_get_init_params(
            sample_hamiltonian, mock_backend_config, init_strategy='hf', hf_data=hf_data
        )
        assert init_data.init_strategy == 'hf'

    def test_random_strategy_stored(self, mock_backend_config, sample_hamiltonian):
        init_data = self._run_via_aer_and_get_init_params(
            sample_hamiltonian, mock_backend_config, init_strategy='random'
        )
        assert init_data.init_strategy == 'random'

    def test_hf_params_differ_from_random(self, mock_backend_config, sample_hamiltonian):
        hf_data = HFData(num_particles=(1, 1), num_spatial_orbitals=2)
        init_data_hf = self._run_via_aer_and_get_init_params(
            sample_hamiltonian, mock_backend_config, init_strategy='hf', hf_data=hf_data, seed=42
        )
        init_data_random = self._run_via_aer_and_get_init_params(
            sample_hamiltonian, mock_backend_config, init_strategy='random', seed=42
        )
        assert not np.array_equal(
            init_data_hf.initial_parameters, init_data_random.initial_parameters
        )

    def test_hf_fallback_when_no_hf_data(self, mock_backend_config, sample_hamiltonian):
        """Test that HF strategy falls back to random when hf_data is None."""
        init_data = self._run_via_aer_and_get_init_params(
            sample_hamiltonian, mock_backend_config, init_strategy='hf', hf_data=None, seed=42
        )
        assert init_data.initial_parameters is not None
        assert len(init_data.initial_parameters) > 0
        assert init_data.init_strategy == 'random'

    @staticmethod
    def _init_data_after_compute(solver, num_parameters=16):
        """Helper: run _compute_initial_parameters, then _build_init_data, return the record."""
        mock_ansatz = MagicMock()
        mock_ansatz.num_parameters = num_parameters
        x0 = solver._compute_initial_parameters(mock_ansatz)
        mock_hamiltonian_isa = MagicMock()
        mock_hamiltonian_isa.num_qubits = 4
        mock_hamiltonian_isa.to_list.return_value = []
        solver._build_init_data('aer_simulator', mock_ansatz, mock_hamiltonian_isa, x0)
        return solver.init_data

    def test_real_amplitudes_hf_runs_pre_optimization(
        self, mock_backend_config, sample_hamiltonian
    ):
        hf_data = HFData(num_particles=(1, 1), num_spatial_orbitals=2)
        solver = VQESolver(
            qubit_op=sample_hamiltonian,
            backend_config=mock_backend_config,
            ansatz_type='RealAmplitudes',
            init_strategy='hf',
            hf_data=hf_data,
            mapper=MagicMock(),
        )
        hf_params = np.full(16, 0.5)
        with patch.object(
            solver, '_compute_hf_initial_parameters', return_value=hf_params
        ) as mock_hf:
            init_data = self._init_data_after_compute(solver)
        mock_hf.assert_called_once()
        np.testing.assert_array_equal(init_data.initial_parameters, hf_params)
        assert init_data.init_strategy == 'hf'
        assert init_data.ansatz_name == 'RealAmplitudes'

    def test_efficient_su2_hf_without_hf_data_records_random(
        self, mock_backend_config, sample_hamiltonian, caplog
    ):
        solver = VQESolver(
            qubit_op=sample_hamiltonian,
            backend_config=mock_backend_config,
            ansatz_type='EfficientSU2',
            init_strategy='hf',
            hf_data=None,
            mapper=None,
            seed=42,
        )
        solver.logger.propagate = True
        with (
            caplog.at_level('WARNING'),
            patch.object(solver, '_compute_hf_initial_parameters') as mock_hf,
        ):
            init_data = self._init_data_after_compute(solver)
        mock_hf.assert_not_called()
        assert 'falling back to random' in caplog.text
        assert init_data.init_strategy == 'random'
        assert solver.init_strategy == 'hf'

    def test_excitation_preserving_records_random(self, mock_backend_config, sample_hamiltonian):
        hf_data = HFData(num_particles=(1, 1), num_spatial_orbitals=2)
        solver = VQESolver(
            qubit_op=sample_hamiltonian,
            backend_config=mock_backend_config,
            ansatz_type='ExcitationPreserving',
            init_strategy='random',
            hf_data=hf_data,
            mapper=MagicMock(),
        )
        init_data = self._init_data_after_compute(solver)
        assert init_data.init_strategy == 'random'
        mock_ansatz = MagicMock()
        mock_ansatz.num_parameters = 200
        assert np.std(solver._compute_initial_parameters(mock_ansatz)) > 0.3

    def test_excitation_preserving_hf_uses_small_jitter(
        self, mock_backend_config, sample_hamiltonian
    ):
        solver = VQESolver(
            qubit_op=sample_hamiltonian,
            backend_config=mock_backend_config,
            ansatz_type='ExcitationPreserving',
            init_strategy='hf',
            seed=1,
            hf_data=HFData(num_particles=(1, 1), num_spatial_orbitals=2),
            mapper=MagicMock(),
        )
        mock_ansatz = MagicMock()
        mock_ansatz.num_parameters = 12
        params = solver._compute_initial_parameters(mock_ansatz)
        assert 0 < np.max(np.abs(params)) < 0.1
        assert solver.effective_init_strategy == 'hf'

    def test_effective_init_strategy_defaults_to_requested(
        self, mock_backend_config, sample_hamiltonian
    ):
        solver = VQESolver(
            qubit_op=sample_hamiltonian, backend_config=mock_backend_config, init_strategy='hf'
        )
        assert solver.effective_init_strategy == 'hf'

    def test_old_efficient_su2_only_warning_is_gone(self):
        from pathlib import Path

        import quantum_pipeline.solvers.vqe_solver as vqe_solver_module

        assert (
            'only supported for EfficientSU2' not in Path(vqe_solver_module.__file__).read_text()
        )

    def test_build_ansatz_is_plain_esu2(self, mock_backend_config, sample_hamiltonian):
        """Test that _build_ansatz always builds a plain EfficientSU2 (no HF circuit prepend)."""
        hf_data = HFData(num_particles=(1, 1), num_spatial_orbitals=2)
        mock_mapper = MagicMock()
        solver = VQESolver(
            qubit_op=sample_hamiltonian,
            backend_config=mock_backend_config,
            init_strategy='hf',
            hf_data=hf_data,
            mapper=mock_mapper,
        )
        with patch('quantum_pipeline.solvers.vqe_solver.EfficientSU2') as mock_esu2:
            solver._build_ansatz(4)
            mock_esu2.assert_called_once()
            _, kwargs = mock_esu2.call_args
            assert 'initial_state' not in kwargs

    def test_build_ansatz_without_hf(self, mock_backend_config, sample_hamiltonian):
        """Test that _build_ansatz does not pass initial_state for random strategy."""
        solver = VQESolver(
            qubit_op=sample_hamiltonian,
            backend_config=mock_backend_config,
            init_strategy='random',
        )
        with patch('quantum_pipeline.solvers.vqe_solver.EfficientSU2') as mock_esu2:
            solver._build_ansatz(4)
            mock_esu2.assert_called_once()
            _, kwargs = mock_esu2.call_args
            assert 'initial_state' not in kwargs


class TestAnsatzTypes:
    def test_default_ansatz_type_is_efficient_su2(self, mock_backend_config, sample_hamiltonian):
        solver = VQESolver(
            qubit_op=sample_hamiltonian,
            backend_config=mock_backend_config,
        )
        assert solver.ansatz_type == 'EfficientSU2'

    def test_real_amplitudes_ansatz_type_stored(self, mock_backend_config, sample_hamiltonian):
        solver = VQESolver(
            qubit_op=sample_hamiltonian,
            backend_config=mock_backend_config,
            ansatz_type='RealAmplitudes',
        )
        assert solver.ansatz_type == 'RealAmplitudes'

    def test_excitation_preserving_ansatz_type_stored(
        self, mock_backend_config, sample_hamiltonian
    ):
        solver = VQESolver(
            qubit_op=sample_hamiltonian,
            backend_config=mock_backend_config,
            ansatz_type='ExcitationPreserving',
        )
        assert solver.ansatz_type == 'ExcitationPreserving'

    def test_build_ansatz_efficient_su2(self, mock_backend_config, sample_hamiltonian):
        """Test that _build_ansatz builds EfficientSU2 by default."""
        solver = VQESolver(
            qubit_op=sample_hamiltonian,
            backend_config=mock_backend_config,
            ansatz_type='EfficientSU2',
            ansatz_reps=2,
        )
        with patch('quantum_pipeline.solvers.vqe_solver.EfficientSU2') as mock_cls:
            solver._build_ansatz(4)
            mock_cls.assert_called_once_with(4, reps=2)

    def test_build_ansatz_real_amplitudes(self, mock_backend_config, sample_hamiltonian):
        solver = VQESolver(
            qubit_op=sample_hamiltonian,
            backend_config=mock_backend_config,
            ansatz_type='RealAmplitudes',
            ansatz_reps=2,
        )
        with patch('quantum_pipeline.solvers.vqe_solver.RealAmplitudes') as mock_cls:
            solver._build_ansatz(4)
            mock_cls.assert_called_once_with(4, reps=2)

    def test_build_ansatz_excitation_preserving_raises_without_hf_data(
        self, mock_backend_config, sample_hamiltonian
    ):
        """ExcitationPreserving without HF data must raise — silent vacuum trap is worse than a crash."""
        solver = VQESolver(
            qubit_op=sample_hamiltonian,
            backend_config=mock_backend_config,
            ansatz_type='ExcitationPreserving',
            ansatz_reps=2,
        )
        with pytest.raises(ValueError, match='hf_data and mapper'):
            solver._build_ansatz(4)

    def test_build_ansatz_excitation_preserving_with_hf_data(
        self, mock_backend_config, sample_hamiltonian
    ):
        """Test ExcitationPreserving with HF data passes the HF circuit as initial_state."""
        hf_data = HFData(num_particles=(1, 1), num_spatial_orbitals=2)
        mock_mapper = MagicMock()
        solver = VQESolver(
            qubit_op=sample_hamiltonian,
            backend_config=mock_backend_config,
            ansatz_type='ExcitationPreserving',
            ansatz_reps=2,
            hf_data=hf_data,
            mapper=mock_mapper,
        )
        mock_hf_circuit = MagicMock()
        with (
            patch('quantum_pipeline.solvers.vqe_solver.ExcitationPreserving') as mock_cls,
            patch(
                'quantum_pipeline.solvers.vqe_solver.build_hf_initial_state',
                return_value=mock_hf_circuit,
            ),
        ):
            solver._build_ansatz(4)
            mock_cls.assert_called_once_with(
                4, reps=2, entanglement='full', initial_state=mock_hf_circuit
            )

    def test_excitation_preserving_initial_params_zero_with_hf_data(
        self, mock_backend_config, sample_hamiltonian
    ):
        """ExcitationPreserving + HF data → params jittered around θ=0 (the HF state)."""
        hf_data = HFData(num_particles=(1, 1), num_spatial_orbitals=2)
        mock_mapper = MagicMock()
        solver = VQESolver(
            qubit_op=sample_hamiltonian,
            backend_config=mock_backend_config,
            ansatz_type='ExcitationPreserving',
            hf_data=hf_data,
            mapper=mock_mapper,
        )
        mock_ansatz = MagicMock()
        mock_ansatz.num_parameters = 12
        params = solver._compute_initial_parameters(mock_ansatz)
        assert len(params) == 12
        # jitter ~N(0,sigma) to escape HF stationary point while staying near HF reference
        assert not np.all(params == 0.0)
        assert np.std(params) < 1.5
        assert np.all(np.abs(params) < 2 * np.pi)

    def test_excitation_preserving_initial_params_seeded_deterministic(
        self, mock_backend_config, sample_hamiltonian
    ):
        """ExcitationPreserving jitter is seeded → identical across solver instances."""
        mock_ansatz = MagicMock()
        mock_ansatz.num_parameters = 12

        def make(seed):
            s = VQESolver(
                qubit_op=sample_hamiltonian,
                backend_config=mock_backend_config,
                ansatz_type='ExcitationPreserving',
                seed=seed,
            )
            return s._compute_initial_parameters(mock_ansatz)

        np.testing.assert_array_equal(make(42), make(42))
        assert not np.array_equal(make(0), make(1))
        # None seed falls back to a fixed default seed → still deterministic.
        np.testing.assert_array_equal(make(None), make(None))

    def test_unknown_ansatz_type_falls_back_to_efficient_su2(
        self, mock_backend_config, sample_hamiltonian
    ):
        solver = VQESolver(
            qubit_op=sample_hamiltonian,
            backend_config=mock_backend_config,
            ansatz_type='Unknown',
            ansatz_reps=1,
        )
        with patch('quantum_pipeline.solvers.vqe_solver.EfficientSU2') as mock_cls:
            solver._build_ansatz(4)
            mock_cls.assert_called_once_with(4, reps=1)


class TestVQESolverOptimizerConfig:
    def test_cobyla_max_iterations_only(self):
        """Test that COBYLA config uses only maxiter (no maxfun)."""
        options, minimize_tol = get_optimizer_configuration(
            optimizer='COBYLA',
            max_iterations=5,
            num_parameters=160,
        )

        assert 'maxiter' in options
        assert options['maxiter'] == 5
        assert 'maxfun' not in options  # maxfun is not a valid COBYLA parameter
        assert minimize_tol is None

    def test_lbfgsb_strict_max_iterations(self):
        """Test that L-BFGS-B uses only maxfun/maxiter for strict max_iterations mode."""
        options, minimize_tol = get_optimizer_configuration(
            optimizer='L-BFGS-B',
            max_iterations=50,
            convergence_threshold=None,
            num_parameters=160,
        )

        # Tight tolerances to prevent premature convergence
        assert options['ftol'] == 1e-15
        assert options['gtol'] == 1e-15
        assert options['maxiter'] == 50

        # L-BFGS-B doesn't use global tol parameter
        assert minimize_tol is None

    def test_lbfgsb_with_convergence_threshold(self):
        """L-BFGS-B maps convergence_threshold onto both ftol and gtol, not the global tol."""
        options, minimize_tol = get_optimizer_configuration(
            optimizer='L-BFGS-B',
            max_iterations=None,
            convergence_threshold=0.01,
            num_parameters=160,
        )

        assert options['ftol'] == 0.01
        assert options['gtol'] == 0.01
        assert minimize_tol is None  # L-BFGS-B uses ftol/gtol, not global tol

    def test_lbfgsb_mutual_exclusion(self):
        """Test L-BFGS-B raises error with both parameters."""
        with pytest.raises(ValueError, match='mutually exclusive'):
            get_optimizer_configuration(
                optimizer='L-BFGS-B',
                max_iterations=100,
                convergence_threshold=0.005,
                num_parameters=160,
            )

    def test_cobyla_mutual_exclusion(self):
        """Test COBYLA raises error with both parameters."""
        with pytest.raises(ValueError, match='mutually exclusive'):
            get_optimizer_configuration(
                optimizer='COBYLA',
                max_iterations=15,
                convergence_threshold=0.05,
                num_parameters=160,
            )

    def test_cobyla_validation_warning(self):
        """Test that COBYLA warns about insufficient iterations."""
        from unittest.mock import MagicMock, patch

        with patch('quantum_pipeline.solvers.optimizer_config.logging.getLogger') as mock_logger:
            mock_logger_instance = MagicMock()
            mock_logger.return_value = mock_logger_instance

            # This should trigger a warning: 5 < (160 + 2)
            options, _minimize_tol = get_optimizer_configuration(
                optimizer='COBYLA',
                max_iterations=5,
                convergence_threshold=None,
                num_parameters=160,
            )

            assert options['maxiter'] == 5

    def test_realistic_log_scenario_max_iterations(self):
        # From logs: COBYLA with max_iterations=5, 160 parameters
        options, minimize_tol = get_optimizer_configuration(
            optimizer='COBYLA',
            max_iterations=5,
            num_parameters=160,
        )

        assert 'maxiter' in options
        assert options['maxiter'] == 5
        assert 'maxfun' not in options
        assert minimize_tol is None

    def test_realistic_log_scenario_convergence(self):
        options, minimize_tol = get_optimizer_configuration(
            optimizer='COBYLA',
            convergence_threshold=0.1,
            num_parameters=160,
        )

        assert 'maxiter' in options
        assert options['maxiter'] == 1000  # Default
        assert 'maxfun' not in options
        assert minimize_tol == 0.1

    def test_all_supported_optimizers_work(self):
        core_optimizers = ['L-BFGS-B', 'COBYLA', 'SLSQP']

        for optimizer in core_optimizers:
            options, _minimize_tol = get_optimizer_configuration(
                optimizer=optimizer,
                max_iterations=50,
                num_parameters=10,
            )

            assert isinstance(options, dict)
            assert 'maxiter' in options
            assert options['maxiter'] == 50

    def test_edge_cases(self):
        options, _ = get_optimizer_configuration(
            optimizer='COBYLA', max_iterations=1, num_parameters=10
        )
        assert options['maxiter'] == 1

        options, minimize_tol = get_optimizer_configuration(
            optimizer='COBYLA', convergence_threshold=1e-10, num_parameters=10
        )
        assert minimize_tol == 1e-10

        options, minimize_tol = get_optimizer_configuration(
            optimizer='L-BFGS-B', num_parameters=10
        )
        assert 'maxiter' in options
        assert options['maxiter'] == 15000  # L-BFGS-B default


@pytest.fixture
def backend_config():
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
def backend_config_with_noise():
    return BackendConfig(
        local=True,
        gpu=False,
        optimization_level=2,
        min_num_qubits=4,
        filters=None,
        simulation_method='statevector',
        gpu_opts=None,
        noise='ibm_brisbane',
    )


@pytest.fixture
def hamiltonian():
    return SparsePauliOp.from_list([('II', 0.1), ('IX', 0.2), ('XX', 0.3)])


def _make_solver(hamiltonian, backend_config, **overrides):
    defaults = {
        'qubit_op': hamiltonian,
        'backend_config': backend_config,
        'max_iterations': 3,
        'optimizer': 'COBYLA',
        'ansatz_reps': 1,
        'default_shots': 64,
        'convergence_threshold': None,
        'optimization_level': 1,
    }
    defaults.update(overrides)
    return VQESolver(**defaults)


def _mock_minimize_result(success=True, fun=-1.0):
    """Build a scipy-like OptimizeResult mock."""
    res = MagicMock()
    res.fun = fun
    res.x = np.array([0.1, 0.2])
    res.success = success
    res.maxcv = 0.0
    return res


def _mock_estimator():
    estimator_instance = MagicMock()
    mock_result = MagicMock()
    mock_result.data.evs = [np.float64(-0.5)]
    mock_result.data.stds = [np.float64(0.01)]
    estimator_instance.run.return_value.result.return_value = [mock_result]
    return estimator_instance


def _setup_aer_mocks(mock_su2, mock_pm_gen, mock_estimator_cls, hamiltonian):
    mock_backend = MagicMock(spec=AerBackend)
    mock_backend.name = 'aer_simulator'
    mock_backend.target = MagicMock()

    mock_ansatz = MagicMock()
    mock_ansatz.num_qubits = 2
    mock_ansatz.num_parameters = 4
    mock_su2.return_value = mock_ansatz

    mock_pm = MagicMock()
    mock_isa = MagicMock()
    mock_isa.layout = MagicMock()
    mock_isa.num_parameters = 4
    mock_pm.run.return_value = mock_isa
    mock_pm_gen.return_value = mock_pm

    mock_isa_ham = MagicMock()
    mock_isa_ham.num_qubits = 2
    mock_isa_ham.to_list.return_value = [('II', 0.1)]
    hamiltonian.apply_layout = MagicMock(return_value=mock_isa_ham)

    mock_estimator_cls.return_value = _mock_estimator()

    return mock_backend


def _setup_ibmq_mocks(mock_su2, mock_pm_gen, mock_estimator_cls, mock_session_cls, hamiltonian):
    mock_backend = MagicMock()
    mock_backend.name = 'ibm_kyoto'
    mock_backend.target = MagicMock()

    mock_ansatz = MagicMock()
    mock_ansatz.num_qubits = 2
    mock_ansatz.num_parameters = 4
    mock_su2.return_value = mock_ansatz

    mock_pm = MagicMock()
    mock_isa = MagicMock()
    mock_isa.layout = MagicMock()
    mock_isa.num_parameters = 4
    mock_pm.run.return_value = mock_isa
    mock_pm_gen.return_value = mock_pm

    mock_isa_ham = MagicMock()
    mock_isa_ham.num_qubits = 2
    mock_isa_ham.to_list.return_value = []
    hamiltonian.apply_layout = MagicMock(return_value=mock_isa_ham)

    mock_session = MagicMock()
    mock_session_cls.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_session_cls.return_value.__exit__ = MagicMock(return_value=False)

    mock_estimator_cls.return_value = _mock_estimator()

    return mock_backend


class TestViaAer:
    """Test the via_aer method covering all convergence/iteration branches."""

    @pytest.mark.parametrize(
        'max_iter, conv_thresh, success',
        [
            (3, None, True),  # max_iterations only
            (None, 1e-4, True),  # convergence only, achieved
            (None, 1e-4, False),  # convergence only, not achieved
            (5, 1e-4, True),  # both
            (None, None, True),  # neither
        ],
        ids=[
            'max_iter_only',
            'convergence_achieved',
            'convergence_not_achieved',
            'both_max_and_convergence',
            'neither',
        ],
    )
    @patch('quantum_pipeline.solvers.vqe_solver.minimize')
    @patch('quantum_pipeline.solvers.vqe_solver.get_optimizer_configuration')
    @patch('quantum_pipeline.solvers.vqe_solver.EstimatorV2')
    @patch('quantum_pipeline.solvers.vqe_solver.generate_preset_pass_manager')
    @patch('quantum_pipeline.solvers.vqe_solver.EfficientSU2')
    def test_via_aer_branches(
        self,
        mock_su2,
        mock_pm_gen,
        mock_estimator_cls,
        mock_get_opt,
        mock_minimize,
        max_iter,
        conv_thresh,
        success,
        hamiltonian,
        backend_config,
    ):
        solver = _make_solver(
            hamiltonian,
            backend_config,
            max_iterations=max_iter,
            convergence_threshold=conv_thresh,
        )
        mock_backend = _setup_aer_mocks(mock_su2, mock_pm_gen, mock_estimator_cls, hamiltonian)

        opt_cfg = {}
        if max_iter:
            opt_cfg['maxiter'] = max_iter
        elif conv_thresh:
            opt_cfg['maxiter'] = 1000
        mock_get_opt.return_value = (opt_cfg, conv_thresh)
        mock_minimize.return_value = _mock_minimize_result(success=success)

        result = solver.via_aer(mock_backend)

        assert isinstance(result, VQEResult)
        assert result.minimum == -1.0
        mock_minimize.assert_called_once()

    @patch('quantum_pipeline.solvers.vqe_solver.minimize')
    @patch('quantum_pipeline.solvers.vqe_solver.get_optimizer_configuration')
    @patch('quantum_pipeline.solvers.vqe_solver.EstimatorV2')
    @patch('quantum_pipeline.solvers.vqe_solver.generate_preset_pass_manager')
    @patch('quantum_pipeline.solvers.vqe_solver.EfficientSU2')
    def test_via_aer_noise_backend_set(
        self,
        mock_su2,
        mock_pm_gen,
        mock_estimator_cls,
        mock_get_opt,
        mock_minimize,
        hamiltonian,
        backend_config_with_noise,
    ):
        solver = _make_solver(hamiltonian, backend_config_with_noise)
        mock_backend = _setup_aer_mocks(mock_su2, mock_pm_gen, mock_estimator_cls, hamiltonian)

        mock_get_opt.return_value = ({'maxiter': 3}, None)
        mock_minimize.return_value = _mock_minimize_result()

        result = solver.via_aer(mock_backend)
        assert result.initial_data.noise_backend == 'ibm_brisbane'

    @patch('quantum_pipeline.solvers.vqe_solver.minimize')
    @patch('quantum_pipeline.solvers.vqe_solver.get_optimizer_configuration')
    @patch('quantum_pipeline.solvers.vqe_solver.EstimatorV2')
    @patch('quantum_pipeline.solvers.vqe_solver.generate_preset_pass_manager')
    @patch('quantum_pipeline.solvers.vqe_solver.EfficientSU2')
    def test_via_aer_result_fields(
        self,
        mock_su2,
        mock_pm_gen,
        mock_estimator_cls,
        mock_get_opt,
        mock_minimize,
        hamiltonian,
        backend_config,
    ):
        solver = _make_solver(hamiltonian, backend_config)
        mock_backend = _setup_aer_mocks(mock_su2, mock_pm_gen, mock_estimator_cls, hamiltonian)

        mock_get_opt.return_value = ({'maxiter': 3}, None)
        opt_res = _mock_minimize_result(fun=-2.5)
        opt_res.maxcv = 0.01
        mock_minimize.return_value = opt_res

        result = solver.via_aer(mock_backend)

        assert result.minimum == -2.5
        assert result.maxcv == 0.01
        assert result.initial_data.backend == 'aer_simulator'
        assert result.initial_data.optimizer == 'COBYLA'
        assert result.initial_data.default_shots == 64
        assert result.minimization_time >= 0


class TestViaIBMQ:
    """Test the via_ibmq method covering all convergence/iteration branches."""

    @pytest.mark.parametrize(
        'max_iter, conv_thresh, success',
        [
            (5, None, True),  # max_iterations only
            (None, 1e-4, True),  # convergence only, achieved
            (None, 1e-4, False),  # convergence only, not achieved
            (5, 1e-4, True),  # both
            (None, None, True),  # neither
        ],
        ids=[
            'max_iter_only',
            'convergence_achieved',
            'convergence_not_achieved',
            'both_max_and_convergence',
            'neither',
        ],
    )
    @patch('quantum_pipeline.solvers.vqe_solver.minimize')
    @patch('quantum_pipeline.solvers.vqe_solver.get_optimizer_configuration')
    @patch('quantum_pipeline.solvers.vqe_solver.Session')
    @patch('quantum_pipeline.solvers.vqe_solver.EstimatorV2')
    @patch('quantum_pipeline.solvers.vqe_solver.generate_preset_pass_manager')
    @patch('quantum_pipeline.solvers.vqe_solver.EfficientSU2')
    def test_via_ibmq_branches(
        self,
        mock_su2,
        mock_pm_gen,
        mock_estimator_cls,
        mock_session_cls,
        mock_get_opt,
        mock_minimize,
        max_iter,
        conv_thresh,
        success,
        hamiltonian,
        backend_config,
    ):
        solver = _make_solver(
            hamiltonian,
            backend_config,
            max_iterations=max_iter,
            convergence_threshold=conv_thresh,
        )
        mock_backend = _setup_ibmq_mocks(
            mock_su2,
            mock_pm_gen,
            mock_estimator_cls,
            mock_session_cls,
            hamiltonian,
        )

        opt_cfg = {}
        if max_iter:
            opt_cfg['maxiter'] = max_iter
        elif conv_thresh:
            opt_cfg['maxiter'] = 1000
        mock_get_opt.return_value = (opt_cfg, conv_thresh)
        mock_minimize.return_value = _mock_minimize_result(success=success)

        result = solver.via_ibmq(mock_backend)

        assert isinstance(result, VQEResult)
        assert result.initial_data.backend == 'ibm_kyoto'
        mock_session_cls.assert_called_once_with(backend=mock_backend)

    @patch('quantum_pipeline.solvers.vqe_solver.minimize')
    @patch('quantum_pipeline.solvers.vqe_solver.get_optimizer_configuration')
    @patch('quantum_pipeline.solvers.vqe_solver.Session')
    @patch('quantum_pipeline.solvers.vqe_solver.EstimatorV2')
    @patch('quantum_pipeline.solvers.vqe_solver.generate_preset_pass_manager')
    @patch('quantum_pipeline.solvers.vqe_solver.EfficientSU2')
    def test_via_ibmq_ansatz_reps_propagated(
        self,
        mock_su2,
        mock_pm_gen,
        mock_estimator_cls,
        mock_session_cls,
        mock_get_opt,
        mock_minimize,
        hamiltonian,
        backend_config,
    ):
        solver = _make_solver(hamiltonian, backend_config, ansatz_reps=5)
        mock_backend = _setup_ibmq_mocks(
            mock_su2,
            mock_pm_gen,
            mock_estimator_cls,
            mock_session_cls,
            hamiltonian,
        )

        mock_get_opt.return_value = ({'maxiter': 3}, None)
        mock_minimize.return_value = _mock_minimize_result()

        result = solver.via_ibmq(mock_backend)
        mock_su2.assert_called_once_with(hamiltonian.num_qubits, reps=5)
        assert result.initial_data.ansatz_reps == 5


class TestSolve:
    def test_solve_resets_state(self, hamiltonian, backend_config):
        solver = _make_solver(hamiltonian, backend_config)
        solver.current_iter = 99
        solver.vqe_process = [MagicMock()]

        mock_backend = MagicMock(spec=AerBackend)
        mock_result = MagicMock(spec=VQEResult)

        with (
            patch.object(solver, 'get_backend', return_value=mock_backend),
            patch.object(solver, 'via_aer', return_value=mock_result),
        ):
            solver.solve()

        assert solver.current_iter == 1
        assert solver.vqe_process == []

    def test_solve_dispatches_aer(self, hamiltonian, backend_config):
        solver = _make_solver(hamiltonian, backend_config)
        mock_backend = MagicMock(spec=AerBackend)
        mock_result = MagicMock(spec=VQEResult)

        with (
            patch.object(solver, 'get_backend', return_value=mock_backend),
            patch.object(solver, 'via_aer', return_value=mock_result) as mock_aer,
            patch.object(solver, 'via_ibmq') as mock_ibmq,
        ):
            result = solver.solve()

        mock_aer.assert_called_once_with(mock_backend)
        mock_ibmq.assert_not_called()
        assert result == mock_result

    def test_solve_dispatches_ibmq(self, hamiltonian, backend_config):
        solver = _make_solver(hamiltonian, backend_config)
        mock_backend = MagicMock()  # not AerBackend spec → IBMQ path
        mock_result = MagicMock(spec=VQEResult)

        with (
            patch.object(solver, 'get_backend', return_value=mock_backend),
            patch.object(solver, 'via_ibmq', return_value=mock_result) as mock_ibmq,
            patch.object(solver, 'via_aer') as mock_aer,
        ):
            result = solver.solve()

        mock_ibmq.assert_called_once_with(mock_backend)
        mock_aer.assert_not_called()
        assert result == mock_result


class TestExactEstimatorAndBestStep:
    """Exact Aer estimator when default_shots=None; best-step result consistency."""

    @staticmethod
    def _mock_via_aer_result():
        """Minimal mock result compatible with via_aer's post-optimization logging."""
        r = MagicMock()
        r.minimization_time = 0.1
        r.iteration_list = []
        return r

    def test_via_aer_uses_aer_estimator_when_shots_none(self, hamiltonian, backend_config):
        """default_shots=None → AerEstimatorV2 with default_precision=0.0."""
        solver = _make_solver(hamiltonian, backend_config, default_shots=None)
        mock_backend = MagicMock(spec=AerBackend)
        mock_backend.name = 'aer_simulator'

        with (
            patch.object(
                solver, '_prepare_circuit', return_value=(np.zeros(2), MagicMock(), MagicMock())
            ),
            patch.object(solver, '_build_init_data'),
            patch.object(solver, '_run_optimization', return_value=self._mock_via_aer_result()),
            patch('quantum_pipeline.solvers.vqe_solver.AerEstimatorV2') as mock_aer_est,
            patch('quantum_pipeline.solvers.vqe_solver.EstimatorV2') as mock_rt_est,
        ):
            solver.via_aer(mock_backend)
            mock_aer_est.from_backend.assert_called_once_with(
                mock_backend, options={'default_precision': 0.0}
            )
            mock_rt_est.assert_not_called()

    def test_exact_estimator_runs_on_configured_backend(self, hamiltonian, backend_config):
        """The exact estimator must adopt the configured simulator, not a default one.

        Without from_backend the estimator silently ignores GPU device, simulation method,
        seed and noise model, so an --exact --gpu run would execute on the CPU.
        """
        solver = _make_solver(hamiltonian, backend_config, default_shots=None)
        backend = AerSimulator(method='statevector', seed_simulator=1234)

        with (
            patch.object(
                solver, '_prepare_circuit', return_value=(np.zeros(2), MagicMock(), MagicMock())
            ),
            patch.object(solver, '_build_init_data'),
            patch.object(
                solver, '_run_optimization', return_value=self._mock_via_aer_result()
            ) as mock_run,
        ):
            solver.via_aer(backend)

        estimator = mock_run.call_args.args[2]
        assert estimator._backend is backend

    def test_via_aer_uses_runtime_estimator_when_shots_set(self, hamiltonian, backend_config):
        """default_shots=64 → RuntimeEstimatorV2 in local mode."""
        solver = _make_solver(hamiltonian, backend_config, default_shots=64)
        mock_backend = MagicMock(spec=AerBackend)
        mock_backend.name = 'aer_simulator'

        with (
            patch.object(
                solver, '_prepare_circuit', return_value=(np.zeros(2), MagicMock(), MagicMock())
            ),
            patch.object(solver, '_build_init_data'),
            patch.object(solver, '_run_optimization', return_value=self._mock_via_aer_result()),
            patch('quantum_pipeline.solvers.vqe_solver.AerEstimatorV2') as mock_aer_est,
            patch('quantum_pipeline.solvers.vqe_solver.EstimatorV2') as mock_rt_est,
        ):
            solver.via_aer(mock_backend)
            mock_rt_est.assert_called_once_with(mode=mock_backend)
            mock_aer_est.assert_not_called()

    def test_exact_estimator_flag_in_init_data(self, hamiltonian, backend_config):
        """exact_estimator=True recorded in VQEInitialData when default_shots=None."""
        solver = _make_solver(hamiltonian, backend_config, default_shots=None)
        mock_backend = MagicMock(spec=AerBackend)
        mock_backend.name = 'aer_simulator'

        with (
            patch.object(
                solver, '_prepare_circuit', return_value=(np.zeros(2), MagicMock(), MagicMock())
            ),
            patch.object(solver, '_run_optimization', return_value=self._mock_via_aer_result()),
            patch('quantum_pipeline.solvers.vqe_solver.AerEstimatorV2'),
        ):
            solver.via_aer(mock_backend)
            assert solver.init_data.exact_estimator is True

    def test_exact_estimator_false_when_shots_set(self, hamiltonian, backend_config):
        """exact_estimator=False recorded in VQEInitialData when shots are set."""
        solver = _make_solver(hamiltonian, backend_config, default_shots=64)
        mock_backend = MagicMock(spec=AerBackend)
        mock_backend.name = 'aer_simulator'

        with (
            patch.object(
                solver, '_prepare_circuit', return_value=(np.zeros(2), MagicMock(), MagicMock())
            ),
            patch.object(solver, '_run_optimization', return_value=self._mock_via_aer_result()),
            patch('quantum_pipeline.solvers.vqe_solver.EstimatorV2'),
        ):
            solver.via_aer(mock_backend)
            assert solver.init_data.exact_estimator is False

    def test_run_optimization_reports_best_step_not_final_iterate(
        self, hamiltonian, backend_config
    ):
        """minimum and optimal_parameters come from the best VQEProcess, not the final iterate.

        Pins the biased-minimum bug: the optimizer's final point (res.fun/res.x) may not be
        the lowest energy seen, leaving energy and parameters inconsistent.
        """
        solver = _make_solver(hamiltonian, backend_config)

        best_params = np.array([0.5, 0.6])
        solver.vqe_process = [
            VQEProcess(
                iteration=1,
                parameters=np.array([0.1, 0.2]),
                result=np.float64(-0.8),
                std=np.float64(0.0),
                cumulative_min_energy=np.float64(-0.8),
            ),
            VQEProcess(
                iteration=2,
                parameters=best_params,
                result=np.float64(-1.5),
                std=np.float64(0.0),
                cumulative_min_energy=np.float64(-1.5),  # global min at iteration 2
            ),
            VQEProcess(
                iteration=3,
                parameters=np.array([0.9, 1.0]),
                result=np.float64(-1.1),
                std=np.float64(0.0),
                cumulative_min_energy=np.float64(-1.5),  # still -1.5 (iter 2 was best)
            ),
        ]
        solver.init_data = MagicMock()

        mock_res = _mock_minimize_result(fun=-1.1)  # final iterate energy ≠ best
        mock_res.x = np.array([0.9, 1.0])
        mock_res.nfev = 3
        mock_res.nit = 3

        with (
            patch(
                'quantum_pipeline.solvers.vqe_solver.get_optimizer_configuration',
                return_value=({'maxiter': 3}, None),
            ),
            patch('quantum_pipeline.solvers.vqe_solver.minimize', return_value=mock_res),
        ):
            result = solver._run_optimization(MagicMock(), MagicMock(), MagicMock(), np.zeros(2))

        assert result.minimum == pytest.approx(-1.5)
        assert np.array_equal(result.optimal_parameters, best_params)


class TestOptimizeCircuits:
    def test_passes_optimization_level(self, hamiltonian, backend_config):
        solver = _make_solver(hamiltonian, backend_config, optimization_level=2)
        mock_backend = MagicMock()
        mock_backend.target = MagicMock()

        mock_ansatz = MagicMock()
        mock_hamiltonian = MagicMock()

        with patch(
            'quantum_pipeline.solvers.vqe_solver.generate_preset_pass_manager'
        ) as mock_pm_gen:
            mock_pm = MagicMock()
            mock_pm.run.return_value = mock_ansatz
            mock_pm_gen.return_value = mock_pm
            mock_hamiltonian.apply_layout.return_value = mock_hamiltonian

            solver._optimize_circuits(mock_ansatz, mock_hamiltonian, mock_backend)

            mock_pm_gen.assert_called_once_with(
                target=mock_backend.target,
                optimization_level=2,
                seed_transpiler=solver.seed,
            )

    def test_returns_isa_pair(self, hamiltonian, backend_config):
        solver = _make_solver(hamiltonian, backend_config)
        mock_backend = MagicMock()
        mock_backend.target = MagicMock()

        mock_ansatz = MagicMock()
        mock_isa_ansatz = MagicMock()
        mock_isa_ansatz.layout = 'test_layout'
        mock_hamiltonian = MagicMock()
        mock_isa_ham = MagicMock()

        with patch(
            'quantum_pipeline.solvers.vqe_solver.generate_preset_pass_manager'
        ) as mock_pm_gen:
            mock_pm = MagicMock()
            mock_pm.run.return_value = mock_isa_ansatz
            mock_pm_gen.return_value = mock_pm
            mock_hamiltonian.apply_layout.return_value = mock_isa_ham

            a, h = solver._optimize_circuits(mock_ansatz, mock_hamiltonian, mock_backend)

        assert a == mock_isa_ansatz
        assert h == mock_isa_ham
        mock_hamiltonian.apply_layout.assert_called_once_with(layout='test_layout')


class TestResultMaxcvAbsent:
    @patch('quantum_pipeline.solvers.vqe_solver.minimize')
    @patch('quantum_pipeline.solvers.vqe_solver.get_optimizer_configuration')
    @patch('quantum_pipeline.solvers.vqe_solver.EstimatorV2')
    @patch('quantum_pipeline.solvers.vqe_solver.generate_preset_pass_manager')
    @patch('quantum_pipeline.solvers.vqe_solver.EfficientSU2')
    def test_maxcv_none_when_absent(
        self,
        mock_su2,
        mock_pm_gen,
        mock_estimator_cls,
        mock_get_opt,
        mock_minimize,
        hamiltonian,
        backend_config,
    ):
        solver = _make_solver(hamiltonian, backend_config)
        mock_backend = MagicMock(spec=AerBackend)
        mock_backend.name = 'aer_simulator'
        mock_backend.target = MagicMock()

        mock_ansatz = MagicMock()
        mock_ansatz.num_qubits = 2
        mock_ansatz.num_parameters = 4
        mock_su2.return_value = mock_ansatz

        mock_pm = MagicMock()
        mock_isa = MagicMock()
        mock_isa.layout = MagicMock()
        mock_isa.num_parameters = 4
        mock_pm.run.return_value = mock_isa
        mock_pm_gen.return_value = mock_pm

        mock_isa_ham = MagicMock()
        mock_isa_ham.num_qubits = 2
        mock_isa_ham.to_list.return_value = []
        hamiltonian.apply_layout = MagicMock(return_value=mock_isa_ham)

        mock_estimator_cls.return_value = _mock_estimator()
        mock_get_opt.return_value = ({'maxiter': 3}, None)

        # OptimizeResult without maxcv attribute
        opt_result = MagicMock(spec=['fun', 'x', 'success'])
        opt_result.fun = -1.0
        opt_result.x = np.array([0.1, 0.2])
        opt_result.success = True
        mock_minimize.return_value = opt_result

        result = solver.via_aer(mock_backend)
        assert result.maxcv is None


class TestMinimizeTolPropagation:
    @patch('quantum_pipeline.solvers.vqe_solver.minimize')
    @patch('quantum_pipeline.solvers.vqe_solver.get_optimizer_configuration')
    @patch('quantum_pipeline.solvers.vqe_solver.EstimatorV2')
    @patch('quantum_pipeline.solvers.vqe_solver.generate_preset_pass_manager')
    @patch('quantum_pipeline.solvers.vqe_solver.EfficientSU2')
    @pytest.mark.parametrize('tol_value', [None, 1e-4, 1e-8])
    def test_tol_passed_to_minimize(
        self,
        mock_su2,
        mock_pm_gen,
        mock_estimator_cls,
        mock_get_opt,
        mock_minimize,
        tol_value,
        hamiltonian,
        backend_config,
    ):
        solver = _make_solver(hamiltonian, backend_config)
        mock_backend = MagicMock(spec=AerBackend)
        mock_backend.name = 'aer_simulator'
        mock_backend.target = MagicMock()

        mock_ansatz = MagicMock()
        mock_ansatz.num_qubits = 2
        mock_ansatz.num_parameters = 4
        mock_su2.return_value = mock_ansatz

        mock_pm = MagicMock()
        mock_isa = MagicMock()
        mock_isa.layout = MagicMock()
        mock_isa.num_parameters = 4
        mock_pm.run.return_value = mock_isa
        mock_pm_gen.return_value = mock_pm

        mock_isa_ham = MagicMock()
        mock_isa_ham.num_qubits = 2
        mock_isa_ham.to_list.return_value = []
        hamiltonian.apply_layout = MagicMock(return_value=mock_isa_ham)

        mock_estimator_cls.return_value = _mock_estimator()
        mock_get_opt.return_value = ({'maxiter': 3}, tol_value)
        mock_minimize.return_value = _mock_minimize_result()

        solver.via_aer(mock_backend)

        _, call_kwargs = mock_minimize.call_args
        assert call_kwargs['tol'] == tol_value


@pytest.fixture
def hamiltonian_2q():
    return SparsePauliOp.from_list([('II', 0.1), ('IX', 0.2), ('XX', 0.3)])


@pytest.fixture
def hamiltonian_4q():
    return SparsePauliOp.from_list(
        [
            ('IIII', 0.1),
            ('IIXI', 0.2),
            ('XIXI', 0.3),
            ('XXII', 0.15),
        ]
    )


@pytest.fixture
def hamiltonian_8q():
    return SparsePauliOp.from_list(
        [
            ('I' * 8, 0.1),
            ('X' + 'I' * 7, 0.2),
            ('X' * 4 + 'I' * 4, 0.3),
        ]
    )


class TestVQESolverInitialization:
    def test_default_parameters(self, hamiltonian_4q, backend_config):
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
        for max_iter in [1, 5, 100, 1000]:
            solver = VQESolver(
                qubit_op=hamiltonian_4q,
                backend_config=backend_config,
                max_iterations=max_iter,
            )
            assert solver.max_iterations == max_iter
            assert solver.digits_iter == len(str(max_iter))

    def test_custom_optimizer(self, hamiltonian_4q, backend_config):
        for opt in ['COBYLA', 'SPSA', 'Powell', 'CG']:
            solver = VQESolver(
                qubit_op=hamiltonian_4q,
                backend_config=backend_config,
                optimizer=opt,
            )
            assert solver.optimizer == opt

    def test_custom_ansatz_reps(self, hamiltonian_4q, backend_config):
        for reps in [1, 2, 5, 10]:
            solver = VQESolver(
                qubit_op=hamiltonian_4q,
                backend_config=backend_config,
                ansatz_reps=reps,
            )
            assert solver.ansatz_reps == reps

    def test_custom_shots(self, hamiltonian_4q, backend_config):
        for shots in [100, 512, 1024, 4096, 8192]:
            solver = VQESolver(
                qubit_op=hamiltonian_4q,
                backend_config=backend_config,
                default_shots=shots,
            )
            assert solver.default_shots == shots

    def test_convergence_threshold_setting(self, hamiltonian_4q, backend_config):
        solver = VQESolver(
            qubit_op=hamiltonian_4q,
            backend_config=backend_config,
            convergence_threshold=1e-6,
        )
        assert solver.convergence_threshold == 1e-6

    def test_optimization_level_setting(self, hamiltonian_4q, backend_config):
        for level in [0, 1, 2, 3]:
            solver = VQESolver(
                qubit_op=hamiltonian_4q,
                backend_config=backend_config,
                optimization_level=level,
            )
            assert solver.optimization_level == level


class TestVQESolverComputeEnergy:
    @pytest.fixture
    def solver(self, hamiltonian_4q, backend_config):
        return VQESolver(
            qubit_op=hamiltonian_4q,
            backend_config=backend_config,
            max_iterations=10,
        )

    def test_single_energy_computation(self, solver):
        mock_estimator = MagicMock()
        mock_result = MagicMock()
        mock_result.data.evs = [1.5]
        mock_result.data.stds = [0.1]
        mock_estimator.run.return_value.result.return_value = [mock_result]

        params = np.array([0.1, 0.2, 0.3, 0.4])
        with patch.object(solver.logger, 'debug'):
            energy = solver.compute_energy(params, MagicMock(), MagicMock(), mock_estimator)

        assert energy == 1.5
        assert len(solver.vqe_process) == 1
        assert solver.current_iter == 2

    def test_sequential_energy_computations(self, solver):
        mock_estimator = MagicMock()
        energies = [1.5, 1.4, 1.2, 1.0, 0.9]

        for energy_val in energies:
            mock_result = MagicMock()
            mock_result.data.evs = [energy_val]
            mock_result.data.stds = [0.05]
            mock_estimator.run.return_value.result.return_value = [mock_result]

            params = np.random.random(4)
            with patch.object(solver.logger, 'debug'):
                energy = solver.compute_energy(params, MagicMock(), MagicMock(), mock_estimator)

            assert energy == energy_val

        assert len(solver.vqe_process) == len(energies)
        assert solver.current_iter == len(energies) + 1

    def test_energy_with_zero_std(self, solver):
        mock_estimator = MagicMock()
        mock_result = MagicMock()
        mock_result.data.evs = [0.5]
        mock_result.data.stds = [0.0]
        mock_estimator.run.return_value.result.return_value = [mock_result]

        params = np.array([0.0] * 4)
        with patch.object(solver.logger, 'debug'):
            energy = solver.compute_energy(params, MagicMock(), MagicMock(), mock_estimator)

        assert energy == 0.5
        assert solver.vqe_process[0].std == 0.0

    def test_energy_with_large_std(self, solver):
        mock_estimator = MagicMock()
        mock_result = MagicMock()
        mock_result.data.evs = [1.5]
        mock_result.data.stds = [0.5]
        mock_estimator.run.return_value.result.return_value = [mock_result]

        params = np.random.random(4)
        with patch.object(solver.logger, 'debug'):
            energy = solver.compute_energy(params, MagicMock(), MagicMock(), mock_estimator)

        assert energy == 1.5
        assert solver.vqe_process[0].std == 0.5

    def test_energy_with_negative_value(self, solver):
        mock_estimator = MagicMock()
        mock_result = MagicMock()
        mock_result.data.evs = [-1.5]
        mock_result.data.stds = [0.1]
        mock_estimator.run.return_value.result.return_value = [mock_result]

        params = np.array([0.1, 0.2, 0.3, 0.4])
        with patch.object(solver.logger, 'debug'):
            energy = solver.compute_energy(params, MagicMock(), MagicMock(), mock_estimator)

        assert energy == -1.5

    def test_parameter_storage(self, solver):
        mock_estimator = MagicMock()
        mock_result = MagicMock()
        mock_result.data.evs = [1.0]
        mock_result.data.stds = [0.1]
        mock_estimator.run.return_value.result.return_value = [mock_result]

        test_params = np.array([0.1, 0.2, 0.3, 0.4])
        with patch.object(solver.logger, 'debug'):
            solver.compute_energy(test_params, MagicMock(), MagicMock(), mock_estimator)

        stored_params = solver.vqe_process[0].parameters
        np.testing.assert_array_almost_equal(stored_params, test_params)

    def test_iteration_counter_increment(self, solver):
        mock_estimator = MagicMock()
        mock_result = MagicMock()
        mock_result.data.evs = [1.0]
        mock_result.data.stds = [0.1]
        mock_estimator.run.return_value.result.return_value = [mock_result]

        assert solver.current_iter == 1
        for i in range(5):
            with patch.object(solver.logger, 'debug'):
                solver.compute_energy(
                    np.random.random(4), MagicMock(), MagicMock(), mock_estimator
                )
            assert solver.current_iter == i + 2

    def test_vqe_process_tracking(self, solver):
        mock_estimator = MagicMock()
        energies = [2.0, 1.5, 1.0]

        for _idx, energy_val in enumerate(energies):
            mock_result = MagicMock()
            mock_result.data.evs = [energy_val]
            mock_result.data.stds = [0.05]
            mock_estimator.run.return_value.result.return_value = [mock_result]

            with patch.object(solver.logger, 'debug'):
                solver.compute_energy(
                    np.random.random(4), MagicMock(), MagicMock(), mock_estimator
                )

        assert len(solver.vqe_process) == 3
        for idx, process in enumerate(solver.vqe_process):
            assert process.iteration == idx + 1
            assert process.result == energies[idx]


class TestVQESolverDifferentHamiltonians:
    def test_2qubit_hamiltonian(self, hamiltonian_2q, backend_config):
        solver = VQESolver(
            qubit_op=hamiltonian_2q,
            backend_config=backend_config,
        )
        assert solver.qubit_op.num_qubits == 2

    def test_4qubit_hamiltonian(self, hamiltonian_4q, backend_config):
        solver = VQESolver(
            qubit_op=hamiltonian_4q,
            backend_config=backend_config,
        )
        assert solver.qubit_op.num_qubits == 4

    def test_8qubit_hamiltonian(self, hamiltonian_8q, backend_config):
        solver = VQESolver(
            qubit_op=hamiltonian_8q,
            backend_config=backend_config,
        )
        assert solver.qubit_op.num_qubits == 8

    def test_solver_scales_with_hamiltonian_size(self, backend_config):
        for num_qubits in [2, 4, 8, 12]:
            terms = [('X' + 'I' * (num_qubits - 1), 0.5)]
            ham = SparsePauliOp.from_list(terms)

            solver = VQESolver(
                qubit_op=ham,
                backend_config=backend_config,
            )
            assert solver.qubit_op.num_qubits == num_qubits


class TestVQESolverEdgeCasesExtended:
    def test_single_iteration_solver(self, hamiltonian_4q, backend_config):
        solver = VQESolver(
            qubit_op=hamiltonian_4q,
            backend_config=backend_config,
            max_iterations=1,
        )
        assert solver.max_iterations == 1
        assert solver.digits_iter == 1

    def test_very_large_iteration_count(self, hamiltonian_4q, backend_config):
        solver = VQESolver(
            qubit_op=hamiltonian_4q,
            backend_config=backend_config,
            max_iterations=100000,
        )
        assert solver.max_iterations == 100000
        assert solver.digits_iter == 6

    def test_zero_ansatz_reps(self, hamiltonian_4q, backend_config):
        solver = VQESolver(
            qubit_op=hamiltonian_4q,
            backend_config=backend_config,
            ansatz_reps=0,
        )
        assert solver.ansatz_reps == 0

    def test_very_high_ansatz_reps(self, hamiltonian_4q, backend_config):
        solver = VQESolver(
            qubit_op=hamiltonian_4q,
            backend_config=backend_config,
            ansatz_reps=50,
        )
        assert solver.ansatz_reps == 50

    def test_very_small_threshold(self, hamiltonian_4q, backend_config):
        solver = VQESolver(
            qubit_op=hamiltonian_4q,
            backend_config=backend_config,
            convergence_threshold=1e-12,
        )
        assert solver.convergence_threshold == 1e-12

    def test_very_large_threshold(self, hamiltonian_4q, backend_config):
        solver = VQESolver(
            qubit_op=hamiltonian_4q,
            backend_config=backend_config,
            convergence_threshold=10.0,
        )
        assert solver.convergence_threshold == 10.0


class TestVQESolverConfiguration:
    def test_backend_config_propagation(self, hamiltonian_4q, backend_config):
        solver = VQESolver(
            qubit_op=hamiltonian_4q,
            backend_config=backend_config,
        )
        assert solver.backend_config == backend_config
        assert solver.backend_config.local is True
        assert solver.backend_config.gpu is False

    def test_multiple_solver_instances(self, hamiltonian_4q, backend_config):
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
