"""Coverage-focused tests for quantum_pipeline.solvers.vqe_solver.

Targets uncovered paths in VQESolver:
- via_aer: full flow with mocked qiskit/scipy, all logging branches
- via_ibmq: full flow with Session context manager, all logging branches
- solve: dispatch to via_aer vs via_ibmq, state reset
- _optimize_circuits: pass-manager + layout
- Convergence logging branches (both/convergence-only/max-only/neither,
  success/failure)
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.backends.aer_simulator import AerBackend

from quantum_pipeline.configs.module.backend import BackendConfig
from quantum_pipeline.solvers.vqe_solver import VQESolver
from quantum_pipeline.structures.vqe_observation import VQEResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
    """Return (estimator_mock, estimator_instance_mock)."""
    estimator_instance = MagicMock()
    mock_result = MagicMock()
    mock_result.data.evs = [np.float64(-0.5)]
    mock_result.data.stds = [np.float64(0.01)]
    estimator_instance.run.return_value.result.return_value = [mock_result]
    return estimator_instance


# ---------------------------------------------------------------------------
# via_aer — full flow
# ---------------------------------------------------------------------------


def _setup_aer_mocks(mock_su2, mock_pm_gen, mock_estimator_cls, hamiltonian):
    """Shared mock wiring for via_aer tests."""
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


# ---------------------------------------------------------------------------
# via_ibmq — full flow
# ---------------------------------------------------------------------------


def _setup_ibmq_mocks(mock_su2, mock_pm_gen, mock_estimator_cls, mock_session_cls, hamiltonian):
    """Shared mock wiring for via_ibmq tests."""
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


# ---------------------------------------------------------------------------
# solve — dispatch + state reset
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _optimize_circuits
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# VQEResult fields — maxcv absent in OptimizeResult
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Minimize tolerance propagation
# ---------------------------------------------------------------------------


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
