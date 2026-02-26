"""Tests for quantum_pipeline.solvers.solver — the Solver base class.

Covers:
- __init__ / logger setup
- supported_optimizers_prompt
- __validate_env (all branches: missing creds, invalid channel, valid channels)
- _get_service (success, RuntimeError propagation)
- _get_noise_model (success, failure)
- get_backend: local/CPU, local/GPU, local+noise, remote with filters,
  remote least-busy, remote failure, missing backend_config
"""

import os
from typing import ClassVar
from unittest.mock import MagicMock, patch

import pytest

from quantum_pipeline.configs.module.backend import BackendConfig
from quantum_pipeline.solvers.solver import Solver

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def solver():
    """Return a bare Solver instance (no backend_config set yet)."""
    return Solver()


@pytest.fixture
def local_cpu_config():
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
def local_gpu_config():
    return BackendConfig(
        local=True,
        gpu=True,
        optimization_level=2,
        min_num_qubits=4,
        filters=None,
        simulation_method='statevector',
        gpu_opts={'device': 'GPU', 'cuStateVec_enable': True},
        noise=None,
    )


@pytest.fixture
def local_noise_config():
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
def remote_config_with_filters():
    return BackendConfig(
        local=False,
        gpu=False,
        optimization_level=2,
        min_num_qubits=4,
        filters=lambda b: True,
        simulation_method=None,
        gpu_opts=None,
        noise=None,
    )


@pytest.fixture
def remote_config_no_filters():
    return BackendConfig(
        local=False,
        gpu=False,
        optimization_level=2,
        min_num_qubits=None,
        filters=None,
        simulation_method=None,
        gpu_opts=None,
        noise=None,
    )


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestSolverInit:
    def test_logger_is_created(self, solver):
        assert solver.logger is not None
        assert solver.logger.name == 'Solver'

    def test_subclass_logger_name(self):
        class MySolver(Solver):
            pass

        s = MySolver()
        assert s.logger.name == 'MySolver'


# ---------------------------------------------------------------------------
# supported_optimizers_prompt
# ---------------------------------------------------------------------------


class TestSupportedOptimizersPrompt:
    def test_returns_string(self, solver):
        result = solver.supported_optimizers_prompt()
        assert isinstance(result, str)

    def test_contains_all_optimizers(self, solver):
        from quantum_pipeline.configs.settings import SUPPORTED_OPTIMIZERS

        result = solver.supported_optimizers_prompt()
        for opt in SUPPORTED_OPTIMIZERS:
            assert opt in result

    def test_contains_descriptions(self, solver):
        from quantum_pipeline.configs.settings import SUPPORTED_OPTIMIZERS

        result = solver.supported_optimizers_prompt()
        for desc in SUPPORTED_OPTIMIZERS.values():
            assert desc in result


# ---------------------------------------------------------------------------
# __validate_env  (name-mangled → _Solver__validate_env)
# ---------------------------------------------------------------------------


class TestValidateEnv:
    """Exercise every branch of __validate_env."""

    VALID_ENV: ClassVar[dict] = {
        'IBM_RUNTIME_CHANNEL': 'ibm_quantum',
        'IBM_RUNTIME_INSTANCE': 'ibm-q/open/main',
        'IBM_RUNTIME_TOKEN': 'abc123',
    }

    @pytest.mark.parametrize(
        'missing_key',
        [
            'IBM_RUNTIME_CHANNEL',
            'IBM_RUNTIME_INSTANCE',
            'IBM_RUNTIME_TOKEN',
        ],
    )
    def test_missing_env_var_raises(self, solver, missing_key):
        env = {k: v for k, v in self.VALID_ENV.items() if k != missing_key}
        with patch.dict(os.environ, env, clear=True), pytest.raises(RuntimeError):
            solver._Solver__validate_env()

    def test_all_env_vars_empty_raises(self, solver):
        env = dict.fromkeys(self.VALID_ENV, '')
        with patch.dict(os.environ, env, clear=True), pytest.raises(RuntimeError):
            solver._Solver__validate_env()

    @pytest.mark.parametrize('bad_channel', ['ibm_fake', '', 'quantum', 'LOCAL'])
    def test_invalid_channel_raises(self, solver, bad_channel):
        env = {**self.VALID_ENV, 'IBM_RUNTIME_CHANNEL': bad_channel}
        with patch.dict(os.environ, env, clear=True), pytest.raises(RuntimeError):
            solver._Solver__validate_env()

    @pytest.mark.parametrize('channel', ['ibm_quantum', 'ibm_cloud', 'local'])
    def test_valid_channel_returns_tuple(self, solver, channel):
        env = {**self.VALID_ENV, 'IBM_RUNTIME_CHANNEL': channel}
        with patch.dict(os.environ, env, clear=True):
            ch, inst, tok = solver._Solver__validate_env()
            assert ch == channel
            assert inst == self.VALID_ENV['IBM_RUNTIME_INSTANCE']
            assert tok == self.VALID_ENV['IBM_RUNTIME_TOKEN']


# ---------------------------------------------------------------------------
# _get_service
# ---------------------------------------------------------------------------


class TestGetService:
    VALID_ENV = TestValidateEnv.VALID_ENV

    @patch('quantum_pipeline.solvers.solver.QiskitRuntimeService')
    def test_success(self, mock_qrs, solver):
        with patch.dict(os.environ, self.VALID_ENV, clear=True):
            service = solver._get_service()
            mock_qrs.assert_called_once_with(
                channel='ibm_quantum',
                instance='ibm-q/open/main',
                token='abc123',
            )
            assert service == mock_qrs.return_value

    def test_raises_on_validation_failure(self, solver):
        with patch.dict(os.environ, {}, clear=True), pytest.raises(RuntimeError):
            solver._get_service()

    @patch('quantum_pipeline.solvers.solver.QiskitRuntimeService', side_effect=Exception('boom'))
    def test_raises_on_connection_failure(self, mock_qrs, solver):
        with patch.dict(os.environ, self.VALID_ENV, clear=True), pytest.raises(RuntimeError):
            solver._get_service()


# ---------------------------------------------------------------------------
# _get_noise_model
# ---------------------------------------------------------------------------


class TestGetNoiseModel:
    @patch('quantum_pipeline.solvers.solver.NoiseModel')
    @patch.object(Solver, '_get_service')
    def test_success(self, mock_get_service, mock_noise_model_cls, solver):
        mock_provider = MagicMock()
        mock_get_service.return_value = mock_provider
        mock_backend_obj = MagicMock()
        mock_provider.get_backend.return_value = mock_backend_obj
        mock_noise = MagicMock()
        mock_noise_model_cls.from_backend.return_value = mock_noise

        result = solver._get_noise_model('ibm_brisbane')

        mock_provider.get_backend.assert_called_once_with('ibm_brisbane')
        mock_noise_model_cls.from_backend.assert_called_once_with(mock_backend_obj)
        assert result == mock_noise

    @patch.object(Solver, '_get_service')
    def test_failure_raises(self, mock_get_service, solver):
        mock_provider = MagicMock()
        mock_get_service.return_value = mock_provider
        mock_provider.get_backend.side_effect = Exception('not found')

        with pytest.raises(RuntimeError):
            solver._get_noise_model('bad_backend')


# ---------------------------------------------------------------------------
# get_backend
# ---------------------------------------------------------------------------


class TestGetBackend:
    """Test get_backend branching: local/CPU, local/GPU, local+noise,
    remote with filters, remote no filters, remote failure, no config."""

    @patch('quantum_pipeline.solvers.solver.AerSimulator')
    def test_local_cpu(self, mock_aer, solver, local_cpu_config):
        solver.backend_config = local_cpu_config
        backend = solver.get_backend()
        mock_aer.assert_called_once_with(
            method='statevector',
            noise_model=None,
        )
        assert backend == mock_aer.return_value

    @patch('quantum_pipeline.solvers.solver.AerSimulator')
    def test_local_gpu(self, mock_aer, solver, local_gpu_config):
        solver.backend_config = local_gpu_config
        backend = solver.get_backend()
        mock_aer.assert_called_once_with(
            method='statevector',
            device='GPU',
            cuStateVec_enable=True,
            noise_model=None,
        )
        assert backend == mock_aer.return_value

    @patch('quantum_pipeline.solvers.solver.AerSimulator')
    @patch.object(Solver, '_get_noise_model')
    def test_local_with_noise(self, mock_noise, mock_aer, solver, local_noise_config):
        fake_noise = MagicMock()
        mock_noise.return_value = fake_noise
        solver.backend_config = local_noise_config

        backend = solver.get_backend()

        mock_noise.assert_called_once_with('ibm_brisbane')
        mock_aer.assert_called_once_with(
            method='statevector',
            noise_model=fake_noise,
        )
        assert backend == mock_aer.return_value

    @patch.object(Solver, '_get_service')
    def test_remote_with_filters(self, mock_svc, solver, remote_config_with_filters):
        solver.backend_config = remote_config_with_filters
        mock_service = MagicMock()
        mock_svc.return_value = mock_service
        mock_backend = MagicMock()
        mock_backend.name = 'ibm_kyoto'
        mock_service.least_busy.return_value = mock_backend

        backend = solver.get_backend()

        mock_service.least_busy.assert_called_once()
        call_kwargs = mock_service.least_busy.call_args
        assert call_kwargs[1]['operational'] is True
        assert backend == mock_backend

    @patch.object(Solver, '_get_service')
    def test_remote_no_filters(self, mock_svc, solver, remote_config_no_filters):
        solver.backend_config = remote_config_no_filters
        mock_service = MagicMock()
        mock_svc.return_value = mock_service
        mock_backend = MagicMock()
        mock_backend.name = 'ibm_least_busy'
        mock_service.least_busy.return_value = mock_backend

        backend = solver.get_backend()

        mock_service.least_busy.assert_called_once_with(operational=True)
        assert backend == mock_backend

    @patch.object(Solver, '_get_service')
    def test_remote_failure(self, mock_svc, solver, remote_config_no_filters):
        solver.backend_config = remote_config_no_filters
        mock_service = MagicMock()
        mock_svc.return_value = mock_service
        mock_service.least_busy.side_effect = Exception('unavailable')

        with pytest.raises(RuntimeError):
            solver.get_backend()

    def test_no_backend_config_raises(self, solver):
        solver.backend_config = None
        with pytest.raises(RuntimeError):
            solver.get_backend()
