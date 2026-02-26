"""Tests for quantum_pipeline.structures.vqe_observation dataclasses."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from quantum_pipeline.structures.vqe_observation import (
    VQEDecoratedResult,
    VQEInitialData,
    VQEProcess,
    VQEResult,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_initial_data():
    """Minimal VQEInitialData for reuse across tests."""
    return VQEInitialData(
        backend='aer_simulator',
        num_qubits=4,
        hamiltonian=np.array([[1, 0], [0, -1]]),
        num_parameters=8,
        initial_parameters=np.zeros(8),
        noise_backend='undef',
        optimizer='COBYLA',
        ansatz=MagicMock(),
        ansatz_reps=3,
        default_shots=1024,
    )


@pytest.fixture
def sample_vqe_process():
    """Single VQEProcess entry."""
    return VQEProcess(
        iteration=1,
        parameters=np.array([0.1, 0.2]),
        result=np.float64(-1.5),
        std=np.float64(0.01),
    )


@pytest.fixture
def sample_vqe_result(sample_initial_data, sample_vqe_process):
    """VQEResult built from sample data."""
    return VQEResult(
        initial_data=sample_initial_data,
        iteration_list=[sample_vqe_process],
        minimum=np.float64(-1.5),
        optimal_parameters=np.array([0.1, 0.2]),
        maxcv=None,
        minimization_time=np.float64(1.23),
    )


@pytest.fixture
def sample_molecule():
    """Mock MoleculeInfo with minimal interface."""
    mol = MagicMock()
    mol.symbols = ['H', 'H']
    return mol


@pytest.fixture
def sample_decorated_result(sample_vqe_result, sample_molecule):
    """VQEDecoratedResult with all required fields."""
    return VQEDecoratedResult(
        vqe_result=sample_vqe_result,
        molecule=sample_molecule,
        basis_set='sto-3g',
        hamiltonian_time=np.float64(0.5),
        mapping_time=np.float64(0.3),
        vqe_time=np.float64(1.0),
        total_time=np.float64(1.8),
        molecule_id=0,
    )


# ---------------------------------------------------------------------------
# VQEProcess
# ---------------------------------------------------------------------------


class TestVQEProcess:
    """Tests for the VQEProcess dataclass."""

    def test_creation(self, sample_vqe_process):
        assert sample_vqe_process.iteration == 1
        assert sample_vqe_process.result == np.float64(-1.5)
        assert sample_vqe_process.std == np.float64(0.01)
        np.testing.assert_array_equal(sample_vqe_process.parameters, [0.1, 0.2])

    @pytest.mark.parametrize(
        'iteration,result,std',
        [
            (0, np.float64(0.0), np.float64(0.0)),
            (999, np.float64(-100.5), np.float64(5.5)),
            (1, np.float64(1e-15), np.float64(1e-20)),
        ],
        ids=['zeros', 'large-values', 'tiny-values'],
    )
    def test_various_values(self, iteration, result, std):
        proc = VQEProcess(
            iteration=iteration,
            parameters=np.array([1.0]),
            result=result,
            std=std,
        )
        assert proc.iteration == iteration
        assert proc.result == result
        assert proc.std == std


# ---------------------------------------------------------------------------
# VQEInitialData
# ---------------------------------------------------------------------------


class TestVQEInitialData:
    """Tests for the VQEInitialData dataclass."""

    def test_creation(self, sample_initial_data):
        assert sample_initial_data.backend == 'aer_simulator'
        assert sample_initial_data.num_qubits == 4
        assert sample_initial_data.num_parameters == 8
        assert sample_initial_data.noise_backend == 'undef'
        assert sample_initial_data.optimizer == 'COBYLA'
        assert sample_initial_data.ansatz_reps == 3
        assert sample_initial_data.default_shots == 1024

    def test_hamiltonian_is_numpy(self, sample_initial_data):
        assert isinstance(sample_initial_data.hamiltonian, np.ndarray)

    def test_initial_parameters_shape(self, sample_initial_data):
        assert sample_initial_data.initial_parameters.shape == (8,)


# ---------------------------------------------------------------------------
# VQEResult
# ---------------------------------------------------------------------------


class TestVQEResult:
    """Tests for the VQEResult dataclass."""

    def test_creation(self, sample_vqe_result, sample_initial_data):
        assert sample_vqe_result.initial_data is sample_initial_data
        assert sample_vqe_result.minimum == np.float64(-1.5)
        assert sample_vqe_result.maxcv is None
        assert sample_vqe_result.minimization_time == np.float64(1.23)

    def test_iteration_list_length(self, sample_vqe_result):
        assert len(sample_vqe_result.iteration_list) == 1

    def test_maxcv_with_value(self, sample_initial_data, sample_vqe_process):
        result = VQEResult(
            initial_data=sample_initial_data,
            iteration_list=[sample_vqe_process],
            minimum=np.float64(-1.0),
            optimal_parameters=np.array([0.5]),
            maxcv=np.float64(0.001),
            minimization_time=np.float64(2.0),
        )
        assert result.maxcv == np.float64(0.001)


# ---------------------------------------------------------------------------
# VQEDecoratedResult
# ---------------------------------------------------------------------------


class TestVQEDecoratedResult:
    """Tests for the VQEDecoratedResult dataclass."""

    def test_creation(self, sample_decorated_result):
        assert sample_decorated_result.basis_set == 'sto-3g'
        assert sample_decorated_result.molecule_id == 0
        assert sample_decorated_result.total_time == np.float64(1.8)

    def test_optional_performance_defaults_to_none(self, sample_decorated_result):
        assert sample_decorated_result.performance_start is None
        assert sample_decorated_result.performance_end is None

    def test_performance_fields_can_be_set(self, sample_vqe_result, sample_molecule):
        perf_start = {'system': {'cpu': {'percent': 10}}}
        perf_end = {'system': {'cpu': {'percent': 50}}}
        decorated = VQEDecoratedResult(
            vqe_result=sample_vqe_result,
            molecule=sample_molecule,
            basis_set='sto-3g',
            hamiltonian_time=np.float64(0.5),
            mapping_time=np.float64(0.3),
            vqe_time=np.float64(1.0),
            total_time=np.float64(1.8),
            molecule_id=0,
            performance_start=perf_start,
            performance_end=perf_end,
        )
        assert decorated.performance_start is perf_start
        assert decorated.performance_end is perf_end


class TestGetResultSuffix:
    """Tests for VQEDecoratedResult.get_result_suffix."""

    def test_suffix_with_single_iteration(self, sample_decorated_result):
        assert sample_decorated_result.get_result_suffix() == '_it1'

    def test_suffix_with_multiple_iterations(self, sample_decorated_result):
        extra = VQEProcess(
            iteration=2,
            parameters=np.array([0.3, 0.4]),
            result=np.float64(-1.6),
            std=np.float64(0.005),
        )
        sample_decorated_result.vqe_result.iteration_list.append(extra)
        assert sample_decorated_result.get_result_suffix() == '_it2'


class TestGetSchemaSuffix:
    """Tests for VQEDecoratedResult.get_schema_suffix."""

    def test_schema_suffix_format(self, sample_decorated_result):
        suffix = sample_decorated_result.get_schema_suffix()
        assert suffix.startswith('_mol0')
        assert '_HH_' in suffix
        assert '_it1_' in suffix
        assert '_bs_sto_3g_' in suffix
        assert '_bk_aer_simulator' in suffix

    def test_schema_suffix_replaces_hyphens(self, sample_vqe_result, sample_molecule):
        sample_vqe_result.initial_data.backend = 'ibm-brisbane'
        decorated = VQEDecoratedResult(
            vqe_result=sample_vqe_result,
            molecule=sample_molecule,
            basis_set='6-31g',
            hamiltonian_time=np.float64(0.5),
            mapping_time=np.float64(0.3),
            vqe_time=np.float64(1.0),
            total_time=np.float64(1.8),
            molecule_id=2,
        )
        suffix = decorated.get_schema_suffix()
        assert '-' not in suffix.split('_mol')[1]
        assert '_bs_6_31g_' in suffix
        assert '_bk_ibm_brisbane' in suffix


class TestGetPerformanceDelta:
    """Tests for VQEDecoratedResult.get_performance_delta."""

    def test_empty_when_no_performance_data(self, sample_decorated_result):
        assert sample_decorated_result.get_performance_delta() == {}

    def test_empty_when_only_start(self, sample_decorated_result):
        sample_decorated_result.performance_start = {'system': {}}
        assert sample_decorated_result.get_performance_delta() == {}

    def test_cpu_delta(self, sample_vqe_result, sample_molecule):
        decorated = VQEDecoratedResult(
            vqe_result=sample_vqe_result,
            molecule=sample_molecule,
            basis_set='sto-3g',
            hamiltonian_time=np.float64(0.5),
            mapping_time=np.float64(0.3),
            vqe_time=np.float64(1.0),
            total_time=np.float64(1.8),
            molecule_id=0,
            performance_start={'system': {'cpu': {'percent': 10}, 'memory': {}}},
            performance_end={'system': {'cpu': {'percent': 60}, 'memory': {}}},
        )
        delta = decorated.get_performance_delta()
        assert delta['cpu_usage_delta'] == 50
        assert delta['vqe_total_time'] == 1.8

    def test_memory_delta(self, sample_vqe_result, sample_molecule):
        mem_start = 2 * 1024**3  # 2 GB
        mem_end = 4 * 1024**3  # 4 GB
        decorated = VQEDecoratedResult(
            vqe_result=sample_vqe_result,
            molecule=sample_molecule,
            basis_set='sto-3g',
            hamiltonian_time=np.float64(0.5),
            mapping_time=np.float64(0.3),
            vqe_time=np.float64(1.0),
            total_time=np.float64(1.8),
            molecule_id=0,
            performance_start={'system': {'cpu': {}, 'memory': {'used': mem_start}}},
            performance_end={'system': {'cpu': {}, 'memory': {'used': mem_end}}},
        )
        delta = decorated.get_performance_delta()
        assert delta['memory_usage_delta'] == mem_end - mem_start
        assert delta['memory_usage_delta_gb'] == pytest.approx(2.0)

    def test_gpu_delta(self, sample_vqe_result, sample_molecule):
        decorated = VQEDecoratedResult(
            vqe_result=sample_vqe_result,
            molecule=sample_molecule,
            basis_set='sto-3g',
            hamiltonian_time=np.float64(0.5),
            mapping_time=np.float64(0.3),
            vqe_time=np.float64(1.0),
            total_time=np.float64(1.8),
            molecule_id=0,
            performance_start={
                'system': {'cpu': {}, 'memory': {}},
                'gpu': [{'utilization_gpu': 10, 'utilization_memory': 20, 'power_draw': 50}],
            },
            performance_end={
                'system': {'cpu': {}, 'memory': {}},
                'gpu': [{'utilization_gpu': 80, 'utilization_memory': 60, 'power_draw': 150}],
            },
        )
        delta = decorated.get_performance_delta()
        assert 'gpu_deltas' in delta
        gpu = delta['gpu_deltas'][0]
        assert gpu['utilization_gpu_delta'] == 70
        assert gpu['utilization_memory_delta'] == 40
        assert gpu['power_draw_delta'] == 100
        assert gpu['gpu_index'] == 0

    def test_container_type_from_start(self, sample_vqe_result, sample_molecule):
        decorated = VQEDecoratedResult(
            vqe_result=sample_vqe_result,
            molecule=sample_molecule,
            basis_set='sto-3g',
            hamiltonian_time=np.float64(0.5),
            mapping_time=np.float64(0.3),
            vqe_time=np.float64(1.0),
            total_time=np.float64(1.8),
            molecule_id=0,
            performance_start={'container_type': 'docker', 'system': {'cpu': {}, 'memory': {}}},
            performance_end={'system': {'cpu': {}, 'memory': {}}},
        )
        delta = decorated.get_performance_delta()
        assert delta['container_type'] == 'docker'

    def test_handles_missing_nested_keys_gracefully(self, sample_vqe_result, sample_molecule):
        decorated = VQEDecoratedResult(
            vqe_result=sample_vqe_result,
            molecule=sample_molecule,
            basis_set='sto-3g',
            hamiltonian_time=np.float64(0.5),
            mapping_time=np.float64(0.3),
            vqe_time=np.float64(1.0),
            total_time=np.float64(1.8),
            molecule_id=0,
            performance_start={'system': {}},
            performance_end={'system': {}},
        )
        delta = decorated.get_performance_delta()
        assert 'cpu_usage_delta' not in delta
        assert 'memory_usage_delta' not in delta
        assert delta['vqe_total_time'] == 1.8
