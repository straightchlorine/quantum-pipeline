from typing import Any, Dict

import pytest

from quantum_pipeline.configs.module.backend import BackendConfig


@pytest.fixture
def sample_backend_defaults() -> Dict[str, Any]:
    """Fixture to provide sample default backend configurations."""
    return {
        'backend': {
            'local': True,
            'optimization_level': 3,
            'min_num_qubits': None,
            'filters': None,
            'gpu': False,
            'method': 'automatic',
            'gpu_opts': {'cuStateVec_enable': True},
            'noise_backend': None,
        }
    }


def test_backend_config_init():
    """Test initialization of BackendConfig with all parameters."""
    backend_config = BackendConfig(
        local=True,
        gpu=True,
        optimization_level=2,
        min_num_qubits=10,
        filters=None,
        simulation_method='statevector',
        gpu_opts={'cuStateVec_enable': True},
        noise='depolarizing',
    )

    assert backend_config.local is True
    assert backend_config.gpu is True
    assert backend_config.optimization_level == 2
    assert backend_config.min_num_qubits == 10
    assert backend_config.filters is None
    assert backend_config.simulation_method == 'statevector'
    assert backend_config.gpu_opts == {'cuStateVec_enable': True}
    assert backend_config.noise == 'depolarizing'


def test_backend_config_to_dict():
    """Test conversion of BackendConfig to dictionary."""
    backend_config = BackendConfig(
        local=True,
        gpu=True,
        optimization_level=2,
        min_num_qubits=10,
        filters=None,
        simulation_method='statevector',
        gpu_opts={'cuStateVec_enable': True},
        noise='depolarizing',
    )

    config_dict = backend_config.to_dict()

    assert config_dict['local'] is True
    assert config_dict['gpu'] is True
    assert config_dict['optimization_level'] == 2
    assert config_dict['min_num_qubits'] == 10
    assert config_dict['filters'] is None
    assert config_dict['simulation_method'] == 'statevector'
    assert config_dict['gpu_opts'] == {'cuStateVec_enable': True}
    assert config_dict['noise'] == 'depolarizing'


def test_backend_config_from_dict():
    """Test creating BackendConfig from a dictionary."""
    config_dict = {
        'local': False,
        'gpu': True,
        'optimization_level': 3,
        'min_num_qubits': 20,
        'simulation_method': 'matrix_product_state',
        'gpu_opts': {'cuStateVec_enable': True},
        'noise': 'thermal_relaxation',
    }

    backend_config = BackendConfig.from_dict(config_dict)

    assert backend_config.local is False
    assert backend_config.gpu is True
    assert backend_config.optimization_level == 3
    assert backend_config.min_num_qubits == 20
    assert backend_config.filters is None
    assert backend_config.simulation_method == 'matrix_product_state'
    assert backend_config.gpu_opts == {'cuStateVec_enable': True}
    assert backend_config.noise == 'thermal_relaxation'


def test_backend_config_from_empty_dict():
    """Test creating BackendConfig from an empty dictionary."""
    backend_config = BackendConfig.from_dict({})

    assert backend_config.local is None
    assert backend_config.gpu is None
    assert backend_config.optimization_level is None
    assert backend_config.min_num_qubits is None
    assert backend_config.filters is None
    assert backend_config.simulation_method is None
    assert backend_config.gpu_opts is None
    assert backend_config.noise is None


def test_backend_config_with_partial_dict():
    """Test creating BackendConfig with a partial dictionary."""
    config_dict = {
        'local': False,
        'optimization_level': 3,
    }

    backend_config = BackendConfig.from_dict(config_dict)

    assert backend_config.local is False
    assert backend_config.gpu is None
    assert backend_config.optimization_level == 3
    assert backend_config.min_num_qubits is None
    assert backend_config.filters is None
    assert backend_config.simulation_method is None
    assert backend_config.gpu_opts is None
    assert backend_config.noise is None


def test_default_backend_config(sample_backend_defaults):
    """Test the default_backend_config method."""

    default_config = BackendConfig.default_backend_config()

    assert default_config.local == sample_backend_defaults['backend']['local']
    assert (
        default_config.optimization_level
        == sample_backend_defaults['backend']['optimization_level']
    )
    assert default_config.min_num_qubits == sample_backend_defaults['backend']['min_num_qubits']
    assert default_config.filters == sample_backend_defaults['backend']['filters']
    assert default_config.gpu == sample_backend_defaults['backend']['gpu']
    assert default_config.simulation_method == sample_backend_defaults['backend']['method']
    assert default_config.noise == sample_backend_defaults['backend']['noise_backend']
