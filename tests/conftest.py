"""Shared fixtures for config dataclasses and monitoring state."""

import tempfile
from pathlib import Path

import pytest

from quantum_pipeline.configs.module.backend import BackendConfig
from quantum_pipeline.configs.module.producer import ProducerConfig
from quantum_pipeline.configs.module.security import SecurityConfig


@pytest.fixture
def sample_backend_config() -> BackendConfig:
    """Local statevector simulator, no GPU or noise."""
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
def sample_security_config() -> SecurityConfig:
    """The shipped default, not a hand-rolled config, so tests follow DEFAULTS."""
    return SecurityConfig.get_default()


@pytest.fixture
def sample_producer_config(sample_security_config: SecurityConfig) -> ProducerConfig:
    """Minimal ProducerConfig wired to the sample security config."""
    return ProducerConfig(
        servers='localhost:9092',
        topic='test-topic',
        security=sample_security_config,
    )


@pytest.fixture
def clean_global_monitor():
    """Reset the PerformanceMonitor singleton so monitoring state does not leak between tests."""
    from quantum_pipeline.monitoring import performance_monitor

    original_monitor = performance_monitor._global_monitor
    performance_monitor._global_monitor = None
    yield
    performance_monitor._global_monitor = original_monitor


@pytest.fixture
def temp_metrics_dir() -> Path:
    """Temporary directory for metrics file output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
