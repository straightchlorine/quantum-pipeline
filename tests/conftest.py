"""Shared test fixtures for the quantum-pipeline test suite.

Provides reusable fixtures for configuration dataclasses and monitoring
state management, reducing duplication across test modules.
"""

import tempfile
from pathlib import Path

import pytest

from quantum_pipeline.configs.module.backend import BackendConfig
from quantum_pipeline.configs.module.producer import ProducerConfig
from quantum_pipeline.configs.module.security import SecurityConfig


@pytest.fixture
def sample_backend_config() -> BackendConfig:
    """Return a default BackendConfig suitable for unit tests.

    Uses sensible test defaults (local simulator, no GPU, no noise)
    so that tests needing a BackendConfig can rely on a consistent
    baseline without constructing one manually each time.
    """
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
    """Return the default SecurityConfig from project defaults.

    Delegates to ``SecurityConfig.get_default()`` so the fixture
    stays in sync with production defaults automatically.
    """
    return SecurityConfig.get_default()


@pytest.fixture
def sample_producer_config(sample_security_config: SecurityConfig) -> ProducerConfig:
    """Return a default ProducerConfig wired to the sample security config.

    Provides a minimal but complete ProducerConfig that can be used
    in any test requiring Kafka producer configuration.
    """
    return ProducerConfig(
        servers='localhost:9092',
        topic='test-topic',
        security=sample_security_config,
    )


@pytest.fixture
def clean_global_monitor():
    """Reset the global PerformanceMonitor singleton before and after a test.

    Saves the current ``_global_monitor`` reference, sets it to ``None``
    for the duration of the test, and restores the original value on
    teardown.  This prevents cross-test pollution of monitoring state.
    """
    from quantum_pipeline.monitoring import performance_monitor

    original_monitor = performance_monitor._global_monitor
    performance_monitor._global_monitor = None
    yield
    performance_monitor._global_monitor = original_monitor


@pytest.fixture
def temp_metrics_dir() -> Path:
    """Provide a temporary directory for metrics file output.

    The directory is automatically cleaned up after the test completes.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
