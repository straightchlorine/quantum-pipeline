from unittest.mock import Mock, patch

from quantum_pipeline.cli import execute_simulation
from quantum_pipeline.configs.defaults import DEFAULTS


def _kwargs(**overrides):
    kwargs = {
        'file': 'molecules.json',
        'basis': 'sto3g',
        'max_iterations': DEFAULTS['max_iterations'],
        'convergence': False,
        'threshold': 1e-6,
        'optimizer': 'COBYLA',
        'ansatz_reps': 3,
        'shots': 1024,
        'report': False,
        'kafka': False,
        'kafka_config': Mock(),
        'backend_config': Mock(),
    }
    kwargs.update(overrides)
    return kwargs


def _run(**overrides):
    with patch('quantum_pipeline.cli.VQERunner') as mock_runner:
        execute_simulation(**_kwargs(**overrides))
    return mock_runner.call_args.kwargs


def test_convergence_clears_explicit_max_iterations():
    """VQESolver rejects both limits at once, so an explicit cap must be dropped, not passed."""
    call = _run(convergence=True, max_iterations=500)
    assert call['max_iterations'] is None
    assert call['convergence_threshold'] == 1e-6


def test_convergence_with_default_max_iterations_clears_it_too():
    call = _run(convergence=True)
    assert call['max_iterations'] is None
    assert call['convergence_threshold'] == 1e-6


def test_without_convergence_max_iterations_is_kept():
    call = _run(max_iterations=500)
    assert call['max_iterations'] == 500
    assert call['convergence_threshold'] is None


def test_kafka_servers_env_ignored_when_kafka_disabled(monkeypatch):
    """kafka_config is None with --kafka off, so the env override must not be applied."""
    monkeypatch.setenv('KAFKA_SERVERS', 'broker:9092')

    call = _run(kafka=False, kafka_config=None)

    assert call['kafka_config'] is None


def test_kafka_servers_env_overrides_config_when_kafka_enabled(monkeypatch):
    monkeypatch.setenv('KAFKA_SERVERS', 'broker:9092')
    kafka_config = Mock()

    call = _run(kafka=True, kafka_config=kafka_config)

    assert call['kafka_config'] is kafka_config
    assert kafka_config.servers == 'broker:9092'
