from unittest.mock import Mock, patch

import pytest

from quantum_pipeline.cli import execute_simulation, main
from quantum_pipeline.configs.defaults import DEFAULTS
from quantum_pipeline.solvers.vqe_solver import InvalidSolverConfigError


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


def test_max_evaluations_is_passed_on_with_and_without_convergence():
    """The ceiling reaches the runner on both paths; --convergence must not drop it."""
    assert _run(max_evaluations=1234)['max_evaluations'] == 1234
    assert _run(convergence=True, max_evaluations=1234)['max_evaluations'] == 1234


def test_default_shots_none_runs_exact():
    """Without --shots the runner gets default_shots=None, the exact path."""
    assert _run(shots=DEFAULTS['shots'])['default_shots'] is None


def test_explicit_shots_are_passed_on():
    assert _run(shots=512)['default_shots'] == 512


def test_exact_overrides_shots():
    assert _run(shots=512, exact=True)['default_shots'] is None


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


def test_invalid_solver_config_exits_cleanly(monkeypatch, capsys):
    """A bad flag combination (gradient optimizer with sampled shots) is a usage error: exit 2
    with the message on stderr, not a Python traceback."""
    monkeypatch.setattr('sys.argv', ['quantum-pipeline', '-f', 'molecules.json'])
    message = 'Optimizer L-BFGS-B needs a gradient, but default_shots=1024 samples the objective.'
    with (
        patch('quantum_pipeline.cli.QuantumPipelineArgParser.get_config', return_value={}),
        patch(
            'quantum_pipeline.cli.execute_simulation',
            side_effect=InvalidSolverConfigError(message),
        ),
        pytest.raises(SystemExit) as excinfo,
    ):
        main()
    assert excinfo.value.code == 2
    assert message in capsys.readouterr().err


def test_missing_molecule_file_exits_cleanly(monkeypatch, capsys, tmp_path):
    """A --file path that does not exist is a usage error (exit 2), not a FileNotFoundError
    traceback from the molecule loader."""
    missing = tmp_path / 'molecules.json'
    monkeypatch.setattr('sys.argv', ['quantum-pipeline', '-f', str(missing)])
    with (
        patch(
            'quantum_pipeline.cli.QuantumPipelineArgParser.get_config',
            return_value={'file': str(missing)},
        ),
        patch('quantum_pipeline.cli.execute_simulation') as execute,
        pytest.raises(SystemExit) as excinfo,
    ):
        main()
    assert excinfo.value.code == 2
    execute.assert_not_called()
    assert 'no such file' in capsys.readouterr().err
