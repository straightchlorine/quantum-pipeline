"""Tests for the shared ML input contract."""

import pytest

from quantum_pipeline.ml.schema import (
    FIRST_ITERATION,
    RUN_KEY,
    default_molecules,
    require_columns,
)


def test_run_key_matches_the_iceberg_table() -> None:
    """ml_iteration_features keys on experiment_id; there is no run_id column."""
    assert RUN_KEY == 'experiment_id'


def test_iterations_are_one_indexed() -> None:
    """VQESolver.current_iter starts at 1 and the Spark job divides by iteration_step."""
    assert FIRST_ITERATION == 1


def test_default_molecules_are_isolated_copies() -> None:
    first = default_molecules()
    first[0]['num_qubits'] = 999

    assert default_molecules()[0]['num_qubits'] != 999


def test_molecules_span_a_range_of_sizes() -> None:
    """Fixtures only need distinct sizes; the values carry no physical claim."""
    qubits = [m['num_qubits'] for m in default_molecules()]

    assert len(set(qubits)) == len(qubits), 'sizes must differ to exercise the models'
    assert all(q > 0 for q in qubits)


def test_require_columns_names_every_missing_column() -> None:
    with pytest.raises(ValueError) as exc:
        require_columns(['energy'], {'experiment_id', 'energy', 'iteration_step'}, 'df_iter')

    message = str(exc.value)
    assert 'df_iter' in message
    assert 'experiment_id' in message
    assert 'iteration_step' in message


def test_require_columns_passes_when_satisfied() -> None:
    require_columns(['experiment_id', 'energy', 'extra'], {'experiment_id', 'energy'}, 'df')
