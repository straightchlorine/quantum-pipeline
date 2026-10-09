"""Tests for quantum_ml_feature_processing.py - window features, summary and guards.

Only the pure DataFrame functions are exercised, on hand-built in-memory frames for the 5
source tables (no Iceberg locally). Covers:
- golden values of the window features on a 6-step run and a 1-step run
- no look-ahead: row k of a truncated run equals row k of the full run (run-level labels excepted)
- build_run_summary aggregates
- validation: fan-out / uniqueness guards raise, NULL guard drops rows and logs experiment_ids
"""

import logging
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

_AIRFLOW = Path(__file__).parent.parent.parent / 'docker' / 'airflow'
# `common` (pipeline_config, spark_factory) lives in docker/airflow; the script itself in scripts/
sys.path.insert(0, str(_AIRFLOW))
sys.path.insert(0, str(_AIRFLOW / 'scripts'))

import quantum_ml_feature_processing as mlf  # noqa: E402


def _java_17_available() -> bool:
    """Return True if Java 17+ is available (required by the installed PySpark)."""
    import shutil
    import subprocess

    java = shutil.which('java')
    if java is None:
        return False
    try:
        result = subprocess.run(  # noqa: S603
            [java, '-version'],
            capture_output=True,
            text=True,
            timeout=5,
        )
        output = result.stderr + result.stdout
        for major in range(17, 30):
            if f'version "{major}.' in output or f'version "{major}"' in output:
                return True
        return False
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _java_17_available(),
    reason='Spark unavailable in this environment (requires Java 17+)',
)

# Columns that describe the whole run and are repeated on every iteration row.
RUN_LEVEL_LABELS = {
    'convergence_iteration',
    'total_iterations',
    'final_energy',
    'converged',
    'relative_iteration',
    'vqe_time',
    'total_time',
}

TODAY = date(2026, 1, 2)

# Run 'a': 6 steps. Step 3 ties the running best (-1.5), so it is NOT a new minimum.
ENERGY_A = [-1.0, -1.5, -1.5, -1.2, -1.8, -1.8]
CUMMIN_A = [-1.0, -1.5, -1.5, -1.5, -1.8, -1.8]  # solver value, includes the current step
DELTA_A = [None, -0.5, 0.0, 0.3, -0.6, 0.0]
PARAM_A = [None, 0.1, 0.2, 0.1, 0.3, 0.05]
# Run 'b': a single iteration.
ENERGY_B = [-0.5]

ITER_DDL = (
    'experiment_id string, iteration_step int, iteration_energy double, energy_std_dev double, '
    'energy_delta double, parameter_delta_norm double, cumulative_min_energy double'
)


@pytest.fixture(scope='session')
def spark():
    """Session-scoped local[1] SparkSession."""
    from pyspark.sql import SparkSession

    os.environ.setdefault('PYSPARK_PYTHON', sys.executable)
    session = (
        SparkSession.builder.master('local[1]')
        .appName('test_ml_feature_processing')
        .config('spark.sql.shuffle.partitions', '2')
        .config('spark.ui.enabled', 'false')
        .getOrCreate()
    )
    yield session
    session.stop()


def _iter_rows(exp_id, energies, cummin, deltas, params):
    return [
        (exp_id, i + 1, e, 0.01, deltas[i], params[i], cummin[i]) for i, e in enumerate(energies)
    ]


def _sources(spark, iter_rows=None, vqe_rows=None, perf_rows=None):
    """Build the 5 source tables for runs 'a' (6 steps) and 'b' (1 step)."""
    if iter_rows is None:
        iter_rows = _iter_rows('a', ENERGY_A, CUMMIN_A, DELTA_A, PARAM_A) + _iter_rows(
            'b', ENERGY_B, ENERGY_B, [None], [None]
        )
    if vqe_rows is None:
        vqe_rows = [
            ('a', 'L-BFGS-B', 2, 42, 1024, 4, 6, -1.8, True),
            ('b', 'COBYLA', 1, 7, 1024, 2, 1, -0.5, False),
        ]
    if perf_rows is None:
        perf_rows = [('a', 1.5, 2.0), ('b', 0.1, 0.2)]
    return {
        'vqe_iterations': spark.createDataFrame(iter_rows, ITER_DDL),
        'vqe_results': spark.createDataFrame(
            vqe_rows,
            'experiment_id string, optimizer string, ansatz_reps int, seed int, '
            'default_shots int, num_qubits int, total_iterations int, '
            'minimum_energy double, success boolean',
        ),
        'molecules': spark.createDataFrame(
            [('a', 'H2', ['H', 'H'], 0, 1), ('b', 'LiH', ['Li', 'H'], 0, 1)],
            'experiment_id string, molecule_name string, atom_symbols array<string>, '
            'charge int, multiplicity int',
        ),
        'performance_metrics': spark.createDataFrame(
            perf_rows, 'experiment_id string, vqe_time double, total_time double'
        ),
        'ansatz_info': spark.createDataFrame(
            [('a', 'sto3g', 'EfficientSU2', 'random'), ('b', 'sto3g', 'EfficientSU2', 'zeros')],
            'experiment_id string, basis_set string, ansatz_name string, init_strategy string',
        ),
    }


def _ids(spark, ids=('a', 'b')):
    return spark.createDataFrame([(i,) for i in ids], 'experiment_id string')


def _rows(df, exp_id):
    out = df.filter(df.experiment_id == exp_id).orderBy('iteration_step').collect()
    return [r.asDict() for r in out]


@pytest.fixture(scope='module')
def golden(spark):
    sources = _sources(spark)
    df = mlf.build_iteration_features(sources, _ids(spark), TODAY).cache()
    df.count()
    return sources, df


def test_golden_window_features(golden):
    _, df = golden
    a = _rows(df, 'a')

    assert [r['is_new_minimum'] for r in a] == [True, True, False, False, True, False]
    assert [r['steps_since_improvement'] for r in a] == [0, 0, 1, 2, 0, 1]
    # last step at which a new minimum was found, repeated on every row of the run
    assert {r['convergence_iteration'] for r in a} == {5}
    assert a[0]['processing_date'] == TODAY

    # step 1 has no previous point
    first = a[0]
    for name in (
        'energy_delta',
        'parameter_delta_norm',
        'mean_param_change',
        'energy_moving_std_5',
    ):
        assert first[name] is None, name

    # rolling std is the sample std (ddof=1) of the trailing window, current row included
    w = mlf.ML_ROLLING_WINDOW
    for i in range(1, len(ENERGY_A)):
        window = ENERGY_A[max(0, i - w + 1) : i + 1]
        assert a[i]['energy_moving_std_5'] == pytest.approx(np.std(window, ddof=1))
        assert a[i]['energy_moving_avg_5'] == pytest.approx(np.mean(window))


def test_mean_param_change_is_mean_of_present_deltas(spark):
    params = [None, 0.2, 0.3, 0.4, 0.5, 0.6]
    energies = [-1.0, -1.1, -1.2, -1.3, -1.4, -1.5]
    rows = _iter_rows('a', energies, energies, [None] + [-0.1] * 5, params)
    sources = _sources(spark, iter_rows=rows)
    df = mlf.build_iteration_features(sources, _ids(spark, ('a',)), TODAY)

    got = [r['mean_param_change'] for r in _rows(df, 'a')]
    assert got[0] is None
    assert got[1:] == pytest.approx([0.2, 0.25, 0.3, 0.35, 0.4])


def test_one_row_run_has_null_std_and_no_divide_error(golden):
    _, df = golden
    (b,) = _rows(df, 'b')

    assert b['energy_moving_std_5'] is None
    assert b['is_new_minimum'] is True
    assert b['steps_since_improvement'] == 0
    assert b['convergence_iteration'] == 1
    assert b['relative_iteration'] == pytest.approx(1.0)


def test_no_lookahead_in_non_label_columns(spark, golden):
    _, full = golden
    full_a = _rows(full, 'a')
    n = len(ENERGY_A)
    for k in range(1, n + 1):
        truncated_iters = _iter_rows('a', ENERGY_A[:k], CUMMIN_A[:k], DELTA_A[:k], PARAM_A[:k])
        src = _sources(spark, iter_rows=truncated_iters)
        trunc = _rows(mlf.build_iteration_features(src, _ids(spark, ['a']), TODAY), 'a')
        assert len(trunc) == k
        row_k, ref = trunc[k - 1], full_a[k - 1]
        for name in ref:
            if name in RUN_LEVEL_LABELS:
                continue
            assert (
                row_k[name] == pytest.approx(ref[name])
                if isinstance(ref[name], float)
                else row_k[name] == ref[name]
            ), f'{name} differs at step {k}'


def test_run_level_labels_do_look_ahead(spark, golden):
    """Sanity check of the label set: truncating the run does change convergence_iteration."""
    _, full = golden
    src = _sources(
        spark, iter_rows=_iter_rows('a', ENERGY_A[:3], CUMMIN_A[:3], DELTA_A[:3], PARAM_A[:3])
    )
    trunc = _rows(mlf.build_iteration_features(src, _ids(spark, ['a']), TODAY), 'a')
    assert trunc[-1]['convergence_iteration'] == 2
    assert _rows(full, 'a')[2]['convergence_iteration'] == 5


def test_run_summary_on_golden_frame(spark, golden):
    _, df = golden
    summary = {r['experiment_id']: r.asDict() for r in mlf.build_run_summary(df, TODAY).collect()}
    a, b = summary['a'], summary['b']

    assert a['num_new_minima'] == 3  # steps 1, 2 and 5; the tie at step 3 does not count
    assert a['longest_plateau'] == 2
    assert a['improvement_ratio'] == pytest.approx(3 / 6)
    assert a['first_10_energy_slope'] == pytest.approx(
        np.polyfit(np.arange(1, 7, dtype=float), ENERGY_A, 1)[0]
    )
    assert a['first_10_mean_energy'] == pytest.approx(np.mean(ENERGY_A))
    assert a['processing_date'] == TODAY

    # step 1 counts as a new minimum, so a one-row run has ratio 1
    assert b['num_new_minima'] == 1
    assert b['improvement_ratio'] == pytest.approx(1.0)
    assert b['first_10_energy_slope'] is None
    assert b['first_10_mean_energy'] == pytest.approx(-0.5)


def test_duplicated_vqe_results_row_raises_fan_out(spark):
    vqe_rows = [
        ('a', 'L-BFGS-B', 2, 42, 1024, 4, 6, -1.8, True),
        ('a', 'L-BFGS-B', 2, 42, 1024, 4, 6, -1.8, True),
        ('b', 'COBYLA', 1, 7, 1024, 2, 1, -0.5, False),
    ]
    src = _sources(spark, vqe_rows=vqe_rows)
    ids = _ids(spark)
    df = mlf.build_iteration_features(src, ids, TODAY)

    with pytest.raises(RuntimeError, match='fan-out'):
        mlf.validate_iteration_features(df, src, ids)


def test_duplicated_iteration_step_raises_uniqueness(spark):
    rows = _iter_rows('a', ENERGY_A, CUMMIN_A, DELTA_A, PARAM_A)
    rows.append(rows[-1])  # same (experiment_id, iteration_step) twice
    src = _sources(spark, iter_rows=rows)
    ids = _ids(spark, ['a'])
    df = mlf.build_iteration_features(src, ids, TODAY)

    with pytest.raises(RuntimeError, match='not unique'):
        mlf.validate_iteration_features(df, src, ids)


def test_missing_performance_row_is_dropped_and_logged(spark, caplog):
    src = _sources(spark, perf_rows=[('a', 1.5, 2.0)])  # 'b' has no performance_metrics row
    ids = _ids(spark)
    df = mlf.build_iteration_features(src, ids, TODAY)

    with caplog.at_level(logging.WARNING, logger=mlf.logger.name):
        clean = mlf.validate_iteration_features(df, src, ids)

    assert {r['experiment_id'] for r in clean.select('experiment_id').distinct().collect()} == {
        'a'
    }
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert 'b' in warnings[0].split('experiment_id(s):')[1]
    assert 'vqe_time' in warnings[0]


class _KeepOpen:
    """Delegates to the shared session but ignores `stop()`, which `main` always calls."""

    def __init__(self, session):
        self._session = session

    def __getattr__(self, name):
        return getattr(self._session, name)

    def stop(self):
        pass


def test_main_gates_each_table_on_its_own_ids(spark, monkeypatch):
    """Run 'a' is only missing from the summary, run 'b' only from the iteration table."""
    sources = _sources(spark)
    events, writes = [], []

    def fake_new_ids(_spark, _source_ids, target_table):
        events.append('ids')
        return ['b'] if target_table.endswith('ml_iteration_features') else ['a']

    def fake_write(_spark, df, table_name, _partition_columns):
        events.append('write')
        ids = sorted({r['experiment_id'] for r in df.select('experiment_id').collect()})
        writes.append((table_name, ids, df.count()))

    monkeypatch.setattr(mlf, 'create_spark_session', lambda _name: _KeepOpen(spark))
    monkeypatch.setattr(mlf, 'load_source_tables', lambda _spark: sources)
    monkeypatch.setattr(mlf, 'get_new_experiment_ids', fake_new_ids)
    monkeypatch.setattr(mlf, 'write_incremental', fake_write)

    mlf.main()

    # 'a' has 6 steps but one summary row; 'b' has 1 step. No id is written to a table twice.
    assert writes == [('ml_iteration_features', ['b'], 1), ('ml_run_summary', ['a'], 1)]
    # both id lists are collected before the first write
    assert events == ['ids', 'ids', 'write', 'write']
