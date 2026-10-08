"""Tests for quantum_incremental_processing.py - Spark extraction logic.

Covers new and fixed fields added in QUA-18:
- molecule_name derived from atom symbols
- init_strategy / ansatz_name in ansatz_info and vqe_results
- seed / nuclear_repulsion_energy / success / nfev / nit in vqe_results
- posexplode-based parameter_index in iteration_parameters
- deduplication of vqe_iterations on (experiment_id, iteration_step)
- collision-free positional child ids, in-batch dedup of redelivered records
- exact_estimator and performance_start / performance_end extraction
- schema-independent experiment_id
"""

import os
import sys
from pathlib import Path

import pytest

# docker/airflow is on the path so `common.*` resolves like it does in the Airflow container
_AIRFLOW_DIR = Path(__file__).parent.parent.parent / 'docker' / 'airflow'
sys.path.insert(0, str(_AIRFLOW_DIR))
sys.path.insert(0, str(_AIRFLOW_DIR / 'scripts'))

from quantum_incremental_processing import (  # noqa: E402
    get_table_configs,
    identify_new_records,
    transform_quantum_data,
)


def _tables(df):
    """The feature tables of one transform call, keyed by table name."""
    return transform_quantum_data(df).tables


def _java_17_available() -> bool:
    """Return True if Java 17+ is available (required by the installed PySpark)."""
    import subprocess

    try:
        result = subprocess.run(
            ['java', '-version'],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=5,
        )
        output = result.stderr + result.stdout
        # Java 17 reports "17.", Java 11 reports "11."
        for major in range(17, 30):
            if f'version "{major}.' in output or f'version "{major}"' in output:
                return True
        return False
    except Exception:
        return False


_SPARK_AVAILABLE = _java_17_available()
pytestmark = pytest.mark.skipif(
    not _SPARK_AVAILABLE,
    reason='Spark unavailable in this environment (requires Java 17+)',
)


@pytest.fixture(scope='session')
def spark():
    """Session-scoped SparkSession for all Spark extraction tests."""
    from pyspark.sql import SparkSession

    os.environ.setdefault('PYSPARK_PYTHON', sys.executable)

    session = (
        SparkSession.builder.master('local[1]')
        .appName('test_spark_extraction')
        .config('spark.sql.shuffle.partitions', '2')
        .config('spark.ui.enabled', 'false')
        .getOrCreate()
    )
    session.sparkContext.setLogLevel('ERROR')
    yield session
    session.stop()


def _build_schema():
    """Spark schema matching the VQEDecoratedResult Avro structure."""
    from pyspark.sql.types import (
        ArrayType,
        BooleanType,
        DoubleType,
        IntegerType,
        StringType,
        StructField,
        StructType,
    )

    return StructType(
        [
            StructField('molecule_id', IntegerType(), False),
            StructField('basis_set', StringType(), False),
            StructField('hamiltonian_time', DoubleType(), False),
            StructField('mapping_time', DoubleType(), False),
            StructField('vqe_time', DoubleType(), False),
            StructField('total_time', DoubleType(), False),
            StructField('performance_start', StringType(), True),
            StructField('performance_end', StringType(), True),
            StructField(
                'molecule',
                StructType(
                    [
                        StructField(
                            'molecule_data',
                            StructType(
                                [
                                    StructField('symbols', ArrayType(StringType()), False),
                                    StructField(
                                        'coords', ArrayType(ArrayType(DoubleType())), False
                                    ),
                                    StructField('multiplicity', IntegerType(), False),
                                    StructField('charge', IntegerType(), False),
                                    StructField('units', StringType(), False),
                                    StructField('masses', ArrayType(DoubleType()), True),
                                ]
                            ),
                        )
                    ]
                ),
            ),
            StructField(
                'vqe_result',
                StructType(
                    [
                        StructField(
                            'initial_data',
                            StructType(
                                [
                                    StructField('backend', StringType(), False),
                                    StructField('num_qubits', IntegerType(), False),
                                    StructField(
                                        'hamiltonian',
                                        ArrayType(
                                            StructType(
                                                [
                                                    StructField('label', StringType()),
                                                    StructField(
                                                        'coefficients',
                                                        StructType(
                                                            [
                                                                StructField('real', DoubleType()),
                                                                StructField(
                                                                    'imaginary', DoubleType()
                                                                ),
                                                            ]
                                                        ),
                                                    ),
                                                ]
                                            )
                                        ),
                                    ),
                                    StructField('num_parameters', IntegerType()),
                                    StructField('initial_parameters', ArrayType(DoubleType())),
                                    StructField('optimizer', StringType()),
                                    StructField('ansatz', StringType()),
                                    StructField('noise_backend', StringType()),
                                    StructField('default_shots', IntegerType()),
                                    StructField('ansatz_reps', IntegerType()),
                                    StructField('init_strategy', StringType(), True),
                                    StructField('seed', IntegerType(), True),
                                    StructField('ansatz_name', StringType(), True),
                                    StructField('exact_estimator', BooleanType(), True),
                                ]
                            ),
                        ),
                        StructField(
                            'iteration_list',
                            ArrayType(
                                StructType(
                                    [
                                        StructField('iteration', IntegerType()),
                                        StructField('parameters', ArrayType(DoubleType())),
                                        StructField('result', DoubleType()),
                                        StructField('std', DoubleType()),
                                        StructField('energy_delta', DoubleType(), True),
                                        StructField('parameter_delta_norm', DoubleType(), True),
                                        StructField('cumulative_min_energy', DoubleType(), True),
                                    ]
                                )
                            ),
                        ),
                        StructField('minimum', DoubleType()),
                        StructField('optimal_parameters', ArrayType(DoubleType())),
                        StructField('maxcv', DoubleType(), True),
                        StructField('minimization_time', DoubleType()),
                        StructField('nuclear_repulsion_energy', DoubleType(), True),
                        StructField('success', BooleanType(), True),
                        StructField('nfev', IntegerType(), True),
                        StructField('nit', IntegerType(), True),
                    ]
                ),
            ),
        ]
    )


def _record(initial=None, result=None, **top_level):
    """One raw experiment record; `initial` / `result` patch `initial_data` / `vqe_result`."""
    record = {
        'molecule_id': 0,
        'basis_set': 'sto-3g',
        'hamiltonian_time': 1.0,
        'mapping_time': 0.5,
        'vqe_time': 10.0,
        'total_time': 11.5,
        'performance_start': None,
        'performance_end': None,
        'molecule': {
            'molecule_data': {
                'symbols': ['H', 'H'],
                'coords': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.735]],
                'multiplicity': 1,
                'charge': 0,
                'units': 'angstrom',
                'masses': None,
            }
        },
        'vqe_result': {
            'initial_data': {
                'backend': 'qasm_simulator',
                'num_qubits': 4,
                'hamiltonian': [{'label': 'II', 'coefficients': {'real': -1.0, 'imaginary': 0.0}}],
                'num_parameters': 4,
                'initial_parameters': [0.1, 0.2, 0.3, 0.4],
                'optimizer': 'L-BFGS-B',
                'ansatz': 'OPENQASM 3.0;',
                'noise_backend': 'none',
                'default_shots': 1024,
                'ansatz_reps': 1,
                'init_strategy': 'hf',
                'seed': 42,
                'ansatz_name': 'EfficientSU2',
                'exact_estimator': False,
            },
            'iteration_list': [
                {
                    'iteration': 0,
                    'parameters': [0.1, 0.2, 0.3, 0.4],
                    'result': -1.1,
                    'std': 0.01,
                    'energy_delta': None,
                    'parameter_delta_norm': None,
                    'cumulative_min_energy': -1.1,
                },
                {
                    'iteration': 1,
                    'parameters': [0.15, 0.25, 0.35, 0.45],
                    'result': -1.15,
                    'std': 0.01,
                    'energy_delta': -0.05,
                    'parameter_delta_norm': 0.1,
                    'cumulative_min_energy': -1.15,
                },
            ],
            'minimum': -1.15,
            'optimal_parameters': [0.15, 0.25, 0.35, 0.45],
            'maxcv': None,
            'minimization_time': 10.0,
            'nuclear_repulsion_energy': 0.715,
            'success': True,
            'nfev': 20,
            'nit': 2,
        },
    }
    record['vqe_result']['initial_data'].update(initial or {})
    record['vqe_result'].update(result or {})
    record.update(top_level)
    return record


def _frame(spark, *records):
    return spark.createDataFrame(list(records), schema=_build_schema())


def _experiment_id(spark, **record_kwargs):
    df = _frame(spark, _record(**record_kwargs))
    return _tables(df)['vqe_results'].select('experiment_id').first()['experiment_id']


@pytest.fixture(scope='session')
def sample_df(spark):
    """Minimal Spark DataFrame matching the VQEDecoratedResult Avro structure."""
    return _frame(spark, _record())


@pytest.fixture(scope='session')
def transformed(sample_df):
    return _tables(sample_df)


class TestMoleculeName:
    def test_molecule_name_column_present(self, transformed):
        assert 'molecule_name' in transformed['molecules'].columns

    def test_molecule_name_derived_from_symbols(self, transformed):
        row = transformed['molecules'].select('molecule_name').first()
        assert row['molecule_name'] == 'HH'


class TestAnsatzInfo:
    def test_init_strategy_column_present(self, transformed):
        assert 'init_strategy' in transformed['ansatz_info'].columns

    def test_ansatz_name_column_present(self, transformed):
        assert 'ansatz_name' in transformed['ansatz_info'].columns

    def test_init_strategy_value(self, transformed):
        row = transformed['ansatz_info'].select('init_strategy').first()
        assert row['init_strategy'] == 'hf'

    def test_ansatz_name_value(self, transformed):
        row = transformed['ansatz_info'].select('ansatz_name').first()
        assert row['ansatz_name'] == 'EfficientSU2'

    def test_init_strategy_defaults_to_random(self, spark):
        """Null init_strategy should fall back to 'random'."""
        df = _frame(spark, _record(initial={'init_strategy': None}))
        row = _tables(df)['ansatz_info'].select('init_strategy').first()
        assert row['init_strategy'] == 'random'

    def test_ansatz_name_defaults_to_efficient_su2(self, spark):
        """Null ansatz_name should fall back to 'EfficientSU2'."""
        df = _frame(spark, _record(initial={'ansatz_name': None}))
        row = _tables(df)['ansatz_info'].select('ansatz_name').first()
        assert row['ansatz_name'] == 'EfficientSU2'


class TestVQEResults:
    def test_new_columns_present(self, transformed):
        cols = transformed['vqe_results'].columns
        for expected in [
            'init_strategy',
            'seed',
            'ansatz_name',
            'nuclear_repulsion_energy',
            'success',
            'nfev',
            'nit',
        ]:
            assert expected in cols, f"'{expected}' missing from vqe_results"

    def test_nuclear_repulsion_energy_value(self, transformed):
        row = transformed['vqe_results'].select('nuclear_repulsion_energy').first()
        assert abs(row['nuclear_repulsion_energy'] - 0.715) < 1e-9

    def test_success_value(self, transformed):
        row = transformed['vqe_results'].select('success').first()
        assert row['success'] is True

    def test_nfev_value(self, transformed):
        row = transformed['vqe_results'].select('nfev').first()
        assert row['nfev'] == 20

    def test_nit_value(self, transformed):
        row = transformed['vqe_results'].select('nit').first()
        assert row['nit'] == 2

    def test_seed_value(self, transformed):
        row = transformed['vqe_results'].select('seed').first()
        assert row['seed'] == 42

    def test_null_ml_fields_handled(self, spark, sample_df):
        """Nulls in nuclear_repulsion_energy/success/nfev/nit should not crash."""
        from pyspark.sql.functions import col, lit

        df_nulls = sample_df.withColumn(
            'vqe_result',
            col('vqe_result')
            .withField('nuclear_repulsion_energy', lit(None).cast('double'))
            .withField('success', lit(None).cast('boolean'))
            .withField('nfev', lit(None).cast('int'))
            .withField('nit', lit(None).cast('int')),
        )
        result = _tables(df_nulls)
        row = (
            result['vqe_results']
            .select('nuclear_repulsion_energy', 'success', 'nfev', 'nit')
            .first()
        )
        assert row['nuclear_repulsion_energy'] is None
        assert row['success'] is None
        assert row['nfev'] is None
        assert row['nit'] is None


class TestIterationParametersPosexplode:
    def test_parameter_index_is_positional(self, transformed):
        """parameter_index must be the actual array position, not a hash."""
        rows = (
            transformed['iteration_parameters']
            .filter('iteration_step = 0')
            .orderBy('parameter_index')
            .select('parameter_index', 'parameter_value')
            .collect()
        )
        assert len(rows) == 4
        for expected_idx, row in enumerate(rows):
            assert row['parameter_index'] == expected_idx, (
                f'Expected parameter_index={expected_idx}, got {row["parameter_index"]}'
            )

    def test_parameter_values_match_source(self, transformed):
        """Extracted parameter values must match the source iteration parameters."""
        rows = (
            transformed['iteration_parameters']
            .filter('iteration_step = 0')
            .orderBy('parameter_index')
            .select('parameter_value')
            .collect()
        )
        expected = [0.1, 0.2, 0.3, 0.4]
        for row, exp in zip(rows, expected, strict=True):
            assert abs(row['parameter_value'] - exp) < 1e-9

    def test_parameter_id_is_iteration_id_plus_position(self, transformed):
        """parameter_id is `<iteration_id>_p<position>`, built from plain strings."""
        rows = (
            transformed['iteration_parameters']
            .filter('iteration_step = 0')
            .orderBy('parameter_index')
            .select('experiment_id', 'iteration_id', 'parameter_id', 'parameter_index')
            .collect()
        )
        for row in rows:
            assert row['iteration_id'] == f'{row["experiment_id"]}_iter_0'
            assert row['parameter_id'] == f'{row["iteration_id"]}_p{row["parameter_index"]}'


class TestVQEIterationsDeduplication:
    def test_no_duplicate_iteration_steps(self, transformed):
        """Each (experiment_id, iteration_step) pair must be unique."""
        from pyspark.sql.functions import count

        df = transformed['vqe_iterations']
        dup_count = (
            df.groupBy('experiment_id', 'iteration_step')
            .agg(count('*').alias('cnt'))
            .filter('cnt > 1')
            .count()
        )
        assert dup_count == 0, f'Found {dup_count} duplicate (experiment_id, iteration_step) pairs'

    def test_correct_iteration_count(self, transformed):
        """Should have exactly 2 distinct iterations for the test record."""
        count = transformed['vqe_iterations'].count()
        assert count == 2


class TestDeterministicExperimentId:
    """experiment_id must be a deterministic hash of the experiment identity.

    Regression for the CRITICAL uuid4 bug: a random per-row experiment_id defeats
    the dedup in identify_new_records (the same record gets a fresh id every run),
    so each daily run re-appends the whole dataset. The id must be stable across
    re-processing of the same record, and distinct across genuinely different runs.
    """

    def test_experiment_id_is_deterministic_across_runs(self, sample_df):
        """Re-processing the same record yields the same experiment_id (dedup can match)."""
        first = _tables(sample_df)['vqe_results'].select('experiment_id').first()
        second = _tables(sample_df)['vqe_results'].select('experiment_id').first()
        assert first['experiment_id'] == second['experiment_id']

    def test_experiment_id_is_sha256_not_uuid(self, transformed):
        """The id is a 64-char hex sha256 digest, not a random uuid4 (36 chars, dashed)."""
        exp_id = transformed['vqe_results'].select('experiment_id').first()['experiment_id']
        assert len(exp_id) == 64
        assert '-' not in exp_id
        assert all(c in '0123456789abcdef' for c in exp_id)

    def test_distinct_seed_yields_distinct_experiment_id(self, spark):
        """A different seed is a different experiment -> different id (no false dedup)."""
        assert _experiment_id(spark) != _experiment_id(spark, initial={'seed': 99})

    def test_distinct_basis_yields_distinct_experiment_id(self, spark):
        """A different basis set is a different experiment -> different id."""
        assert _experiment_id(spark) != _experiment_id(spark, basis_set='cc-pvdz')

    def test_distinct_exact_estimator_yields_distinct_experiment_id(self, spark):
        assert _experiment_id(spark) != _experiment_id(spark, initial={'exact_estimator': True})

    def test_hamiltonian_is_part_of_the_identity(self, spark):
        other = [{'label': 'II', 'coefficients': {'real': -2.0, 'imaginary': 0.0}}]
        assert _experiment_id(spark) != _experiment_id(spark, initial={'hamiltonian': other})

    def test_distinct_performance_start_yields_distinct_experiment_id(self, spark):
        """The snapshot taken at the start of a run tells monitored runs apart."""
        first = _experiment_id(spark, performance_start='{"timestamp": "2026-10-01T10:00:00"}')
        later = _experiment_id(spark, performance_start='{"timestamp": "2026-10-02T10:00:00"}')
        assert first != later
        assert first != _experiment_id(spark)

    def test_same_performance_start_yields_same_experiment_id(self, spark):
        snapshot = '{"timestamp": "2026-10-01T10:00:00"}'
        assert _experiment_id(spark, performance_start=snapshot) == _experiment_id(
            spark, performance_start=snapshot
        )

    def test_results_are_not_part_of_the_identity(self, spark):
        """A rerun with the same inputs but other results keeps the id (first run wins)."""
        other = _experiment_id(spark, result={'minimum': -0.5, 'nfev': 99}, total_time=99.0)
        assert _experiment_id(spark) == other

    def test_id_does_not_depend_on_inferred_schema(self, spark, sample_df):
        """JSON inference sorts fields alphabetically and types ints as long; same id."""
        import json

        raw = json.dumps(_record())
        json_df = spark.read.json(spark.sparkContext.parallelize([raw]))
        assert json_df.schema != sample_df.schema

        json_id = _tables(json_df)['vqe_results'].select('experiment_id').first()['experiment_id']
        assert json_id == _experiment_id(spark)

    def test_id_ignores_the_sign_of_zero(self, spark):
        """Avro keeps -0.0, but Spark reads a JSON `-0` as 0.0; both must hash the same."""
        import json
        import re

        negative = {
            'hamiltonian': [{'label': 'II', 'coefficients': {'real': -1.0, 'imaginary': -0.0}}],
            'initial_parameters': [-0.0, 0.2, 0.3, 0.4],
        }
        record = _record(initial=negative)
        record['molecule']['molecule_data']['coords'] = [[-0.0, 0.0, 0.0], [0.0, 0.0, 0.735]]
        record['molecule']['molecule_data']['masses'] = [-0.0, 1.0]

        avro_id = (
            _tables(_frame(spark, record))['vqe_results']
            .select('experiment_id')
            .first()['experiment_id']
        )
        raw = re.sub(r'-0\.0(?![0-9])', '-0', json.dumps(record))
        json_df = spark.read.json(spark.sparkContext.parallelize([raw]))
        json_id = _tables(json_df)['vqe_results'].select('experiment_id').first()['experiment_id']

        assert json_id == avro_id

    def test_child_tables_share_the_same_experiment_id(self, transformed):
        """The deterministic id propagates unchanged to every child table."""
        vqe_id = transformed['vqe_results'].select('experiment_id').first()['experiment_id']
        for table in ('molecules', 'ansatz_info', 'performance_metrics', 'vqe_iterations'):
            row = transformed[table].select('experiment_id').first()
            assert row['experiment_id'] == vqe_id, f'{table} experiment_id diverged'


def _distinct(df, column):
    return df.select(column).distinct().count()


class TestChildTableIds:
    """Child ids are plain `<experiment_id>_<kind>_<position>` strings, unique within a run."""

    @pytest.fixture(scope='class')
    def zero_parameters(self, spark):
        zeros = [0.0] * 4
        df = _frame(
            spark,
            _record(
                initial={'initial_parameters': zeros},
                result={'optimal_parameters': zeros},
            ),
        )
        return _tables(df)

    @pytest.mark.parametrize(
        ('table', 'kind'), [('initial_parameters', 'init'), ('optimal_parameters', 'opt')]
    )
    def test_equal_values_still_get_unique_ids(self, zero_parameters, table, kind):
        rows = zero_parameters[table].orderBy('parameter_index').collect()
        assert [r['parameter_index'] for r in rows] == [0, 1, 2, 3]
        assert len({r['parameter_id'] for r in rows}) == 4
        for row in rows:
            assert row['parameter_id'] == f'{row["experiment_id"]}_{kind}_{row["parameter_index"]}'

    def test_position_order_is_preserved(self, transformed):
        initial = transformed['initial_parameters'].orderBy('parameter_index').collect()
        assert [r['parameter_index'] for r in initial] == [0, 1, 2, 3]
        assert [r['initial_parameter_value'] for r in initial] == [0.1, 0.2, 0.3, 0.4]

        optimal = transformed['optimal_parameters'].orderBy('parameter_index').collect()
        assert [r['parameter_index'] for r in optimal] == [0, 1, 2, 3]
        assert [r['optimal_parameter_value'] for r in optimal] == [0.15, 0.25, 0.35, 0.45]

    @pytest.fixture(scope='class')
    def long_run(self, spark):
        """1200 steps of 3 parameters and a Hamiltonian whose labels repeat."""
        labels = ['II', 'ZI', 'II', 'ZI', 'IZ', 'II']
        hamiltonian = [
            {'label': label, 'coefficients': {'real': 0.5, 'imaginary': 0.0}} for label in labels
        ]
        steps = [
            {
                'iteration': step,
                'parameters': [0.0, 0.0, 0.0],
                'result': -1.0,
                'std': 0.0,
                'energy_delta': None,
                'parameter_delta_norm': None,
                'cumulative_min_energy': -1.0,
            }
            for step in range(1200)
        ]
        df = _frame(
            spark, _record(initial={'hamiltonian': hamiltonian}, result={'iteration_list': steps})
        )
        return _tables(df)

    def test_iteration_ids_are_unique_over_1200_steps(self, long_run):
        iterations = long_run['vqe_iterations']
        assert iterations.count() == 1200
        assert _distinct(iterations, 'iteration_id') == 1200
        # a hash modulo 1e6 could by luck also be unique, so pin the format as well
        row = iterations.filter('iteration_step = 1199').first()
        assert row['iteration_id'] == f'{row["experiment_id"]}_iter_1199'

    def test_iteration_parameter_ids_are_unique(self, long_run):
        params = long_run['iteration_parameters']
        assert params.count() == 3600
        assert _distinct(params, 'parameter_id') == 3600

    def test_term_ids_are_unique_with_repeated_labels(self, long_run):
        terms = long_run['hamiltonian_terms'].orderBy('term_index').collect()
        assert [r['term_label'] for r in terms] == ['II', 'ZI', 'II', 'ZI', 'IZ', 'II']
        assert [r['term_index'] for r in terms] == list(range(6))
        assert len({r['term_id'] for r in terms}) == 6
        for row in terms:
            assert row['term_id'] == f'{row["experiment_id"]}_term_{row["term_index"]}'

    def test_key_columns_exist_in_their_tables(self, transformed):
        for name, config in get_table_configs().items():
            for column in config['key_columns'] + config['partition_columns']:
                assert column in transformed[name].columns, f'{name} lacks {column}'


class TestInBatchDeduplication:
    """The same experiment delivered twice must not double any table."""

    @staticmethod
    def _counts(tables):
        return {name: frame.count() for name, frame in tables.items()}

    def test_identical_copies_give_one_row_per_table(self, spark, transformed):
        doubled = _tables(_frame(spark, _record(), _record()))
        assert self._counts(doubled) == self._counts(transformed)

    def test_differing_copies_pick_the_same_one_in_any_order(self, spark):
        """A rerun with other results shares the id; the pick must not depend on row order."""
        slow = _record(result={'minimum': -1.0}, total_time=50.0)
        fast = _record(result={'minimum': -1.15}, total_time=11.5)

        energies = set()
        for records in ((slow, fast), (fast, slow)):
            tables = _tables(_frame(spark, *records))
            assert tables['vqe_results'].count() == 1
            assert tables['vqe_iterations'].count() == 2
            energies.add(tables['vqe_results'].first()['minimum_energy'])

        assert energies == {-1.0}


class TestAvroOnlyFields:
    """Fields the Avro schema carries that earlier versions never extracted."""

    def test_exact_estimator_lands_in_vqe_results(self, spark):
        for value in (True, False):
            df = _frame(spark, _record(initial={'exact_estimator': value}))
            row = _tables(df)['vqe_results'].select('exact_estimator').first()
            assert row['exact_estimator'] is value

    def test_exact_estimator_null_takes_the_schema_default(self, spark):
        """The Avro schema declares default False, so NULL becomes False."""
        df = _frame(spark, _record(initial={'exact_estimator': None}))
        row = _tables(df)['vqe_results'].select('exact_estimator').first()
        assert row['exact_estimator'] is False

    def test_exact_estimator_is_not_in_other_tables(self, transformed):
        assert 'exact_estimator' not in transformed['ansatz_info'].columns
        assert 'exact_estimator' not in transformed['performance_metrics'].columns

    def test_performance_snapshots_land_in_performance_metrics(self, spark):
        start = '{"timestamp": "2026-10-01T10:00:00"}'
        end = '{"timestamp": "2026-10-01T10:05:00"}'
        df = _frame(spark, _record(performance_start=start, performance_end=end))
        tables = _tables(df)

        row = tables['performance_metrics'].select('performance_start', 'performance_end').first()
        assert (row['performance_start'], row['performance_end']) == (start, end)
        assert 'performance_start' not in tables['vqe_results'].columns

    def test_performance_snapshots_stay_null_without_monitoring(self, transformed):
        """No default in the Avro schema (null), so nothing is invented."""
        row = (
            transformed['performance_metrics']
            .select('performance_start', 'performance_end')
            .first()
        )
        assert row['performance_start'] is None
        assert row['performance_end'] is None


class TestIdentifyNewRecords:
    """identify_new_records is an anti-join of the batch against the table's keys."""

    @staticmethod
    def _batch(spark):
        return spark.createDataFrame([('a', 1), ('b', 2), ('c', 3)], ['key', 'value'])

    @staticmethod
    def _patch_existing(monkeypatch, spark, existing):
        # no Iceberg runtime locally: serve the "existing table keys" from memory
        monkeypatch.setattr(spark, 'sql', lambda _query: existing)

    def test_returns_only_rows_with_absent_keys(self, spark, monkeypatch):
        existing = spark.createDataFrame([('a',), ('c',)], ['key'])
        self._patch_existing(monkeypatch, spark, existing)

        result = identify_new_records(spark, self._batch(spark), 'any_table', ['key'])

        assert [(r['key'], r['value']) for r in result.collect()] == [('b', 2)]

    def test_returns_all_rows_when_table_is_empty(self, spark, monkeypatch):
        existing = spark.createDataFrame([], 'key string')
        self._patch_existing(monkeypatch, spark, existing)

        result = identify_new_records(spark, self._batch(spark), 'any_table', ['key'])

        assert sorted(r['key'] for r in result.collect()) == ['a', 'b', 'c']


class TestBatchMetadataIsShared:
    def test_timestamp_and_date_identical_across_all_tables(self, sample_df):
        """One transform call = one timestamp, one date and one batch id for every table."""
        result = transform_quantum_data(sample_df)

        assert len(result.tables) == 9
        seen = set()
        for name, frame in result.tables.items():
            rows = (
                frame.select('processing_timestamp', 'processing_date', 'processing_batch_id')
                .distinct()
                .collect()
            )
            assert len(rows) == 1, f'{name} has several metadata values'
            seen.add(tuple(rows[0]))

        assert len(seen) == 1
        _, date_value, batch_id = seen.pop()
        assert date_value == result.batch.date
        assert batch_id == result.batch.batch_id
