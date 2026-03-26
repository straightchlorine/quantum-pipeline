"""Tests for quantum_incremental_processing.py — Spark extraction logic.

Covers new and fixed fields added in QUA-18:
- molecule_name derived from atom symbols
- init_strategy / ansatz_name in ansatz_info and vqe_results
- seed / nuclear_repulsion_energy / success / nfev / nit in vqe_results
- posexplode-based parameter_index in iteration_parameters
- deduplication of vqe_iterations on (experiment_id, iteration_step)
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'docker' / 'airflow' / 'scripts'))

from quantum_incremental_processing import transform_quantum_data


def _java_17_available() -> bool:
    """Return True if Java 17+ is available (required by the installed PySpark)."""
    import subprocess

    try:
        result = subprocess.run(
            ['java', '-version'], capture_output=True, text=True, timeout=5  # noqa: S607
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


@pytest.fixture(scope='session')
def sample_df(spark):
    """Minimal Spark DataFrame matching the VQEDecoratedResult Avro structure."""
    from pyspark.sql.types import (
        ArrayType,
        BooleanType,
        DoubleType,
        IntegerType,
        StringType,
        StructField,
        StructType,
    )

    schema = StructType([
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
            StructType([
                StructField(
                    'molecule_data',
                    StructType([
                        StructField('symbols', ArrayType(StringType()), False),
                        StructField('coords', ArrayType(ArrayType(DoubleType())), False),
                        StructField('multiplicity', IntegerType(), False),
                        StructField('charge', IntegerType(), False),
                        StructField('units', StringType(), False),
                        StructField('masses', ArrayType(DoubleType()), True),
                    ]),
                )
            ]),
        ),
        StructField(
            'vqe_result',
            StructType([
                StructField(
                    'initial_data',
                    StructType([
                        StructField('backend', StringType(), False),
                        StructField('num_qubits', IntegerType(), False),
                        StructField(
                            'hamiltonian',
                            ArrayType(
                                StructType([
                                    StructField('label', StringType()),
                                    StructField(
                                        'coefficients',
                                        StructType([
                                            StructField('real', DoubleType()),
                                            StructField('imaginary', DoubleType()),
                                        ]),
                                    ),
                                ])
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
                    ]),
                ),
                StructField(
                    'iteration_list',
                    ArrayType(
                        StructType([
                            StructField('iteration', IntegerType()),
                            StructField('parameters', ArrayType(DoubleType())),
                            StructField('result', DoubleType()),
                            StructField('std', DoubleType()),
                            StructField('energy_delta', DoubleType(), True),
                            StructField('parameter_delta_norm', DoubleType(), True),
                            StructField('cumulative_min_energy', DoubleType(), True),
                        ])
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
            ]),
        ),
    ])

    data = [
        {
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
                    'hamiltonian': [
                        {'label': 'II', 'coefficients': {'real': -1.0, 'imaginary': 0.0}}
                    ],
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
    ]

    return spark.createDataFrame(data, schema=schema)


@pytest.fixture(scope='session')
def transformed(sample_df):
    """Run transform_quantum_data once for all transform tests."""
    return transform_quantum_data(sample_df)


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

    def test_init_strategy_defaults_to_random(self, spark, sample_df):
        """Null init_strategy should fall back to 'random'."""
        from pyspark.sql.functions import col

        df_null_strategy = sample_df.withColumn(
            'vqe_result',
            col('vqe_result').withField('initial_data.init_strategy', col('performance_start')),
        )
        result = transform_quantum_data(df_null_strategy)
        row = result['ansatz_info'].select('init_strategy').first()
        assert row['init_strategy'] == 'random'

    def test_ansatz_name_defaults_to_efficient_su2(self, spark, sample_df):
        """Null ansatz_name should fall back to 'EfficientSU2'."""
        from pyspark.sql.functions import col

        df_null_name = sample_df.withColumn(
            'vqe_result',
            col('vqe_result').withField('initial_data.ansatz_name', col('performance_start')),
        )
        result = transform_quantum_data(df_null_name)
        row = result['ansatz_info'].select('ansatz_name').first()
        assert row['ansatz_name'] == 'EfficientSU2'


class TestVQEResults:
    def test_new_columns_present(self, transformed):
        cols = transformed['vqe_results'].columns
        for expected in ['init_strategy', 'seed', 'ansatz_name', 'nuclear_repulsion_energy',
                         'success', 'nfev', 'nit']:
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

        df_nulls = (
            sample_df.withColumn(
                'vqe_result',
                col('vqe_result')
                .withField('nuclear_repulsion_energy', lit(None).cast('double'))
                .withField('success', lit(None).cast('boolean'))
                .withField('nfev', lit(None).cast('int'))
                .withField('nit', lit(None).cast('int')),
            )
        )
        result = transform_quantum_data(df_nulls)
        row = result['vqe_results'].select(
            'nuclear_repulsion_energy', 'success', 'nfev', 'nit'
        ).first()
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
                f"Expected parameter_index={expected_idx}, got {row['parameter_index']}"
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

    def test_parameter_id_uses_positional_index(self, transformed):
        """parameter_id should encode the positional index."""
        rows = (
            transformed['iteration_parameters']
            .filter('iteration_step = 0')
            .orderBy('parameter_index')
            .select('parameter_id', 'parameter_index')
            .collect()
        )
        for row in rows:
            assert row['parameter_id'].endswith(f"_param_{row['parameter_index']}")


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
        assert dup_count == 0, f"Found {dup_count} duplicate (experiment_id, iteration_step) pairs"

    def test_correct_iteration_count(self, transformed):
        """Should have exactly 2 distinct iterations for the test record."""
        count = transformed['vqe_iterations'].count()
        assert count == 2
