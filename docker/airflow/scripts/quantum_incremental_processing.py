"""Incremental VQE experiment data -> feature tables in Iceberg format.

Contract of one run (`main`), for every topic directory under `S3_BUCKET_URL`:

1. Re-read EVERY raw Avro/JSON file of the topic. There is no watermark and no
   file bookkeeping, so the cost of a run grows with the whole history.
2. Collapse records that share an `experiment_id` (a raw file delivered twice)
   to one, then flatten them into 9 feature tables (molecules, vqe_results,
   vqe_iterations, ...). Each table has derived key columns: `experiment_id`
   (hash of the record's inputs, see `_experiment_identity`) or an id built
   from it by plain string concatenation with the position of the row.
3. For each table, anti-join against the keys already in the Iceberg table and
   append only the rows whose key is absent. Already-seen rows are never
   updated, so the first copy of a key wins. If the table does not exist yet,
   it is created from the whole frame instead.
4. After each write, tag the new snapshot `v_<batch>` / `v_incr_<batch>` (an
   Iceberg tag is a named pointer to a snapshot: it lets you query the table
   exactly as it was after this run with `VERSION AS OF`, and keeps that
   snapshot from being removed by snapshot expiry). The tags and row counts
   of one run are recorded in the `processing_metadata` table.

Tables created by an older version of this script have different key values
and must be dropped and rebuilt from the raw files.
"""

import logging
import sys
import uuid
from datetime import date, datetime, timezone
from typing import NamedTuple

sys.path.insert(0, '/opt/airflow/dags')

from common.pipeline_config import CATALOG_FQN, S3_BUCKET_URL
from common.spark_factory import create_spark_session
from pyspark.sql import Window
from pyspark.sql.functions import (
    array_join,
    coalesce,
    col,
    concat,
    concat_ws,
    explode,
    lit,
    posexplode,
    row_number,
    sha2,
    size,
    transform,
)
from pyspark.sql.types import ArrayType

logger = logging.getLogger(__name__)

_APP_NAME = 'Quantum Pipeline Feature Processing'

# Partitions of the raw frame after the read (the raw files are small and numerous).
_RAW_NUM_PARTITIONS = 4


class BatchInfo(NamedTuple):
    """Identity of one job run, computed once in Python and shared by every table."""

    batch_id: str
    name: str
    timestamp: datetime
    date: date


class TransformResult(NamedTuple):
    """Output of `transform_quantum_data`.

    Attributes:
        tables: The 9 feature DataFrames, keyed by target table name.
        batch: Identity of this run.
        base_df: Persisted intermediate frame all tables derive from. The caller
            must `unpersist()` it once the tables are written.
    """

    tables: dict
    batch: BatchInfo
    base_df: object


def list_available_topics(spark, bucket_path):
    """List the topic directories directly under `bucket_path`.

    Exceptions from the storage layer (bad bucket, bad credentials, network)
    are NOT caught, so the Airflow task fails instead of "succeeding" with zero
    topics.

    Args:
        spark: SparkSession.
        bucket_path: S3 path that contains one sub-directory per topic.

    Returns:
        list: Topic (directory) names; empty if `bucket_path` does not exist or
        is not a directory.
    """
    # S3A filesystem and credentials provider are configured in spark-defaults.conf
    # (EnvironmentVariableCredentialsProvider reads AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY)
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(
        spark._jvm.java.net.URI.create(bucket_path), spark._jsc.hadoopConfiguration()
    )

    path = spark._jvm.org.apache.hadoop.fs.Path(bucket_path)

    if fs.exists(path) and fs.isDirectory(path):
        return [f.getPath().getName() for f in fs.listStatus(path) if f.isDirectory()]

    logger.warning('Path %s does not exist or is not a directory, no topics found', bucket_path)
    return []


def read_experiments_by_topic(spark, bucket_path, topic_name, num_partitions=None):
    """Read the raw experiment files of one topic directory, Avro first, JSON as fallback.

    Args:
        spark: SparkSession.
        bucket_path: S3 bucket path.
        topic_name: Name of the topic to read.
        num_partitions: Optional number of partitions for the dataframe.

    Returns:
        DataFrame: the topic data, cached because every table reads it.
    """
    # support both Avro (Kafka Connect) and JSON (Redpanda Connect) output
    # Use recursiveFileLookup to handle both flat and time-partitioned layouts
    # (S3A does not reliably support ** recursive globs)
    topic_dir = f'{bucket_path}{topic_name}'

    try:
        df = (
            spark.read.format('avro')
            .option('recursiveFileLookup', 'true')
            .option('pathGlobFilter', '*.avro')
            .load(topic_dir)
        )
    except Exception:
        # TODO: JSON infers bigint where Avro has int, so the same column differs per source
        df = (
            spark.read.option('recursiveFileLookup', 'true')
            .option('pathGlobFilter', '*.json')
            .json(topic_dir)
        )

    if num_partitions:
        df = df.repartition(num_partitions)

    # cache the dataframe as it will be used multiple times
    return df.cache()


# Separator between the hashed fields. A control character, so it cannot appear in
# a molecule symbol, a Pauli label or a backend name and shift text between fields.
_IDENTITY_SEPARATOR = '\u001f'


def _positive_zero(column, spark_type):
    """Turn every -0.0 inside a double-typed `column` into 0.0, leave other types alone.

    Avro keeps the sign of a negative zero, while Spark reads the JSON number `-0` as
    0.0 and renders -0.0 as '-0.0', so the same record would hash differently. Adding
    0.0 works because IEEE 754 defines -0.0 + 0.0 as 0.0.
    """
    if spark_type == 'double':
        return column + lit(0.0)
    if spark_type == 'array<double>':
        return transform(column, lambda value: value + lit(0.0))
    if spark_type == 'array<array<double>>':
        return transform(column, lambda row: transform(row, lambda value: value + lit(0.0)))
    return column


def _as_text(column, spark_type):
    """Render one field as text for hashing: cast to `spark_type` first, NULL as ''."""
    typed = _positive_zero(column.cast(spark_type), spark_type)
    return coalesce(typed.cast('string'), lit(''))


def _array_or_null(dataframe, path, element_type):
    """Return `path` cast to `array<element_type>`, or NULL if the column is not an array.

    JSON schema inference types a column that is NULL in every record as string,
    and a string cannot be cast to an array. Such a column can only hold NULLs.
    """
    if isinstance(dataframe.select(path).schema[0].dataType, ArrayType):
        return col(path).cast(f'array<{element_type}>')
    return lit(None).cast(f'array<{element_type}>')


def _experiment_identity(dataframe):
    """Build the `experiment_id` column: sha256 over an explicit, ordered field list.

    What goes into the hash, in this order (numerics are cast to a fixed type, ints
    to `bigint` and floats to `double`, so Avro INT and JSON LONG render the same, and
    a negative zero is rendered as 0.0 because JSON input cannot keep its sign):

    - `molecule_data`: symbols, coords, multiplicity, charge, units, masses
    - `basis_set`
    - `initial_data`: backend, num_qubits, hamiltonian (each term as
      `label:real:imaginary`), num_parameters, initial_parameters, optimizer,
      ansatz, ansatz_reps, noise_backend, default_shots, init_strategy, seed,
      ansatz_name, exact_estimator
    - `performance_start`, the only run-specific field the record has (a JSON
      snapshot with a timestamp, set only when performance monitoring was on)

    No result field is hashed. Two properties follow. The id does not depend on the
    schema Spark inferred (field order, extra fields), because nothing is read
    through `to_json(struct(...))`, so Avro and JSON input give the same id and a
    new field in the data cannot change it. And the same raw record always gets the
    same id, which is what the key anti-join needs to recognise it, while a
    different seed, basis or (with monitoring on) a different run gets a new one.
    Without monitoring, a rerun with identical inputs shares the id of the first
    run and is skipped by the anti-join (the first run wins).

    Args:
        dataframe: Raw experiment DataFrame; only its schema is inspected.

    Returns:
        Column: 64-char hex sha256 digest.
    """
    molecule = col('molecule.molecule_data')
    initial = col('vqe_result.initial_data')

    hamiltonian = array_join(
        transform(
            initial['hamiltonian'],
            lambda term: concat_ws(
                ':',
                coalesce(term['label'], lit('')),
                _as_text(term['coefficients']['real'], 'double'),
                _as_text(term['coefficients']['imaginary'], 'double'),
            ),
        ),
        ',',
    )

    fields = [
        _as_text(molecule['symbols'], 'array<string>'),
        _as_text(molecule['coords'], 'array<array<double>>'),
        _as_text(molecule['multiplicity'], 'bigint'),
        _as_text(molecule['charge'], 'bigint'),
        _as_text(molecule['units'], 'string'),
        _as_text(
            _array_or_null(dataframe, 'molecule.molecule_data.masses', 'double'), 'array<double>'
        ),
        _as_text(col('basis_set'), 'string'),
        _as_text(initial['backend'], 'string'),
        _as_text(initial['num_qubits'], 'bigint'),
        coalesce(hamiltonian, lit('')),
        _as_text(initial['num_parameters'], 'bigint'),
        _as_text(initial['initial_parameters'], 'array<double>'),
        _as_text(initial['optimizer'], 'string'),
        _as_text(initial['ansatz'], 'string'),
        _as_text(initial['ansatz_reps'], 'bigint'),
        _as_text(initial['noise_backend'], 'string'),
        _as_text(initial['default_shots'], 'bigint'),
        _as_text(initial['init_strategy'], 'string'),
        _as_text(initial['seed'], 'bigint'),
        _as_text(initial['ansatz_name'], 'string'),
        _as_text(initial['exact_estimator'], 'boolean'),
        _as_text(col('performance_start'), 'string'),
    ]
    return sha2(concat_ws(_IDENTITY_SEPARATOR, *fields), 256)


def _iteration_id():
    """Key of one optimizer step: `<experiment_id>_iter_<iteration_step>`."""
    return concat(col('experiment_id'), lit('_iter_'), col('iteration_step').cast('string'))


def add_metadata_columns(dataframe, processing_name):
    """Add `experiment_id` (see `_experiment_identity`) and the per-run metadata columns.

    Args:
        dataframe: Raw experiment DataFrame (nested Avro/JSON structure).
        processing_name: Name of the processing job.

    Returns:
        tuple: (DataFrame with `experiment_id`, `processing_timestamp`,
        `processing_date`, `processing_batch_id` and `processing_name` added,
        BatchInfo with the same values for the caller).
    """
    # Evaluated once here and passed as literals. current_timestamp() and
    # current_date() are re-evaluated per Spark job, so the 9 tables and the
    # metadata row would get different timestamps, and processing_date (a
    # partition key) could differ across midnight within a single run.
    now = datetime.now(timezone.utc)
    batch = BatchInfo(
        # one id per job run, not a dedup key, so random is fine
        batch_id=str(uuid.uuid4()),
        name=processing_name,
        timestamp=now,
        date=now.date(),
    )

    df = (
        dataframe.withColumn('experiment_id', _experiment_identity(dataframe))
        .withColumn('processing_timestamp', lit(batch.timestamp))
        .withColumn('processing_date', lit(batch.date))
        .withColumn('processing_batch_id', lit(batch.batch_id))
        .withColumn('processing_name', lit(batch.name))
    )
    return df, batch


def identify_new_records(spark, new_data_df, table_name, key_columns):
    """Return the rows of `new_data_df` whose key is not in the Iceberg table yet.

    The caller must have checked that the table exists and that `new_data_df`
    is not empty (see `process_incremental_data`).

    Args:
        spark: SparkSession.
        new_data_df: DataFrame containing potentially new data.
        table_name: Name of the table to check against.
        key_columns: List of column names that uniquely identify records.

    Returns:
        DataFrame: the rows of `new_data_df` that do not exist in the target table.
    """
    try:
        existing_keys = spark.sql(
            f'SELECT DISTINCT {", ".join(key_columns)} '  # noqa: S608
            f'FROM {CATALOG_FQN}.{table_name}'
        )
        # left_anti keeps the left rows with no match on the right. An empty
        # right side keeps everything. Keys are compared with `=`, so a row
        # with a NULL key never matches and is re-appended on every run.
        return new_data_df.join(existing_keys, on=key_columns, how='left_anti')

    except Exception as e:
        raise RuntimeError(f'Error identifying new records: {e!s}') from e


def _tag_latest_snapshot(spark, table_name, version_tag):
    """Point the Iceberg tag `version_tag` at the newest snapshot of `table_name`.

    The `snapshots` metadata table lists every commit of the table; the write that
    just finished is the one with the latest `committed_at`.
    """
    snapshot_id = spark.sql(
        f'SELECT snapshot_id FROM {CATALOG_FQN}.{table_name}.snapshots '  # noqa: S608
        'ORDER BY committed_at DESC LIMIT 1'
    ).collect()[0][0]

    spark.sql(f"""
    ALTER TABLE {CATALOG_FQN}.{table_name}
    CREATE TAG {version_tag} AS OF VERSION {snapshot_id}
    """)


def process_incremental_data(
    spark, new_data_df, table_name, key_columns, batch_id, partition_columns=None, comment=None
):
    """Write the rows of `new_data_df` that the target table does not have yet.

    Creates the table on the first run, appends afterwards, then tags the new
    snapshot so this run's result can be found later (`v_<batch>` for the
    creating run, `v_incr_<batch>` for appends).

    Args:
        spark: SparkSession.
        new_data_df: DataFrame containing potentially new data.
        table_name: Name of the target table.
        key_columns: List of column names that uniquely identify records.
        batch_id: `processing_batch_id` of this run, used to name the tag.
        partition_columns: Optional list of columns to partition by.
        comment: Optional table comment, only applied when the table is created.

    Returns:
        tuple: (version_tag, new_record_count), or (None, 0) when there was
        nothing to write: the input was empty or every key already exists.
    """
    # An exploded child frame is empty when a run has no hamiltonian terms or
    # iterations. Writing it would create an empty table and an empty snapshot.
    if new_data_df.isEmpty():
        logger.info('Input dataset is empty for %s', table_name)
        return None, 0

    batch_suffix = batch_id.replace('-', '')

    # The existence check comes before any filtering because the two cases
    # differ in more than the filter: with no table there is nothing to
    # anti-join against (querying it would fail), and only the creating write
    # can set partitioning and the table comment.
    if not spark.catalog.tableExists(f'{CATALOG_FQN}.{table_name}'):
        writer = new_data_df.write.format('iceberg').option('write-format', 'parquet')

        if partition_columns:
            writer = writer.partitionBy(*partition_columns)

        if comment:
            writer = writer.option('comment', comment)

        writer.mode('overwrite').saveAsTable(f'{CATALOG_FQN}.{table_name}')

        # Lets a later append add new columns: without this property Iceberg rejects
        # a write whose schema differs from the table, whatever the writer asks for.
        spark.sql(f"""
        ALTER TABLE {CATALOG_FQN}.{table_name}
        SET TBLPROPERTIES ('write.spark.accept-any-schema'='true')
        """)

        logger.info('Created table: %s.%s', CATALOG_FQN, table_name)

        version_tag = f'v_{batch_suffix}'
        _tag_latest_snapshot(spark, table_name, version_tag)

        logger.info('Created version tag: %s for table %s', version_tag, table_name)

        return version_tag, new_data_df.count()

    truly_new_data = identify_new_records(spark, new_data_df, table_name, key_columns)

    new_record_count = truly_new_data.count()
    if new_record_count == 0:
        logger.info('No new records found for table %s', table_name)
        return None, 0

    # DataFrameWriterV2 is the API the Iceberg docs show for schema merge. With
    # `mergeSchema` a column that is new in the frame is added to the table (old rows
    # get NULL); the table property set at creation is the other half of the opt-in.
    # Partitioning is fixed by the table, so there is no partitionBy on append.
    (truly_new_data.writeTo(f'{CATALOG_FQN}.{table_name}').option('mergeSchema', 'true').append())

    logger.info('Appended %d new records to table %s', new_record_count, table_name)

    version_tag = f'v_incr_{batch_suffix}'
    _tag_latest_snapshot(spark, table_name, version_tag)

    logger.info('Created incremental version tag: %s for table %s', version_tag, table_name)

    return version_tag, new_record_count


def transform_quantum_data(df):
    """Flatten the raw experiment records into the feature tables.

    Args:
        df: Original dataframe with quantum simulation data.

    Returns:
        TransformResult: the 9 table frames, the run's BatchInfo and the
        persisted intermediate frame the caller has to `unpersist()`.
    """
    base_df, batch = add_metadata_columns(df, 'quantum_base_processing')

    base_df = base_df.select(
        col('experiment_id'),
        col('molecule_id'),
        col('basis_set'),
        col('vqe_result.initial_data').alias('initial_data'),
        col('vqe_result.iteration_list').alias('iteration_list'),
        col('vqe_result.minimum').alias('minimum_energy'),
        col('vqe_result.optimal_parameters').alias('optimal_parameters'),
        col('vqe_result.maxcv').alias('maxcv'),
        col('vqe_result.minimization_time').alias('minimization_time'),
        col('vqe_result.nuclear_repulsion_energy').alias('nuclear_repulsion_energy'),
        col('vqe_result.success').alias('success'),
        col('vqe_result.nfev').alias('nfev'),
        col('vqe_result.nit').alias('nit'),
        col('hamiltonian_time'),
        col('mapping_time'),
        col('vqe_time'),
        col('total_time'),
        col('performance_start'),
        col('performance_end'),
        col('molecule.molecule_data').alias('molecule_data'),
        col('processing_timestamp'),
        col('processing_date'),
        col('processing_batch_id'),
        col('processing_name'),
    )

    # Redpanda Connect delivers at least once, so a raw file can arrive twice and the
    # same experiment_id would then appear in every table twice (the key anti-join
    # only compares with rows already in the table, not within the batch). The record
    # has no reliable timestamp ("keep the latest" is not possible: performance_end is
    # a JSON snapshot that is NULL without monitoring), and copies with the same id may
    # still differ in their results (a rerun with identical inputs). So dropDuplicates,
    # whose pick is arbitrary, is not enough; row_number over a fixed ordering of
    # result columns picks the same copy on every run.
    one_per_experiment = Window.partitionBy('experiment_id').orderBy(
        col('total_time').desc(),
        col('minimum_energy').asc(),
        col('nfev').asc(),
        col('nit').asc(),
    )
    base_df = (
        base_df.withColumn('_copy_rank', row_number().over(one_per_experiment))
        .filter(col('_copy_rank') == 1)
        .drop('_copy_rank')
    )

    # Every table below derives from base_df, and base_df holds the experiment_id
    # hash. Without persist() each count/isEmpty/write of each table would
    # recompute the hash from the raw files. The caller unpersists it.
    base_df = base_df.persist()

    df_molecule = base_df.select(
        col('experiment_id'),
        col('molecule_id'),
        col('molecule_data.symbols').alias('atom_symbols'),
        # TODO: molecule_name = array_join(symbols) is shared by geometries and charge states
        array_join(col('molecule_data.symbols'), '').alias('molecule_name'),
        col('molecule_data.coords').alias('coordinates'),
        col('molecule_data.multiplicity').alias('multiplicity'),
        col('molecule_data.charge').alias('charge'),
        col('molecule_data.units').alias('coordinate_units'),
        col('molecule_data.masses').alias('atomic_masses'),
        col('processing_timestamp'),
        col('processing_date'),
        col('processing_batch_id'),
        col('processing_name'),
    )

    df_ansatz = base_df.select(
        col('experiment_id'),
        col('molecule_id'),
        col('basis_set'),
        col('initial_data.ansatz').alias('ansatz'),
        col('initial_data.ansatz_reps').alias('ansatz_reps'),
        # TODO: these defaults fill ansatz_name/init_strategy when the record has NULL there
        coalesce(col('initial_data.ansatz_name'), lit('EfficientSU2')).alias('ansatz_name'),
        coalesce(col('initial_data.init_strategy'), lit('random')).alias('init_strategy'),
        col('processing_timestamp'),
        col('processing_date'),
        col('processing_batch_id'),
        col('processing_name'),
    )

    df_metrics = base_df.select(
        col('experiment_id'),
        col('molecule_id'),
        col('basis_set'),
        col('hamiltonian_time'),
        col('mapping_time'),
        col('vqe_time'),
        col('total_time'),
        col('minimization_time'),
        (col('hamiltonian_time') + col('mapping_time') + col('vqe_time')).alias(
            'computed_total_time'
        ),
        col('performance_start'),
        col('performance_end'),
        col('processing_timestamp'),
        col('processing_date'),
        col('processing_batch_id'),
        col('processing_name'),
    )

    df_vqe = base_df.select(
        col('experiment_id'),
        col('molecule_id'),
        col('basis_set'),
        col('initial_data.backend').alias('backend'),
        col('initial_data.num_qubits').alias('num_qubits'),
        col('initial_data.optimizer').alias('optimizer'),
        col('initial_data.noise_backend').alias('noise_backend'),
        col('initial_data.default_shots').alias('default_shots'),
        coalesce(col('initial_data.exact_estimator'), lit(False)).alias('exact_estimator'),
        col('initial_data.ansatz_reps').alias('ansatz_reps'),
        # TODO: these defaults fill ansatz_name/init_strategy when the record has NULL there
        coalesce(col('initial_data.ansatz_name'), lit('EfficientSU2')).alias('ansatz_name'),
        coalesce(col('initial_data.init_strategy'), lit('random')).alias('init_strategy'),
        col('initial_data.seed').alias('seed'),
        col('minimum_energy'),
        col('maxcv'),
        col('nuclear_repulsion_energy'),
        col('success'),
        col('nfev'),
        col('nit'),
        size(col('iteration_list')).alias('total_iterations'),
        col('processing_timestamp'),
        col('processing_date'),
        col('processing_batch_id'),
        col('processing_name'),
    )

    # posexplode keeps the position in the parameter vector, which says which gate
    # the value belongs to. The position is also what makes the id unique: values
    # repeat (all-zero initial parameters), positions do not.
    df_initial_parameters = base_df.select(
        col('experiment_id'),
        col('molecule_id'),
        col('basis_set'),
        col('initial_data.backend').alias('backend'),
        col('initial_data.num_qubits').alias('num_qubits'),
        posexplode(col('initial_data.initial_parameters')).alias(
            'parameter_index', 'initial_parameter_value'
        ),
        col('processing_timestamp'),
        col('processing_date'),
        col('processing_batch_id'),
        col('processing_name'),
    ).withColumn(
        'parameter_id',
        concat(col('experiment_id'), lit('_init_'), col('parameter_index').cast('string')),
    )

    df_optimal_parameters = base_df.select(
        col('experiment_id'),
        col('molecule_id'),
        col('basis_set'),
        col('initial_data.backend').alias('backend'),
        col('initial_data.num_qubits').alias('num_qubits'),
        posexplode(col('optimal_parameters')).alias('parameter_index', 'optimal_parameter_value'),
        col('processing_timestamp'),
        col('processing_date'),
        col('processing_batch_id'),
        col('processing_name'),
    ).withColumn(
        'parameter_id',
        concat(col('experiment_id'), lit('_opt_'), col('parameter_index').cast('string')),
    )

    # One row per (experiment, step) is kept. base_df already has one record per
    # experiment, so this guards a run that reports the same step twice: the steps
    # would repeat the same iteration_id inside one batch, and the key anti-join only
    # sees ROWS ALREADY IN THE TABLE, not duplicates within the batch. Which copy
    # survives is arbitrary.
    df_iterations = (
        base_df.select(
            col('experiment_id'),
            col('molecule_id'),
            col('basis_set'),
            col('initial_data.backend').alias('backend'),
            col('initial_data.num_qubits').alias('num_qubits'),
            explode(col('iteration_list')).alias('iteration'),
            col('processing_timestamp'),
            col('processing_date'),
            col('processing_batch_id'),
            col('processing_name'),
        )
        .select(
            col('experiment_id'),
            col('molecule_id'),
            col('basis_set'),
            col('backend'),
            col('num_qubits'),
            col('iteration.iteration').alias('iteration_step'),
            col('iteration.result').alias('iteration_energy'),
            col('iteration.std').alias('energy_std_dev'),
            col('iteration.energy_delta').alias('energy_delta'),
            col('iteration.parameter_delta_norm').alias('parameter_delta_norm'),
            col('iteration.cumulative_min_energy').alias('cumulative_min_energy'),
            col('processing_timestamp'),
            col('processing_date'),
            col('processing_batch_id'),
            col('processing_name'),
        )
        .withColumn('iteration_id', _iteration_id())
        .dropDuplicates(['experiment_id', 'iteration_step'])
    )

    df_iteration_parameters = (
        base_df.select(
            col('experiment_id'),
            col('molecule_id'),
            col('basis_set'),
            col('initial_data.backend').alias('backend'),
            col('initial_data.num_qubits').alias('num_qubits'),
            explode(col('iteration_list')).alias('iteration'),
            col('processing_timestamp'),
            col('processing_date'),
            col('processing_batch_id'),
            col('processing_name'),
        )
        .select(
            col('experiment_id'),
            col('molecule_id'),
            col('basis_set'),
            col('backend'),
            col('num_qubits'),
            col('iteration.iteration').alias('iteration_step'),
            posexplode(col('iteration.parameters')).alias('parameter_index', 'parameter_value'),
            col('processing_timestamp'),
            col('processing_date'),
            col('processing_batch_id'),
            col('processing_name'),
        )
        .withColumn('iteration_id', _iteration_id())
        .withColumn(
            'parameter_id',
            concat(col('iteration_id'), lit('_p'), col('parameter_index').cast('string')),
        )
    )

    df_hamiltonian = (
        base_df.select(
            col('experiment_id'),
            col('molecule_id'),
            col('basis_set'),
            col('initial_data.backend').alias('backend'),
            posexplode(col('initial_data.hamiltonian')).alias('term_index', 'hamiltonian_term'),
            col('processing_timestamp'),
            col('processing_date'),
            col('processing_batch_id'),
            col('processing_name'),
        )
        .select(
            col('experiment_id'),
            col('molecule_id'),
            col('basis_set'),
            col('backend'),
            col('term_index'),
            col('hamiltonian_term.label').alias('term_label'),
            col('hamiltonian_term.coefficients.real').alias('coeff_real'),
            col('hamiltonian_term.coefficients.imaginary').alias('coeff_imag'),
            col('processing_timestamp'),
            col('processing_date'),
            col('processing_batch_id'),
            col('processing_name'),
        )
        .withColumn(
            'term_id',
            concat(col('experiment_id'), lit('_term_'), col('term_index').cast('string')),
        )
    )

    tables = {
        'molecules': df_molecule,
        'ansatz_info': df_ansatz,
        'performance_metrics': df_metrics,
        'vqe_results': df_vqe,
        'initial_parameters': df_initial_parameters,
        'optimal_parameters': df_optimal_parameters,
        'vqe_iterations': df_iterations,
        'iteration_parameters': df_iteration_parameters,
        'hamiltonian_terms': df_hamiltonian,
    }
    return TransformResult(tables=tables, batch=batch, base_df=base_df)


def create_metadata_table_if_not_exists(spark):
    spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {CATALOG_FQN}.processing_metadata (
        processing_batch_id STRING,
        processing_name STRING,
        processing_timestamp TIMESTAMP,
        processing_date DATE,
        table_names ARRAY<STRING>,
        table_versions ARRAY<STRING>,
        record_counts ARRAY<BIGINT>,
        source_data_info STRING
    ) USING iceberg
    """)


def update_metadata_table(spark, batch, results, source_info):
    """Append one row to `processing_metadata` describing this run.

    Args:
        spark: SparkSession.
        batch: BatchInfo of this run (same values as in the written tables).
        results: dict of table name -> (version_tag or None, record_count).
        source_info: Free-text description of the data source.
    """
    processing_info = spark.createDataFrame(
        [
            {
                'processing_batch_id': batch.batch_id,
                'processing_name': batch.name,
                'processing_timestamp': batch.timestamp,
                'processing_date': batch.date,
                'table_names': list(results),
                'table_versions': [tag or 'no_changes' for tag, _ in results.values()],
                'record_counts': [count for _, count in results.values()],
                'source_data_info': source_info,
            }
        ]
    )

    processing_info.write.format('iceberg').mode('append').saveAsTable(
        f'{CATALOG_FQN}.processing_metadata'
    )

    logger.info('Updated metadata table with processing batch %s', batch.batch_id)


def get_table_configs():
    """Returns the configuration for each table."""
    return {
        'molecules': {
            'key_columns': ['experiment_id', 'molecule_id'],
            'partition_columns': ['processing_date'],
            'comment': 'Molecule information for quantum simulations',
        },
        'ansatz_info': {
            'key_columns': ['experiment_id', 'molecule_id'],
            'partition_columns': ['processing_date', 'basis_set'],
            'comment': 'Ansatz configurations for quantum simulations',
        },
        'performance_metrics': {
            'key_columns': ['experiment_id', 'molecule_id', 'basis_set'],
            'partition_columns': ['processing_date', 'basis_set'],
            'comment': 'Performance metrics for quantum simulations',
        },
        'vqe_results': {
            'key_columns': ['experiment_id', 'molecule_id', 'basis_set'],
            'partition_columns': ['processing_date', 'basis_set', 'backend'],
            'comment': 'VQE optimization results for quantum simulations',
        },
        'initial_parameters': {
            'key_columns': ['parameter_id'],
            'partition_columns': ['processing_date', 'basis_set'],
            'comment': 'Initial parameters for VQE optimization',
        },
        'optimal_parameters': {
            'key_columns': ['parameter_id'],
            'partition_columns': ['processing_date', 'basis_set'],
            'comment': 'Optimal parameters found by VQE optimization',
        },
        'vqe_iterations': {
            'key_columns': ['iteration_id'],
            'partition_columns': ['processing_date', 'basis_set', 'backend'],
            'comment': 'VQE optimization iterations and energy values',
        },
        'iteration_parameters': {
            'key_columns': ['parameter_id'],
            'partition_columns': ['processing_date', 'basis_set'],
            'comment': 'Parameters at each iteration of VQE optimization',
        },
        'hamiltonian_terms': {
            'key_columns': ['term_id'],
            'partition_columns': ['processing_date', 'basis_set', 'backend'],
            'comment': 'Hamiltonian terms for quantum simulations',
        },
    }


def process_experiments_incrementally(spark, df, topic_name=None):
    """Write all feature tables of one topic and record the run in the metadata table.

    Args:
        spark: SparkSession.
        df: Original dataframe with quantum simulation data.
        topic_name: Optional name of the topic being processed.

    Returns:
        dict: Table name -> number of new records written (0 if nothing changed).
    """
    create_metadata_table_if_not_exists(spark)

    transformed = transform_quantum_data(df)
    try:
        results = {}
        for table_name, config in get_table_configs().items():
            logger.info('Processing table: %s', table_name)
            results[table_name] = process_incremental_data(
                spark,
                transformed.tables[table_name],
                table_name,
                config['key_columns'],
                transformed.batch.batch_id,
                config['partition_columns'],
                config['comment'],
            )

        source_info = 'Incremental VQE simulation data processing' + (
            f' from topic {topic_name}' if topic_name else ''
        )
        update_metadata_table(spark, transformed.batch, results, source_info)
    finally:
        transformed.base_df.unpersist()

    logger.info('Incremental processing completed!')

    counts = {table: count for table, (_, count) in results.items()}

    processed_count = sum(1 for c in counts.values() if c > 0)
    logger.info(
        'Volume check: %d/%d tables received new records this batch',
        processed_count,
        len(counts),
    )

    logger.info(
        'Table row counts this batch: %s', ', '.join(f'{t}={c}' for t, c in counts.items())
    )

    iter_count = counts['vqe_iterations']
    results_count = counts['vqe_results']
    if results_count > 0 and iter_count < results_count:
        logger.warning(
            'Volume mismatch: vqe_iterations (%d) < vqe_results (%d) - '
            'possible truncated Kafka events for this batch',
            iter_count,
            results_count,
        )

    return counts


def check_for_new_data(spark, topic, bucket_path):
    """Read a topic and return its DataFrame unless it holds no records.

    Args:
        spark: SparkSession.
        topic: Topic name to read.
        bucket_path: S3 path that contains the topic directories.

    Returns:
        DataFrame: the (cached) topic data, or None if the topic is empty.
    """
    logger.info('Processing topic: %s', topic)

    df = read_experiments_by_topic(spark, bucket_path, topic, num_partitions=_RAW_NUM_PARTITIONS)

    if df.isEmpty():
        logger.info('No data available in topic %s', topic)
        return None

    return df


def main():
    """Main entry point for the quantum feature processing script."""
    # spark-submit does not configure Python logging, and the root logger
    # defaults to WARNING, so without this every logger.info is dropped.
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s'
    )

    spark = create_spark_session(_APP_NAME)

    try:
        bucket_path = S3_BUCKET_URL

        for topic in list_available_topics(spark, bucket_path):
            logger.info('Found topic: %s', topic)

            df = check_for_new_data(spark, topic, bucket_path)

            if df is None:
                logger.info('No new data in topic %s, skipping.', topic)
                continue

            results = process_experiments_incrementally(spark, df, topic)

            logger.info('Processing Summary for topic %s:', topic)
            for table, count in results.items():
                logger.info('%s: %d new records processed', table, count)
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
