"""
Quantum ML Feature Processing Script

Reads 5 of the normalized Iceberg tables produced by quantum_incremental_processing.py
(vqe_iterations, vqe_results, molecules, performance_metrics, ansatz_info) and joins them
into two ML-ready feature tables:

  - ml_iteration_features  (one row per iteration per experiment)
  - ml_run_summary         (one row per VQE run, aggregated from iterations)

Each table is written incrementally and gated on its own: for every target table only the
experiment_ids not yet present in THAT table are processed. A crash between the two writes
therefore self-heals on the next run (the run missing from ml_run_summary is rebuilt and
only that table is written), instead of leaving a run without a summary row forever.

Run-level labels (look-ahead):
  Some ml_iteration_features columns are properties of the WHOLE run, repeated unchanged on
  every iteration row: convergence_iteration, total_iterations, final_energy, converged,
  relative_iteration (step / total_iterations), vqe_time and total_time. At iteration k they
  already contain information from iterations after k (or from the run's end), so for any
  "predict from the first K steps" model they are look-ahead. quantum_pipeline/ml/ must
  never read them as features; they are labels or bookkeeping only. Every other column at
  step k is computed from steps <= k of the same run (tests/stream/
  test_ml_feature_processing.py checks this by truncating runs).

Conventions the code relies on:
  - iteration_step is 1-based and unique per experiment_id. Every window below orders by it,
    and a tie would make lag/row_number pick an arbitrary row, so validate_iteration_features
    raises if (experiment_id, iteration_step) is not unique.
  - `converged` is the optimizer's success flag (scipy `success`, renamed from vqe_results.
    success). It means the optimizer met ITS OWN stopping criterion, not that the energy is
    accurate; a run can "converge" onto a poor local minimum.
  - Step 1 has no previous point, so energy_delta, parameter_delta_norm, mean_param_change
    and energy_moving_std_5 are NULL there.
"""

import logging
import sys
from datetime import date
from functools import reduce
from operator import or_

sys.path.insert(0, '/opt/airflow/dags')

from common.pipeline_config import (
    CATALOG_FQN,
    ML_ROLLING_WINDOW,
    ML_TRAJECTORY_HEAD,
    ML_TRAJECTORY_TAIL,
)
from common.spark_factory import create_spark_session
from pyspark.sql import Window
from pyspark.sql.functions import (
    abs as spark_abs,
)
from pyspark.sql.functions import (
    avg,
    col,
    count,
    first,
    lag,
    lit,
    regr_slope,
    row_number,
    size,
    stddev,
    when,
)
from pyspark.sql.functions import (
    max as spark_max,
)
from pyspark.sql.functions import (
    min as spark_min,
)
from pyspark.sql.functions import (
    sum as spark_sum,
)

logger = logging.getLogger(__name__)

_APP_NAME = 'Quantum ML Feature Processing'

# Columns that come from the left-joined context tables and are written once, so a NULL
# here would stay NULL in Iceberg forever (the run is never reprocessed once its id exists
# in the table). The first four are also the grouping/partition keys of ml_run_summary;
# the rest are the run-level labels of vqe_results / performance_metrics.
_REQUIRED_COLUMNS = [
    'molecule_name',
    'optimizer',
    'basis_set',
    'num_qubits',
    'total_iterations',
    'final_energy',
    'converged',
    'vqe_time',
    'total_time',
]

# Upper bound on experiment_ids printed in the NULL-drop warning, to keep the log readable.
_MAX_LOGGED_IDS = 20


def get_new_experiment_ids(spark, source_experiment_ids, target_table):
    """
    Return the experiment_ids from source that are not yet in target_table, as a list.

    The result is collected to the driver on purpose. A lazy left_anti join against the
    target table would be re-evaluated every time it is used, and after the target has been
    appended to it would silently return nothing (the ids are in the table by then). A
    plain Python list is immune to that.

    Args:
        spark: SparkSession
        source_experiment_ids: DataFrame with a single column 'experiment_id'
        target_table: Fully-qualified Iceberg table name

    Returns:
        list: sorted new experiment_ids (all source ids if the table does not exist yet)
    """
    new_ids = source_experiment_ids
    if spark.catalog.tableExists(target_table):
        existing = spark.sql(f'SELECT DISTINCT experiment_id FROM {target_table}')  # noqa: S608
        new_ids = source_experiment_ids.join(existing, on='experiment_id', how='left_anti')

    return sorted(row['experiment_id'] for row in new_ids.collect())


def experiment_ids_to_df(spark, experiment_ids, schema):
    """
    Turn a collected id list back into a small in-memory DataFrame.

    Args:
        spark: SparkSession
        experiment_ids: list of experiment_ids
        schema: StructType with the single 'experiment_id' column (keeps the source type)

    Returns:
        DataFrame: one row per id; used to filter other frames via left_semi joins
    """
    return spark.createDataFrame([(i,) for i in experiment_ids], schema=schema)


def load_source_tables(spark):
    """
    Load the 5 source Iceberg tables needed for ML feature materialization.

    Returns:
        dict: DataFrames keyed by table name
    """
    tables = ['vqe_iterations', 'vqe_results', 'molecules', 'performance_metrics', 'ansatz_info']
    return {t: spark.table(f'{CATALOG_FQN}.{t}') for t in tables}


def build_iteration_features(source_dfs, new_experiment_ids_df, processing_date):
    """
    Join source tables and compute per-iteration ML features for the given experiment_ids.

    Derived features computed via Spark window functions, all ordered by iteration_step
    within one experiment_id:
      - is_new_minimum, steps_since_improvement
      - energy_moving_avg_5, energy_moving_std_5
      - relative_iteration, energy_improvement_rate, normalized_energy
      - mean_param_change, convergence_iteration

    See the module docstring for which output columns are run-level labels (look-ahead).

    Args:
        source_dfs: dict of source DataFrames
        new_experiment_ids_df: DataFrame of experiment_ids to process
        processing_date: `datetime.date` stamped on every row; computed once by the caller
            so both output tables carry the same date even if the job crosses midnight

    Returns:
        DataFrame: ml_iteration_features for the given experiments
    """
    iters = source_dfs['vqe_iterations']
    vqe = source_dfs['vqe_results']
    mols = source_dfs['molecules']
    perf = source_dfs['performance_metrics']
    ansatz = source_dfs['ansatz_info']

    # left_semi keeps only the iteration rows, without adding columns or multiplying rows
    # if the id frame ever contained a duplicate.
    iters = iters.join(new_experiment_ids_df.select('experiment_id'), 'experiment_id', 'left_semi')

    # num_qubits/basis_set also live in the context tables joined below, so keeping both
    # copies would make every later reference ambiguous. Removing them upstream in
    # quantum_incremental_processing.py instead is still open.
    iters = iters.drop('num_qubits', 'basis_set')

    vqe_cols = vqe.select(
        'experiment_id',
        col('optimizer'),
        col('ansatz_reps'),
        col('seed'),
        col('default_shots'),
        col('num_qubits'),
        col('total_iterations'),
        col('minimum_energy').alias('final_energy'),
        # scipy's `success`: the optimizer met its own stopping criterion. Says nothing
        # about whether the energy is close to the true ground state.
        # TODO: success is nullable in Avro; a NULL row is dropped by validation on every run
        col('success').alias('converged'),
    )

    # basis_set and ansatz_type sourced from ansatz_info (consistently populated
    # for both Avro/Kafka Connect and JSON/Redpanda Connect data)
    ansatz_cols = ansatz.select(
        'experiment_id',
        col('basis_set'),
        col('ansatz_name').alias('ansatz_type'),
        # TODO: the solver stores the requested init_strategy, even when it fell back to random
        col('init_strategy'),
    )

    mol_cols = mols.select(
        'experiment_id',
        # TODO: molecule_name is only the joined symbols, so geometries/charges share a name
        col('molecule_name'),
        size(col('atom_symbols')).alias('num_atoms'),
        col('charge'),
        col('multiplicity'),
    )

    perf_cols = perf.select(
        'experiment_id',
        col('vqe_time'),
        col('total_time'),
    )

    # Left joins keep every iteration row even when a context row is missing (the columns
    # are then NULL); validate_iteration_features decides what to do with those rows.
    df = (
        iters.join(vqe_cols, on='experiment_id', how='left')
        .join(ansatz_cols, on='experiment_id', how='left')
        .join(mol_cols, on='experiment_id', how='left')
        .join(perf_cols, on='experiment_id', how='left')
    )

    # w_exp_ord: one run, steps in order; with an orderBy Spark's default frame is "start of
    # partition up to the current row", which is what `first()` below relies on.
    w_exp_ord = Window.partitionBy('experiment_id').orderBy('iteration_step')
    w_exp_full = Window.partitionBy('experiment_id')
    # Trailing window of ML_ROLLING_WINDOW rows INCLUDING the current one (the `_5` in the
    # column names is the default width of 5).
    w_rolling = w_exp_ord.rowsBetween(-(ML_ROLLING_WINDOW - 1), 0)
    w_cumul = w_exp_ord.rowsBetween(Window.unboundedPreceding, 0)

    # is_new_minimum
    # --
    # cumulative_min_energy comes from the solver and already includes the current step, so
    # comparing a row's own energy to it can never be "strictly below" (at best equal).
    # The previous row's value is the best seen BEFORE this step, hence lag(). Strict `<`:
    # an energy equal to the running best is not a new minimum. lag() is NULL on step 1,
    # which is always True, so step 1 counts as a "new minimum" and later feeds
    # num_new_minima / improvement_ratio in ml_run_summary.
    prev_cumulative_min = lag('cumulative_min_energy').over(w_exp_ord)
    df = df.withColumn(
        'is_new_minimum',
        when(prev_cumulative_min.isNull(), lit(True))
        .when(col('cumulative_min_energy') < prev_cumulative_min, lit(True))
        .otherwise(lit(False)),
    )

    # improvement_group: monotonically increasing counter per experiment, increments at new minima
    df = df.withColumn(
        'improvement_group',
        spark_sum(when(col('is_new_minimum'), lit(1)).otherwise(lit(0))).over(w_cumul),
    )

    # steps_since_improvement: 0 at the row where a new minimum was found, counts up after.
    # Numbering rows inside each (experiment_id, improvement_group) block gives this without
    # a self-join.
    df = df.withColumn(
        'steps_since_improvement',
        row_number().over(
            Window.partitionBy('experiment_id', 'improvement_group').orderBy('iteration_step')
        )
        - 1,
    )

    # convergence_iteration: last iteration_step at which a new minimum was found.
    # RUN-LEVEL LABEL: w_exp_full spans the whole run, so on step k this already knows
    # about steps > k (look-ahead for any horizon-K feature set).
    df = df.withColumn(
        'convergence_iteration',
        spark_max(when(col('is_new_minimum'), col('iteration_step'))).over(w_exp_full),
    )

    # Rolling statistics (width ML_ROLLING_WINDOW; column names keep the default width 5).
    # Spark's stddev is the SAMPLE std (n-1), so it is NULL on a one-row window, i.e. on
    # step 1; from step 2 on it is defined.
    # TODO: _5 hardcodes the default ML_ROLLING_WINDOW; pin the width or derive the names
    df = df.withColumn('energy_moving_avg_5', avg('iteration_energy').over(w_rolling)).withColumn(
        'energy_moving_std_5', stddev('iteration_energy').over(w_rolling)
    )

    df = df.withColumn('_initial_energy', first('iteration_energy').over(w_exp_ord))

    # mean_param_change
    # --
    # Running mean of parameter_delta_norm over steps 1..k. The step-1 value is NULL (no
    # previous parameters) and sum()/count() both skip NULLs, so the divisor is the number
    # of deltas actually present (k - 1 at step k), not iteration_step. At step k = n this
    # equals avg('parameter_delta_norm') over the run in ml_run_summary
    # (mean_param_delta_norm). It is NULL at step 1, where the count is 0; the when() guard
    # keeps ANSI mode from raising on 0/0.
    cumul_param_norm = spark_sum('parameter_delta_norm').over(w_cumul)
    cumul_param_count = count('parameter_delta_norm').over(w_cumul)
    df = df.withColumn(
        'mean_param_change',
        when(cumul_param_count > 0, cumul_param_norm / cumul_param_count.cast('double')),
    )

    df = (
        df
        # RUN-LEVEL LABEL: divides by the final iteration count of the run (look-ahead).
        .withColumn(
            'relative_iteration',
            col('iteration_step').cast('double') / col('total_iterations').cast('double'),
        )
        # (first energy - best so far) / steps taken: average improvement per step.
        .withColumn(
            'energy_improvement_rate',
            (col('_initial_energy') - col('cumulative_min_energy'))
            / col('iteration_step').cast('double'),
        )
        # Distance above the best energy so far, relative to its magnitude. The `!= 0`
        # guard avoids a divide-by-zero, which Spark's ANSI mode would turn into an error.
        .withColumn(
            'normalized_energy',
            when(
                col('cumulative_min_energy') != 0,
                (col('iteration_energy') - col('cumulative_min_energy'))
                / spark_abs(col('cumulative_min_energy')),
            ).otherwise(lit(0.0)),
        )
    )

    return df.select(
        # identity
        col('experiment_id'),
        col('iteration_step'),
        # molecule context
        col('molecule_name'),
        col('num_atoms'),
        col('num_qubits'),
        col('charge'),
        col('multiplicity'),
        col('basis_set'),
        # configuration context
        col('optimizer'),
        col('ansatz_type'),
        col('ansatz_reps'),
        col('init_strategy'),
        col('seed'),
        col('default_shots'),
        # per-iteration raw signals (energy_delta / parameter_delta_norm are NULL on step 1)
        col('iteration_energy').alias('energy'),
        col('energy_std_dev').alias('energy_std'),
        col('energy_delta'),
        col('parameter_delta_norm'),
        col('cumulative_min_energy'),
        # per-iteration derived features (relative_iteration is a run-level label, see above)
        col('relative_iteration'),
        col('energy_improvement_rate'),
        col('normalized_energy'),
        col('is_new_minimum'),
        col('steps_since_improvement'),
        col('energy_moving_avg_5'),
        col('energy_moving_std_5'),
        col('mean_param_change'),
        # run-level labels: identical on every row of a run, look-ahead for horizon-K features
        col('total_iterations'),
        col('final_energy'),
        col('converged'),
        col('convergence_iteration'),
        # timing (also run-level)
        col('vqe_time'),
        col('total_time'),
        lit(processing_date).alias('processing_date'),
    )


def validate_iteration_features(df_iter, source_dfs, experiment_ids_df):
    """
    Check the joined frame against its source and drop rows with unusable NULLs.

    Order matters: row-count guards run BEFORE the NULL drop, because dropping rows would
    otherwise be indistinguishable from a join that lost rows.

    Args:
        df_iter: output of build_iteration_features (should be persisted, it is scanned
            several times here)
        source_dfs: dict of source DataFrames
        experiment_ids_df: the experiment_ids that df_iter was built for

    Returns:
        DataFrame: df_iter without rows that have a NULL in `_REQUIRED_COLUMNS`

    Raises:
        RuntimeError: if the join changed the row count (fan-out or lost rows), or if
            (experiment_id, iteration_step) is not unique
    """
    source_count = (
        source_dfs['vqe_iterations']
        .join(experiment_ids_df.select('experiment_id'), 'experiment_id', 'left_semi')
        .count()
    )
    joined_count = df_iter.count()

    # Left joins onto one-row-per-experiment tables must neither add nor remove rows, so
    # `!=` (not `>`): a duplicated vqe_results row multiplies iteration rows, while a
    # broken join key could lose them.
    if joined_count != source_count:
        raise RuntimeError(
            f'Join fan-out detected: ml_iteration_features has {joined_count} rows '
            f'but source vqe_iterations for the selected experiments has {source_count} rows. '
            f'Check for many-to-many join explosion in build_iteration_features().'
        )

    # Every window orders by iteration_step; with a tie, lag()/row_number() would pick an
    # arbitrary row and the features would differ between runs of the job.
    distinct_count = df_iter.select('experiment_id', 'iteration_step').distinct().count()
    if distinct_count != joined_count:
        raise RuntimeError(
            f'(experiment_id, iteration_step) is not unique: {joined_count} rows but '
            f'{distinct_count} distinct keys. Window features would be nondeterministic.'
        )

    # A NULL here means a left join missed its context row (or the run-level label is not
    # there yet). The first four columns are the grouping keys of the summary table and
    # basis_set partitions the write; the rest would be frozen as NULL, because once an
    # id is in the table it is never processed again. Dropping the rows instead leaves the
    # id "new", so the next run retries it after the missing data has arrived.
    has_null = reduce(or_, [col(c).isNull() for c in _REQUIRED_COLUMNS])
    bad_rows = df_iter.filter(has_null)
    bad_count = bad_rows.count()
    if bad_count > 0:
        bad_ids = [
            r['experiment_id']
            for r in bad_rows.select('experiment_id')
            .distinct()
            .orderBy('experiment_id')
            .limit(_MAX_LOGGED_IDS + 1)
            .collect()
        ]
        logger.warning(
            'Dropping %d row(s) with NULLs in %s after joins; experiment_id(s): %s%s',
            bad_count,
            _REQUIRED_COLUMNS,
            ', '.join(str(i) for i in bad_ids[:_MAX_LOGGED_IDS]),
            ' (and more)' if len(bad_ids) > _MAX_LOGGED_IDS else '',
        )

    return df_iter.na.drop(subset=_REQUIRED_COLUMNS)


def build_run_summary(df_iter, processing_date):
    """
    Aggregate ml_iteration_features into ml_run_summary (one row per experiment).

    Every column here describes the finished run, so none of it is usable as an input
    feature for early prediction; it is for labels, analysis and dashboards.

    Args:
        df_iter: ml_iteration_features DataFrame (experiments to summarise)
        processing_date: `datetime.date`, the same value used for df_iter

    Returns:
        DataFrame: ml_run_summary for those experiments
    """
    w_exp_ord = Window.partitionBy('experiment_id').orderBy('iteration_step')

    # Rank each row from both ends of its run so the first/last N steps can be selected with
    # a plain filter. The `first_10_*` / `last_10_*` column names are the defaults of
    # ML_TRAJECTORY_HEAD / ML_TRAJECTORY_TAIL; the real width follows those settings.
    df_with_rank = df_iter.withColumn('_rn_asc', row_number().over(w_exp_ord)).withColumn(
        '_rn_desc',
        row_number().over(
            Window.partitionBy('experiment_id').orderBy(col('iteration_step').desc())
        ),
    )

    # TODO: first_10/last_10 hardcode the defaults of ML_TRAJECTORY_HEAD/TAIL (both 10)
    first_10 = df_with_rank.filter(col('_rn_asc') <= ML_TRAJECTORY_HEAD)
    last_10 = df_with_rank.filter(col('_rn_desc') <= ML_TRAJECTORY_TAIL)

    # regr_slope needs at least 2 points, so first_10_energy_slope is NULL for a one-row run.
    first_10_agg = first_10.groupBy('experiment_id').agg(
        avg('energy').alias('first_10_mean_energy'),
        regr_slope(col('energy'), col('iteration_step').cast('double')).alias(
            'first_10_energy_slope'
        ),
    )

    last_10_agg = last_10.groupBy('experiment_id').agg(
        avg('energy').alias('last_10_mean_energy'),
    )

    # The long groupBy lists columns that are constant within a run (so grouping by them
    # does not split a run), which keeps them in the output without an extra first().
    summary = df_iter.groupBy(
        # identity
        'experiment_id',
        # molecule + config (constant per experiment)
        'molecule_name',
        'num_atoms',
        'num_qubits',
        'charge',
        'multiplicity',
        'basis_set',
        'optimizer',
        'ansatz_type',
        'ansatz_reps',
        'init_strategy',
        'seed',
        'default_shots',
        # run-level labels and timing (constant per experiment)
        'total_iterations',
        'final_energy',
        'converged',
        'convergence_iteration',
        'vqe_time',
        'total_time',
    ).agg(
        # energy-delta stats (mean/std ignore the NULL step-1 delta)
        avg('energy_delta').alias('mean_energy_delta'),
        stddev('energy_delta').alias('std_energy_delta'),
        # energy range over the whole run
        spark_min('energy').alias('min_energy'),
        spark_max('energy').alias('max_energy'),
        (spark_max('energy') - spark_min('energy')).alias('energy_range'),
        # parameter movement (also ignores the NULL step-1 value)
        avg('parameter_delta_norm').alias('mean_param_delta_norm'),
        stddev('parameter_delta_norm').alias('std_param_delta_norm'),
        spark_sum('parameter_delta_norm').alias('total_param_distance'),
        # improvement behaviour: count(when(...)) counts only the True rows. Step 1 is
        # always a "new minimum", so num_new_minima >= 1 for every run.
        count(when(col('is_new_minimum'), True)).alias('num_new_minima'),
        # longest stretch without improvement; a plateau still open at the last step counts
        spark_max('steps_since_improvement').alias('longest_plateau'),
        (
            count(when(col('is_new_minimum'), True)).cast('double')
            / col('total_iterations').cast('double')
        ).alias('improvement_ratio'),
        # partitioning
        lit(processing_date).alias('processing_date'),
    )

    return (
        summary.join(first_10_agg, on='experiment_id', how='left')
        .join(last_10_agg, on='experiment_id', how='left')
        .withColumn(
            'time_per_iteration',
            col('vqe_time') / col('total_iterations').cast('double'),
        )
    )


def write_incremental(spark, df, table_name, partition_columns):
    """
    Write a DataFrame to an Iceberg table, creating it on first run or appending.

    Known limitation: there is no full-refresh path, and the first write uses
    mode('overwrite'), which replaces the table if it was created concurrently.

    Args:
        spark: SparkSession
        df: DataFrame to write
        table_name: Unqualified table name (written under CATALOG_FQN)
        partition_columns: list of partition column names
    """
    full_name = f'{CATALOG_FQN}.{table_name}'
    table_exists = spark.catalog.tableExists(full_name)

    writer = df.write.format('iceberg').option('write-format', 'parquet')
    if partition_columns:
        writer = writer.partitionBy(*partition_columns)

    if table_exists:
        writer.mode('append').saveAsTable(full_name)
        logger.info('Appended to %s', full_name)
    else:
        writer.mode('overwrite').saveAsTable(full_name)
        logger.info('Created %s', full_name)


def main():
    logging.basicConfig(level=logging.INFO)
    spark = create_spark_session(_APP_NAME)
    persisted = []

    try:
        source_dfs = load_source_tables(spark)
        all_experiment_ids = source_dfs['vqe_iterations'].select('experiment_id').distinct()

        # Each table is gated on its own contents. Both lists are collected NOW, before any
        # write: a lazy anti-join would be re-evaluated after the first append (and Spark
        # also recaches plans that read a table it appended to), at which point the ids
        # are "already present" and the second table would get 0 rows.
        iter_table = f'{CATALOG_FQN}.ml_iteration_features'
        summary_table = f'{CATALOG_FQN}.ml_run_summary'
        iter_ids = get_new_experiment_ids(spark, all_experiment_ids, iter_table)
        summary_ids = get_new_experiment_ids(spark, all_experiment_ids, summary_table)

        if not iter_ids and not summary_ids:
            logger.info('No new experiments to process for ML feature tables.')
            return

        logger.info(
            'Processing %d new experiment(s) for ml_iteration_features and %d for ml_run_summary.',
            len(iter_ids),
            len(summary_ids),
        )

        id_schema = all_experiment_ids.schema
        todo_df = experiment_ids_to_df(spark, sorted(set(iter_ids) | set(summary_ids)), id_schema)
        iter_ids_df = experiment_ids_to_df(spark, iter_ids, id_schema)
        summary_ids_df = experiment_ids_to_df(spark, summary_ids, id_schema)

        # One date for both tables, even if the job runs past midnight.
        processing_date = date.today()

        # Build the features once for the union of ids (the summary is derived from the
        # iteration rows, so a run missing only its summary still needs them).
        df_all = build_iteration_features(source_dfs, todo_df, processing_date).persist()
        persisted.append(df_all)
        df_clean = validate_iteration_features(df_all, source_dfs, todo_df)

        # Build BOTH output frames and materialise them before the first write. Their plans
        # read only source tables and in-memory id lists, never the ml_* tables, so the
        # writes below cannot invalidate these caches or change what the second one holds.
        df_iter = df_clean.join(iter_ids_df, 'experiment_id', 'left_semi').persist()
        persisted.append(df_iter)
        df_summary = build_run_summary(
            df_clean.join(summary_ids_df, 'experiment_id', 'left_semi'), processing_date
        ).persist()
        persisted.append(df_summary)
        iter_rows = df_iter.count()
        summary_rows = df_summary.count()

        if iter_rows > 0:
            write_incremental(spark, df_iter, 'ml_iteration_features', ['processing_date'])
            logger.info('ml_iteration_features: %d row(s) written', iter_rows)
        if summary_rows > 0:
            write_incremental(
                spark, df_summary, 'ml_run_summary', ['processing_date', 'basis_set']
            )
            logger.info('ml_run_summary: %d row(s) written', summary_rows)

        logger.info('ML feature processing completed.')

    finally:
        for df in persisted:
            df.unpersist()
        spark.stop()


if __name__ == '__main__':
    main()
