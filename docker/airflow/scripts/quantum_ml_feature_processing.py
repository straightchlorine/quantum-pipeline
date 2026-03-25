"""
Quantum ML Feature Processing Script

Reads the 9 normalized Iceberg tables produced by quantum_incremental_processing.py
and joins them into two ML-ready feature tables:

  - ml_iteration_features  (one row per iteration per experiment)
  - ml_run_summary         (one row per VQE run, aggregated from iterations)

Both tables are written incrementally: only experiment_ids not yet present in the
target table are processed, preventing re-computation of already-materialized runs.
"""

import logging
import os

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    abs as spark_abs,
    avg,
    col,
    count,
    current_date,
    first,
    lag,
    last,
    lit,
    max as spark_max,
    min as spark_min,
    row_number,
    size,
    stddev,
    sum as spark_sum,
    when,
    regr_slope,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    'S3_BUCKET': os.getenv('S3_BUCKET_URL', 's3a://raw-results/experiments/'),
    'APP_NAME': 'Quantum ML Feature Processing',
}

CATALOG = 'quantum_catalog.quantum_features'


def create_spark_session(config=None):
    """Create a Spark session. Infrastructure config (JARs, S3, Iceberg catalog) is
    supplied via spark-defaults.conf mounted into the container."""
    if config is None:
        config = DEFAULT_CONFIG
    return (
        SparkSession.builder
        .appName(config.get('APP_NAME', DEFAULT_CONFIG['APP_NAME']))
        .getOrCreate()
    )


def get_new_experiment_ids(spark, source_experiment_ids, target_table):
    """
    Return only the experiment_ids from source that are not yet in target_table.

    Args:
        spark: SparkSession
        source_experiment_ids: DataFrame with a single column 'experiment_id'
        target_table: Fully-qualified Iceberg table name

    Returns:
        DataFrame: filtered source_experiment_ids with only new IDs
    """
    if not spark.catalog.tableExists(target_table):
        return source_experiment_ids

    existing = spark.sql(f'SELECT DISTINCT experiment_id FROM {target_table}')  # noqa: S608
    if existing.isEmpty():
        return source_experiment_ids

    return source_experiment_ids.join(existing, on='experiment_id', how='left_anti')


def load_source_tables(spark):
    """
    Load the 5 source Iceberg tables needed for ML feature materialization.

    Returns:
        dict: DataFrames keyed by table name
    """
    tables = ['vqe_iterations', 'vqe_results', 'molecules', 'performance_metrics']
    return {t: spark.table(f'{CATALOG}.{t}') for t in tables}


def build_iteration_features(spark, source_dfs, new_experiment_ids_df):
    """
    Join source tables and compute per-iteration ML features for new experiment_ids only.

    Derived features computed via Spark window functions:
      - is_new_minimum, steps_since_improvement
      - energy_moving_avg_5, energy_moving_std_5
      - relative_iteration, energy_improvement_rate, normalized_energy
      - mean_param_change, convergence_iteration

    Args:
        spark: SparkSession
        source_dfs: dict of source DataFrames
        new_experiment_ids_df: DataFrame of experiment_ids to process

    Returns:
        DataFrame: ml_iteration_features for the new experiments
    """
    iters = source_dfs['vqe_iterations']
    vqe = source_dfs['vqe_results']
    mols = source_dfs['molecules']
    perf = source_dfs['performance_metrics']

    # filter to new experiment_ids only
    new_ids = new_experiment_ids_df.select('experiment_id')
    iters = iters.join(new_ids, on='experiment_id', how='inner')

    # --- join context tables (all on experiment_id) ---
    vqe_cols = vqe.select(
        'experiment_id',
        col('optimizer'),
        col('ansatz_name').alias('ansatz_type'),
        col('ansatz_reps'),
        col('init_strategy'),
        col('seed'),
        col('default_shots'),
        col('num_qubits'),
        col('total_iterations'),
        col('minimum_energy').alias('final_energy'),
        col('success').alias('converged'),
        col('basis_set'),
    )

    mol_cols = mols.select(
        'experiment_id',
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

    df = (
        iters.join(vqe_cols, on='experiment_id', how='left')
        .join(mol_cols, on='experiment_id', how='left')
        .join(perf_cols, on='experiment_id', how='left')
    )

    # --- window specs ---
    w_exp_ord = Window.partitionBy('experiment_id').orderBy('iteration_step')
    w_exp_full = Window.partitionBy('experiment_id')
    w_5 = w_exp_ord.rowsBetween(-4, 0)
    w_cumul = w_exp_ord.rowsBetween(Window.unboundedPreceding, 0)

    # is_new_minimum: first iteration always establishes minimum; thereafter when
    # cumulative_min_energy drops below the previous row's value.
    df = df.withColumn(
        'is_new_minimum',
        when(lag('cumulative_min_energy').over(w_exp_ord).isNull(), lit(True))
        .when(
            col('cumulative_min_energy') < lag('cumulative_min_energy').over(w_exp_ord),
            lit(True),
        )
        .otherwise(lit(False)),
    )

    # improvement_group: monotonically increasing counter per experiment, increments at new minima
    df = df.withColumn(
        'improvement_group',
        spark_sum(when(col('is_new_minimum'), lit(1)).otherwise(lit(0))).over(w_cumul),
    )

    # steps_since_improvement: 0 at the row where a new minimum was found, counts up after
    df = df.withColumn(
        'steps_since_improvement',
        row_number().over(
            Window.partitionBy('experiment_id', 'improvement_group').orderBy('iteration_step')
        )
        - 1,
    )

    # convergence_iteration: last iteration_step at which a new minimum was found
    df = df.withColumn(
        'convergence_iteration',
        spark_max(when(col('is_new_minimum'), col('iteration_step'))).over(w_exp_full),
    )

    # rolling statistics (window=5)
    df = (
        df.withColumn('energy_moving_avg_5', avg('iteration_energy').over(w_5))
        .withColumn('energy_moving_std_5', stddev('iteration_energy').over(w_5))
    )

    # initial_energy: energy at the first iteration of each experiment
    df = df.withColumn('_initial_energy', first('iteration_energy').over(w_exp_ord))

    # cumulative mean parameter change
    cumul_param_norm = spark_sum('parameter_delta_norm').over(w_cumul)
    df = df.withColumn(
        'mean_param_change', cumul_param_norm / col('iteration_step').cast('double')
    )

    # derived per-iteration features
    df = (
        df.withColumn(
            'relative_iteration',
            col('iteration_step').cast('double') / col('total_iterations').cast('double'),
        )
        .withColumn(
            'energy_improvement_rate',
            (col('_initial_energy') - col('cumulative_min_energy'))
            / col('iteration_step').cast('double'),
        )
        .withColumn(
            'normalized_energy',
            when(
                col('cumulative_min_energy') != 0,
                (col('iteration_energy') - col('cumulative_min_energy'))
                / spark_abs(col('cumulative_min_energy')),
            ).otherwise(lit(0.0)),
        )
    )

    # select and rename final columns per thesis schema
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
        # per-iteration raw signals
        col('iteration_energy').alias('energy'),
        col('energy_std_dev').alias('energy_std'),
        col('energy_delta'),
        col('parameter_delta_norm'),
        col('cumulative_min_energy'),
        # per-iteration derived features
        col('relative_iteration'),
        col('energy_improvement_rate'),
        col('normalized_energy'),
        col('is_new_minimum'),
        col('steps_since_improvement'),
        col('energy_moving_avg_5'),
        col('energy_moving_std_5'),
        col('mean_param_change'),
        # run-level labels
        col('total_iterations'),
        col('final_energy'),
        col('converged'),
        col('convergence_iteration'),
        # timing
        col('vqe_time'),
        col('total_time'),
        # partitioning
        current_date().alias('processing_date'),
    ).drop('improvement_group', '_initial_energy')


def build_run_summary(df_iter):
    """
    Aggregate ml_iteration_features into ml_run_summary (one row per experiment).

    Args:
        df_iter: ml_iteration_features DataFrame (new experiments only)

    Returns:
        DataFrame: ml_run_summary for the new experiments
    """
    w_exp_ord = Window.partitionBy('experiment_id').orderBy('iteration_step')

    # first/last energy for early-vs-late trajectory analysis
    df_with_rank = df_iter.withColumn('_rn_asc', row_number().over(w_exp_ord)).withColumn(
        '_rn_desc',
        row_number().over(
            Window.partitionBy('experiment_id').orderBy(col('iteration_step').desc())
        ),
    )

    first_10 = df_with_rank.filter(col('_rn_asc') <= 10)
    last_10 = df_with_rank.filter(col('_rn_desc') <= 10)

    first_10_agg = first_10.groupBy('experiment_id').agg(
        avg('energy').alias('first_10_mean_energy'),
        regr_slope(col('energy'), col('iteration_step').cast('double')).alias(
            'first_10_energy_slope'
        ),
    )

    last_10_agg = last_10.groupBy('experiment_id').agg(
        avg('energy').alias('last_10_mean_energy'),
    )

    # main aggregation
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
        # run-level labels (same for all rows)
        'total_iterations',
        'final_energy',
        'converged',
        'convergence_iteration',
        # timing (same for all rows)
        'vqe_time',
        'total_time',
    ).agg(
        # energy trajectory shape
        avg('energy_delta').alias('mean_energy_delta'),
        stddev('energy_delta').alias('std_energy_delta'),
        spark_min('energy').alias('min_energy'),
        spark_max('energy').alias('max_energy'),
        (spark_max('energy') - spark_min('energy')).alias('energy_range'),
        # parameter movement
        avg('parameter_delta_norm').alias('mean_param_delta_norm'),
        stddev('parameter_delta_norm').alias('std_param_delta_norm'),
        spark_sum('parameter_delta_norm').alias('total_param_distance'),
        # convergence behavior
        count(when(col('is_new_minimum'), True)).alias('num_new_minima'),
        spark_max('steps_since_improvement').alias('longest_plateau'),
        (
            count(when(col('is_new_minimum'), True)).cast('double')
            / col('total_iterations').cast('double')
        ).alias('improvement_ratio'),
        # partitioning
        current_date().alias('processing_date'),
    )

    summary = (
        summary.join(first_10_agg, on='experiment_id', how='left')
        .join(last_10_agg, on='experiment_id', how='left')
        .withColumn(
            'time_per_iteration',
            col('vqe_time') / col('total_iterations').cast('double'),
        )
    )

    return summary


def write_incremental(spark, df, table_name, partition_columns):
    """
    Write a DataFrame to an Iceberg table, creating it on first run or appending.

    Args:
        spark: SparkSession
        df: DataFrame to write
        table_name: Unqualified table name (written under CATALOG)
        partition_columns: list of partition column names
    """
    full_name = f'{CATALOG}.{table_name}'
    table_exists = spark.catalog.tableExists(full_name)

    writer = df.write.format('iceberg').option('write-format', 'parquet')
    if partition_columns:
        writer = writer.partitionBy(*partition_columns)

    if table_exists:
        writer.mode('append').saveAsTable(full_name)
        logger.info(f'Appended to {full_name}: {df.count()} rows')
    else:
        writer.mode('overwrite').saveAsTable(full_name)
        logger.info(f'Created {full_name}: {df.count()} rows')


def main(config=None):
    """Main entry point for the ML feature processing script."""
    spark = create_spark_session(config)

    try:
        source_dfs = load_source_tables(spark)
        all_experiment_ids = source_dfs['vqe_iterations'].select('experiment_id').distinct()

        # find experiment_ids not yet materialized into ml_iteration_features
        new_ids = get_new_experiment_ids(
            spark, all_experiment_ids, f'{CATALOG}.ml_iteration_features'
        )

        if new_ids.isEmpty():
            logger.info('No new experiments to process for ML feature tables.')
            return

        new_count = new_ids.count()
        logger.info(f'Processing {new_count} new experiment(s) into ML feature tables.')

        # build and cache iteration features (used for both tables)
        df_iter = build_iteration_features(spark, source_dfs, new_ids).cache()

        # write ml_iteration_features
        write_incremental(
            spark,
            df_iter,
            'ml_iteration_features',
            partition_columns=['processing_date'],
        )

        # build and write ml_run_summary
        df_summary = build_run_summary(df_iter)
        write_incremental(
            spark,
            df_summary,
            'ml_run_summary',
            partition_columns=['processing_date', 'basis_set'],
        )

        df_iter.unpersist()
        logger.info('ML feature processing completed.')

    finally:
        spark.stop()


if __name__ == '__main__':
    main()
