"""
Quantum Feature Processing Module

This module handles incremental processing of vqe experiment data,
transforming raw data into feature tables stored in Iceberg format.
"""

import os
import uuid

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    current_date,
    current_timestamp,
    explode,
    expr,
    lit,
    size,
    udf,
)
from pyspark.sql.types import StringType

DEFAULT_CONFIG = {
    'S3_ENDPOINT': os.getenv('S3_ENDPOINT', 'http://minio:9000'),
    'SPARK_MASTER': os.getenv('SPARK_ENDPOINT', 'spark://spark-master:7077'),
    'S3_BUCKET': os.getenv('S3_BUCKET_URL', 's3a://local-vqe-results/experiments/'),
    'S3_WAREHOUSE': os.getenv('S3_WAREHOUSE_URL', 's3a://local-features/warehouse/'),
    'APP_NAME': 'Quantum Pipeline Feature Processing',
}


def validate_environment():
    """Validate that required environment variables are set."""
    required_vars = ['MINIO_ACCESS_KEY', 'MINIO_SECRET_KEY']
    missing = [var for var in required_vars if not os.environ.get(var)]

    if missing:
        raise ValueError(f'Missing required environment variables: {", ".join(missing)}')


def create_spark_session(config=None):
    """
    Create and configure a Spark session with Iceberg and S3 support.

    Args:
        config: Dictionary with configuration values (defaults used if None)

    Returns:
        SparkSession: Configured Spark session
    """
    if config is None:
        config = DEFAULT_CONFIG

    validate_environment()

    # access keys
    access_key = os.environ.get('MINIO_ACCESS_KEY')
    secret_key = os.environ.get('MINIO_SECRET_KEY')

    return (
        SparkSession.builder.appName(config.get('APP_NAME', DEFAULT_CONFIG['APP_NAME']))
        .master(config.get('SPARK_MASTER', DEFAULT_CONFIG['SPARK_MASTER']))
        # TODO: check if its required if jars are already in the image
        .config(
            'spark.jars.packages',
            (
                'org.slf4j:slf4j-api:2.0.17,'
                'commons-codec:commons-codec:1.18.0,'
                'com.google.j2objc:j2objc-annotations:3.0.0,'
                'org.apache.spark:spark-avro_2.12:3.5.5,'
                'org.apache.hadoop:hadoop-aws:3.3.1,'
                'org.apache.hadoop:hadoop-common:3.3.1,'
                'org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.4.2,'
            ),
        )
        .config(
            'spark.sql.extensions',
            'org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions',
        )
        .config('spark.sql.catalog.quantum_catalog', 'org.apache.iceberg.spark.SparkCatalog')
        .config('spark.sql.catalog.quantum_catalog.type', 'hadoop')
        .config(
            'spark.sql.catalog.quantum_catalog.warehouse',
            config.get('S3_WAREHOUSE', DEFAULT_CONFIG['S3_WAREHOUSE']),
        )
        .config('spark.hadoop.fs.s3a.impl', 'org.apache.hadoop.fs.s3a.S3AFileSystem')
        .config('spark.hadoop.fs.s3a.access.key', access_key)
        .config('spark.hadoop.fs.s3a.secret.key', secret_key)
        .config(
            'spark.hadoop.fs.s3a.endpoint',
            config.get('S3_ENDPOINT', DEFAULT_CONFIG['S3_ENDPOINT']),
        )
        .config('spark.hadoop.fs.s3a.path.style.access', 'true')
        .config('spark.hadoop.fs.s3a.connection.ssl.enabled', 'false')
        .config(
            'spark.hadoop.fs.s3a.aws.credentials.provider',
            'org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider',
        )
        .config('spark.sql.adaptive.enabled', 'true')
        .config('spark.sql.shuffle.partitions', '200')
        .getOrCreate()
    )


def list_available_topics(spark, bucket_path):
    """
    List available topics within experiments location from the storage.

    Args:
        spark: SparkSession
        bucket_path: S3 bucket path to check

    Returns:
        list: List of available topics
    """
    try:
        # configure hadoop to use appropriate filesystem
        spark._jsc.hadoopConfiguration().set(
            'fs.s3a.impl', 'org.apache.hadoop.fs.s3a.S3AFileSystem'
        )
        spark._jsc.hadoopConfiguration().set(
            'fs.s3a.aws.credentials.provider',
            'org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider',
        )

        # create a configured filesystem
        fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(
            spark._jvm.java.net.URI.create(bucket_path), spark._jsc.hadoopConfiguration()
        )

        path = spark._jvm.org.apache.hadoop.fs.Path(bucket_path)

        if fs.exists(path) and fs.isDirectory(path):
            return [f.getPath().getName() for f in fs.listStatus(path) if f.isDirectory()]
        return []
    except Exception as e:
        print(f'Error accessing S3: {e}')
        return []


def read_experiments_by_topic(spark, bucket_path, topic_name, num_partitions=None):
    """
    Read Avro experiment files from a specific topic's directory.

    Args:
        spark: SparkSession
        bucket_path: S3 bucket path
        topic_name: Name of the topic to read
        num_partitions: Optional number of partitions for the dataframe

    Returns:
        DataFrame: Spark DataFrame with the topic data
    """
    topic_path = f'{bucket_path}{topic_name}/partition=*/*.avro'
    df = spark.read.format('avro').load(topic_path)

    if num_partitions:
        df = df.repartition(num_partitions)

    # cache the dataframe as it will be used multiple times
    return df.cache()


def add_metadata_columns(dataframe, processing_name):
    """
    Add metadata columns to the dataframes.

    Args:
        dataframe: Spark DataFrame
        processing_name: Name of the processing job

    Returns:
        DataFrame: Dataframe with added metadata columns
    """
    #  udf to generate uuids
    generate_uuid = udf(lambda: str(uuid.uuid4()), StringType())

    return (
        dataframe.withColumn('experiment_id', generate_uuid())
        .withColumn('processing_timestamp', current_timestamp())
        .withColumn('processing_date', current_date())
        .withColumn('processing_batch_id', lit(str(uuid.uuid4())))
        .withColumn('processing_name', lit(processing_name))
    )


def identify_new_records(spark, new_data_df, table_name, key_columns):
    """
    Identifies records in new_data_df that don't exist in the target table.
    Uses DataFrame operations for better performance.

    Args:
        new_data_df: DataFrame containing potentially new data
        table_name: Name of the table to check against
        key_columns: List of column names that uniquely identify records

    Returns:
        DataFrame: records that don't exist in the target table
    """
    try:
        # if new_data_df empty, return empty df
        if new_data_df.isEmpty():
            return new_data_df

        # if table does not exist, every record is new
        if not spark.catalog.tableExists(f'quantum_catalog.quantum_features.{table_name}'):
            return new_data_df

        # retrieve existing keys
        existing_keys = spark.sql(
            f'SELECT DISTINCT {", ".join(key_columns)} FROM quantum_catalog.quantum_features.{table_name}'
        )

        # if table exists, but has no records - return all records
        if existing_keys.isEmpty():
            return new_data_df

        # marker column for the join
        new_with_marker = new_data_df.select(*key_columns).distinct().withColumn('is_new', lit(1))
        existing_with_marker = existing_keys.withColumn('exists', lit(1))

        # left join
        joined = new_with_marker.join(existing_with_marker, on=key_columns, how='left')

        # ensure join is not empty
        if joined.isEmpty():
            return new_data_df

        # get only the keys that do not exist in the records yet
        new_keys = joined.filter(col('exists').isNull()).select(*key_columns)

        # ensure new keys are not empty
        if new_keys.isEmpty():
            return new_data_df.limit(0)

        # join back to get an array with full records
        truly_new_data = new_data_df.join(new_keys, on=key_columns, how='inner')

        return truly_new_data

    except Exception as e:
        raise RuntimeError(f'Error identifying new records: {str(e)}') from e


def process_incremental_data(
    spark, new_data_df, table_name, key_columns, partition_columns=None, comment=None
):
    """
    Process the data incrementally - only data that wasn't processed yet.

    Args:
      spark: SparkSession
      new_data_df: DataFrame containing potentially new data
      table_name: Name of the target table
      key_columns: List of column names that uniquely identify records
      partition_columns: Optional list of columns to partition by
      comment: Optional comment for the table

    Returns:
        tuple:  version tag and count of new records processed
    """
    # check if table exists
    table_exists = spark.catalog._jcatalog.tableExists(
        f'quantum_catalog.quantum_features.{table_name}'
    )

    # if the table doesn't exist, create it
    if not table_exists:
        writer = new_data_df.write.format('iceberg').option('write-format', 'parquet')

        # partition, if specified partition columns
        if partition_columns:
            writer = writer.partitionBy(*partition_columns)

        # add optional properties, if defined
        if comment:
            writer = writer.option('comment', comment)

        # create the table for the feature
        writer.mode('overwrite').saveAsTable(f'quantum_catalog.quantum_features.{table_name}')

        print(f'Created table: quantum_catalog.quantum_features.{table_name}')

        # create a tag for this version of the table
        snapshot_id = spark.sql(
            f'SELECT snapshot_id FROM quantum_catalog.quantum_features.{table_name}.snapshots ORDER BY committed_at DESC LIMIT 1'
        ).collect()[0][0]

        # Get processing_batch_id before we lose reference to the DataFrame
        first_row = new_data_df.limit(1).collect()
        processing_batch_id = first_row[0]['processing_batch_id'].replace('-', '')
        version_tag = f'v_{processing_batch_id}'

        spark.sql(f"""
        ALTER TABLE quantum_catalog.quantum_features.{table_name}
        CREATE TAG {version_tag} AS OF VERSION {snapshot_id}
        """)

        print(f'Created version tag: {version_tag} for table {table_name}')

        return version_tag, new_data_df.count()

    # if table does exist
    # Store the processing_batch_id before filtering for new records
    first_row = new_data_df.limit(1).collect()
    if not first_row:
        print(f'Input dataset is empty for {table_name}')
        return None, 0

    processing_batch_id = first_row[0]['processing_batch_id'].replace('-', '')

    truly_new_data = identify_new_records(spark, new_data_df, table_name, key_columns)

    # if no new data, do not move with the process
    new_record_count = truly_new_data.count()
    if new_record_count == 0:
        print(f'No new records found for table {table_name}')
        return None, 0

    # write only new data to the Iceberg
    writer = truly_new_data.write.format('iceberg').option('write-format', 'parquet')

    # add partitioning if specified
    if partition_columns:
        writer = writer.partitionBy(*partition_columns)

    # append to the existing table
    writer.mode('append').saveAsTable(f'quantum_catalog.quantum_features.{table_name}')

    print(f'Appended {new_record_count} new records to table {table_name}')

    # create a tag for the incremental update
    snapshot_id = spark.sql(
        f'SELECT snapshot_id FROM quantum_catalog.quantum_features.{table_name}.snapshots ORDER BY committed_at DESC LIMIT 1'
    ).collect()[0][0]

    version_tag = f'v_incr_{processing_batch_id}'

    spark.sql(f"""
        ALTER TABLE quantum_catalog.quantum_features.{table_name}
        CREATE TAG {version_tag} AS OF VERSION {snapshot_id}
    """)

    print(f'Created incremental version tag: {version_tag} for table {table_name}')

    return version_tag, new_record_count


def transform_quantum_data(df):
    """
    Transforms the original quantum data into various feature tables.

    Args:
        df: Original dataframe with quantum simulation data

    Returns:
        dict: Dictionary of transformed dataframes
    """
    # original dataframe with metadata attached
    base_df = add_metadata_columns(df, 'quantum_base_processing')

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
        col('hamiltonian_time'),
        col('mapping_time'),
        col('vqe_time'),
        col('total_time'),
        col('molecule.molecule_data').alias('molecule_data'),
        col('processing_timestamp'),
        col('processing_date'),
        col('processing_batch_id'),
        col('processing_name'),
    )

    # molecule information dataframe
    df_molecule = base_df.select(
        col('experiment_id'),
        col('molecule_id'),
        col('molecule_data.symbols').alias('atom_symbols'),
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

    # ansatz information dataframe
    df_ansatz = base_df.select(
        col('experiment_id'),
        col('molecule_id'),
        col('basis_set'),
        col('initial_data.ansatz').alias('ansatz'),
        col('initial_data.ansatz_reps').alias('ansatz_reps'),
        col('processing_timestamp'),
        col('processing_date'),
        col('processing_batch_id'),
        col('processing_name'),
    )

    # metrics information dataframe
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
        col('processing_timestamp'),
        col('processing_date'),
        col('processing_batch_id'),
        col('processing_name'),
    )

    # VQE results dataframe
    df_vqe = base_df.select(
        col('experiment_id'),
        col('molecule_id'),
        col('basis_set'),
        col('initial_data.backend').alias('backend'),
        col('initial_data.num_qubits').alias('num_qubits'),
        col('initial_data.optimizer').alias('optimizer'),
        col('initial_data.noise_backend').alias('noise_backend'),
        col('initial_data.default_shots').alias('default_shots'),
        col('initial_data.ansatz_reps').alias('ansatz_reps'),
        col('minimum_energy'),
        col('maxcv'),
        size(col('iteration_list')).alias('total_iterations'),
        col('processing_timestamp'),
        col('processing_date'),
        col('processing_batch_id'),
        col('processing_name'),
    )

    # initial parameters dataframe
    df_initial_parameters = (
        base_df.select(
            col('experiment_id'),
            col('molecule_id'),
            col('basis_set'),
            col('initial_data.backend').alias('backend'),
            col('initial_data.num_qubits').alias('num_qubits'),
            explode(col('initial_data.initial_parameters')).alias('initial_parameter_value'),
            col('processing_timestamp'),
            col('processing_date'),
            col('processing_batch_id'),
            col('processing_name'),
        )
        .withColumn(
            'parameter_index',
            expr('hash(concat(experiment_id, initial_parameter_value)) % 1000000'),
        )
        .withColumn(
            'parameter_id',
            expr("concat(experiment_id, '_init_', cast(parameter_index as string))"),
        )
    )

    # optimal parameters
    df_optimal_parameters = (
        base_df.select(
            col('experiment_id'),
            col('molecule_id'),
            col('basis_set'),
            col('initial_data.backend').alias('backend'),
            col('initial_data.num_qubits').alias('num_qubits'),
            explode(col('optimal_parameters')).alias('optimal_parameter_value'),
            col('processing_timestamp'),
            col('processing_date'),
            col('processing_batch_id'),
            col('processing_name'),
        )
        .withColumn(
            'parameter_index',
            expr('hash(concat(experiment_id, optimal_parameter_value)) % 1000000'),
        )
        .withColumn(
            'parameter_id', expr("concat(experiment_id, '_opt_', cast(parameter_index as string))")
        )
    )

    # iterations
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
            col('processing_timestamp'),
            col('processing_date'),
            col('processing_batch_id'),
            col('processing_name'),
        )
        .withColumn(
            'iteration_id',
            expr(
                "concat(experiment_id, '_iter_', cast(hash(concat(experiment_id, cast(iteration_step as string))) % 1000000 as string))"
            ),
        )
    )

    # iteration parameters
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
            explode(col('iteration.parameters')).alias('parameter_value'),
            col('processing_timestamp'),
            col('processing_date'),
            col('processing_batch_id'),
            col('processing_name'),
        )
        .withColumn(
            'parameter_index',
            expr(
                'hash(concat(experiment_id, cast(iteration_step as string), parameter_value)) % 1000000'
            ),
        )
        .withColumn(
            'iteration_id',
            expr(
                "concat(experiment_id, '_iter_', cast(hash(concat(experiment_id, cast(iteration_step as string))) % 1000000 as string))"
            ),
        )
        .withColumn(
            'parameter_id',
            expr("concat(iteration_id, '_param_', cast(parameter_index as string))"),
        )
    )

    # hamiltonian terms
    df_hamiltonian = (
        base_df.select(
            col('experiment_id'),
            col('molecule_id'),
            col('basis_set'),
            col('initial_data.backend').alias('backend'),
            explode(col('initial_data.hamiltonian')).alias('hamiltonian_term'),
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
            col('hamiltonian_term.label').alias('term_label'),
            col('hamiltonian_term.coefficients.real').alias('coeff_real'),
            col('hamiltonian_term.coefficients.imaginary').alias('coeff_imag'),
            col('processing_timestamp'),
            col('processing_date'),
            col('processing_batch_id'),
            col('processing_name'),
        )
        .withColumn('term_index', expr('hash(concat(experiment_id, term_label)) % 1000000'))
        .withColumn('term_id', expr("concat(experiment_id, '_term_', cast(term_index as string))"))
    )

    # return all the transformed dataframes
    return {
        'molecules': df_molecule,
        'ansatz_info': df_ansatz,
        'performance_metrics': df_metrics,
        'vqe_results': df_vqe,
        'initial_parameters': df_initial_parameters,
        'optimal_parameters': df_optimal_parameters,
        'vqe_iterations': df_iterations,
        'iteration_parameters': df_iteration_parameters,
        'hamiltonian_terms': df_hamiltonian,
        'base_df': base_df,
    }


def create_metadata_table_if_not_exists(spark):
    """Create the metadata tracking table if it doesn't exist."""
    spark.sql("""
    CREATE TABLE IF NOT EXISTS quantum_catalog.quantum_features.processing_metadata (
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


def update_metadata_table(spark, dfs, table_names, table_versions, record_counts, source_info):
    """Update the metadata table with processing information."""
    base_df = dfs['base_df']

    processing_info = spark.createDataFrame(
        [
            {
                'processing_batch_id': base_df.first().processing_batch_id,
                'processing_name': base_df.first().processing_name,
                'processing_timestamp': base_df.first().processing_timestamp,
                'processing_date': base_df.first().processing_date,
                'table_names': table_names,
                'table_versions': table_versions,
                'record_counts': record_counts,
                'source_data_info': source_info,
            }
        ]
    )

    processing_info.write.format('iceberg').mode('append').saveAsTable(
        'quantum_catalog.quantum_features.processing_metadata'
    )

    print(f'Updated metadata table with processing batch {base_df.first().processing_batch_id}')


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
    """
    Main function to process quantum data incrementally.

    Args:
        spark: SparkSession
        df: Original dataframe with quantum simulation data
        topic_name: Optional name of the topic being processed

    Returns:
        dict: Summary of processed record counts
    """
    # create metadata tracking table if it doesn't exist
    create_metadata_table_if_not_exists(spark)

    # transform the data
    dfs = transform_quantum_data(df)

    # get table configurations
    table_configs = get_table_configs()

    # process each table incrementally
    table_versions = []
    record_counts = []
    table_names = []

    for table_name, config in table_configs.items():
        print(f'Processing table: {table_name}')
        version_tag, count = process_incremental_data(
            spark,
            dfs[table_name],
            table_name,
            config['key_columns'],
            config['partition_columns'],
            config['comment'],
        )

        table_names.append(table_name)
        table_versions.append(version_tag if version_tag else 'no_changes')
        record_counts.append(count)

    # update metadata tracking
    source_info = f'Incremental VQE simulation data processing' + (
        f' from topic {topic_name}' if topic_name else ''
    )
    update_metadata_table(spark, dfs, table_names, table_versions, record_counts, source_info)

    print('Incremental processing completed!')

    # release cached dataframes
    for df_name, dataframe in dfs.items():
        if df_name != 'base_df':  # Keep base_df for metadata
            dataframe.unpersist()

    # summary of processed records
    return dict(zip(table_names, record_counts))


def check_for_new_data(spark, topic, config=None):
    """
    Check for new data in the S3 bucket.

    Args:
        spark: SparkSession
        config: Optional configuration dictionary

    Returns:
        tuple: (topic_name, dataframe) or (None, None) if no new data
    """
    if config is None:
        config = DEFAULT_CONFIG

    bucket_path = config.get('S3_BUCKET', DEFAULT_CONFIG['S3_BUCKET'])

    print(f'Processing topic: {topic}')

    # read data from the topic
    df = read_experiments_by_topic(spark, bucket_path, topic, num_partitions=4)

    # check if there's data to process
    if df.isEmpty():
        print(f'No data available in topic {topic}')
        return None, None

    return topic, df


def main(config=None):
    """
    Main entry point for the quantum feature processing script.

    Args:
        config: Optional configuration dictionary
    """
    # create spark session
    spark = create_spark_session(config)

    try:
        if config is None:
            config = DEFAULT_CONFIG

        bucket_path = config.get('S3_BUCKET', DEFAULT_CONFIG['S3_BUCKET'])
        available_topics = list_available_topics(spark, bucket_path)

        for topic in available_topics:
            print(f'Found topic: {topic}')

            # check for new data
            topic_name, df = check_for_new_data(spark, topic, config)

            if df is None:
                print('No new data to process.')
                return

            # Process data incrementally
            results = process_experiments_incrementally(spark, df, topic_name)

            print(f'\nProcessing Summary for topic {topic}:')
            for table, count in results.items():
                print(f'{table}: {count} new records processed')
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
