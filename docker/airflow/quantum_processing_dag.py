"""
Airflow DAG for Quantum Feature Processing Pipeline

This DAG handles:
1. Scheduled processing of quantum experiment data
2. Incremental loading into Iceberg tables
3. Status monitoring and error notification
"""

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import ShortCircuitOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.utils.email import send_email
from airflow.utils.log.logging_mixin import LoggingMixin

logger = LoggingMixin().log

try:
    from scripts.quantum_incremental_processing import (
        DEFAULT_CONFIG,
        check_for_new_data,
        create_spark_session,
        list_available_topics,
    )
except ImportError:
    logger.error(
        "Could not import the quantum processing module. Make sure it's in the PYTHONPATH."
    )
    sys.exit(1)

# default args for the DAG
default_args = {
    'owner': 'quantum_pipeline',
    'depends_on_past': False,
    'email': ['quantum_alerts@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def check_for_data_to_process(**kwargs):
    """
    Check for new data to process in quantum data topics

    Uses the check_for_new_data function from the quantum_incremental_processing module to
    check if there are new experiment results that need to be processed.
    """
    config = kwargs['config']
    logger.info(f'Creating Spark session with config: {config}')

    # create a Spark session
    spark = create_spark_session(config)

    logger.info(f'Spark session created - AppName: {spark.conf.get("spark.app.name")}')
    logger.info(f'Spark version: {spark.version}')
    logger.info(f'Spark master: {spark.conf.get("spark.master")}')
    logger.info(f'Spark driver host: {spark.conf.get("spark.driver.host")}')
    logger.info(
        f'Spark warehouse path: {spark.conf.get("spark.sql.catalog.quantum_catalog.warehouse")}'
    )
    logger.info(f'S3 endpoint: {spark.conf.get("spark.hadoop.fs.s3a.endpoint")}')

    logger.info(f'Driver memory: {spark.conf.get("spark.driver.memory", "default")}')
    logger.info(f'Executor memory: {spark.conf.get("spark.executor.memory", "default")}')
    logger.info(f'Executor cores: {spark.conf.get("spark.executor.cores", "default")}')
    logger.info(f'Executor instances: {spark.conf.get("spark.executor.instances", "default")}')
    logger.info(
        f'Dynamic allocation enabled: {spark.conf.get("spark.dynamicAllocation.enabled", "default")}'
    )

    try:
        bucket_path = config.get('S3_BUCKET', DEFAULT_CONFIG['S3_BUCKET'])
        available_topics = list_available_topics(spark, bucket_path)

        if not available_topics:
            logger.info('No topics found in the bucket')
            return False

        has_new_data = False
        topics_with_data = []

        # check each topic for new data
        for topic in available_topics:
            logger.info(f'Checking topic: {topic}')
            _, df = check_for_new_data(spark, topic, config)

            if df is not None and not df.isEmpty():
                has_new_data = True
                topics_with_data.append(topic)
                logger.info(f'Found new data for topic: {topic}')

        # store information about which topics have data
        kwargs['ti'].xcom_push(key='topics_with_data', value=topics_with_data)

        return has_new_data

    finally:
        spark.stop()


# function to send success email with processing results
def send_success_email(context):
    """Send a detailed success email with processing summary"""
    task_instance = context['task_instance']
    results = task_instance.xcom_pull(task_ids='run_quantum_processing')

    if not results:
        results = 'No data processed'

    subject = f'Quantum Processing Success: {context["execution_date"]}'
    html_content = f"""
    <h3>Quantum Processing Completed Successfully</h3>
    <p><b>Execution Date:</b> {context['execution_date']}</p>
    <p><b>Processing Results:</b></p>
    <pre>{results}</pre>
    <p>View the <a href="{context['task_instance'].log_url}">logs</a> for more details.</p>
    """

    send_email(to=default_args['email'], subject=subject, html_content=html_content)


# create the DAG
with DAG(
    'quantum_feature_processing',
    default_args=default_args,
    description='Process quantum experiment data into feature tables',
    schedule_interval=timedelta(hours=1),  # Run hourly
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['quantum', 'features', 'processing'],
    on_success_callback=send_success_email,
) as dag:
    # set default config values
    for k, v in DEFAULT_CONFIG.items():
        Variable.set(k, v)

    Variable.set('MINIO_ACCESS_KEY', os.getenv('MINIO_ACCESS_KEY'))
    Variable.set('MINIO_SECRET_KEY', os.getenv('MINIO_SECRET_KEY'))

    # check for new data using the imported functions
    check_for_data = ShortCircuitOperator(
        task_id='check_for_data',
        python_callable=check_for_data_to_process,
        op_kwargs={'config': DEFAULT_CONFIG},
        provide_context=True,
    )

    # submit Spark job - only run if check_for_data returns True
    run_quantum_processing = SparkSubmitOperator(
        task_id='run_quantum_processing',
        application='/opt/airflow/dags/scripts/quantum_incremental_processing.py',
        conn_id='spark_default',
        name='quantum_feature_processing',
        conf={
            'spark.master': Variable.get('SPARK_MASTER'),
            'spark.app.name': Variable.get('APP_NAME'),
            'spark.hadoop.fs.s3a.endpoint': Variable.get('S3_ENDPOINT'),
            'spark.hadoop.fs.s3a.access.key': Variable.get('MINIO_ACCESS_KEY'),
            'spark.hadoop.fs.s3a.secret.key': Variable.get('MINIO_SECRET_KEY'),
            'spark.hadoop.fs.s3a.path.style.access': 'true',
            'spark.hadoop.fs.s3a.connection.ssl.enabled': 'false',
            'spark.jars.packages': (
                'org.slf4j:slf4j-api:2.0.17,'
                'commons-codec:commons-codec:1.18.0,'
                'com.google.j2objc:j2objc-annotations:3.0.0,'
                'org.apache.spark:spark-avro_2.12:3.5.5,'
                'org.apache.hadoop:hadoop-aws:3.3.1,'
                'org.apache.hadoop:hadoop-common:3.3.1,'
                'org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.4.2,'
            ),
        },
        env_vars={
            'MINIO_ACCESS_KEY': Variable.get('MINIO_ACCESS_KEY'),
            'MINIO_SECRET_KEY': Variable.get('MINIO_SECRET_KEY'),
            'S3_ENDPOINT': Variable.get('S3_ENDPOINT'),
            'S3_BUCKET': Variable.get('S3_BUCKET'),
            'S3_WAREHOUSE': Variable.get('S3_WAREHOUSE'),
            'SPARK_MASTER': Variable.get('SPARK_MASTER'),
        },
        verbose=True,
    )

    # task dependencies
    check_for_data >> run_quantum_processing
