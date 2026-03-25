"""
Airflow DAG for Quantum Feature Processing Pipeline

This DAG handles:
1. Scheduled processing of quantum experiment data
2. Incremental loading into Iceberg tables
3. Status monitoring and error notification
"""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.utils.email import send_email
from airflow.utils.log.logging_mixin import LoggingMixin

logger = LoggingMixin().log

# default args for the DAG
default_args = {
    'owner': 'quantum_pipeline',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


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
    default_args={**default_args, 'retries': 3, 'retry_delay': timedelta(minutes=20)},
    description='Process quantum experiment data into ML feature tables',
    schedule=timedelta(days=1),
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['quantum', 'ML', 'features', 'processing', 'Apache Spark'],
) as dag:
    quantum_simulation_results_processing = SparkSubmitOperator(
        task_id='run_quantum_processing',
        application='/opt/airflow/dags/scripts/quantum_incremental_processing.py',
        conn_id='spark_default',
        name='quantum_feature_processing',
        conf={
            'spark.jars.ivy': '/tmp/.ivy2',
        },
        env_vars={
            'S3_BUCKET_URL': os.getenv('S3_BUCKET_URL', 's3a://raw-results/experiments/'),
            'S3_WAREHOUSE_URL': os.getenv('S3_WAREHOUSE_URL', 's3a://features/warehouse/'),
        },
        execution_timeout=timedelta(hours=2),
        on_success_callback=send_success_email,
        verbose=True,
    )
