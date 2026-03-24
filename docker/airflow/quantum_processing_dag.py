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
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.utils.email import send_email
from airflow.utils.log.logging_mixin import LoggingMixin

logger = LoggingMixin().log

try:
    from scripts.quantum_incremental_processing import DEFAULT_CONFIG
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


for k, v in DEFAULT_CONFIG.items():
    if Variable.get(k, default_var=None) is None:
        Variable.set(k, v)

Variable.set('S3_ACCESS_KEY', os.getenv('S3_ACCESS_KEY'))
Variable.set('S3_SECRET_KEY', os.getenv('S3_SECRET_KEY'))

# create the DAG
with DAG(
    'quantum_feature_processing',
    default_args={**default_args, 'retries': 3, 'retry_delay': timedelta(minutes=20)},
    description='Process quantum experiment data into ML feature tables',
    schedule=timedelta(days=1),
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['quantum', 'ML', 'features', 'processing', 'Apache Spark'],
    on_success_callback=send_success_email,
) as dag:
    # submit Spark job - only run if check_for_data returns True
    quantum_simulation_results_processing = SparkSubmitOperator(
        task_id='run_quantum_processing',
        application='/opt/airflow/dags/scripts/quantum_incremental_processing.py',
        conn_id='spark_default',
        name='quantum_feature_processing',
        conf={
            'spark.app.name': Variable.get('APP_NAME'),
        },
        env_vars={
            'S3_ACCESS_KEY': Variable.get('S3_ACCESS_KEY'),
            'S3_SECRET_KEY': Variable.get('S3_SECRET_KEY'),
            'S3_ENDPOINT': Variable.get('S3_ENDPOINT'),
            'S3_BUCKET': Variable.get('S3_BUCKET'),
            'S3_WAREHOUSE': Variable.get('S3_WAREHOUSE'),
        },
        verbose=True,
    )
