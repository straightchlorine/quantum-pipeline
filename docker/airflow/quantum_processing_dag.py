"""
Airflow DAG for Quantum Feature Processing Pipeline

This DAG handles:
1. Scheduled processing of quantum experiment data
2. Incremental loading into Iceberg tables
3. Status monitoring and error notification
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from airflow.models import Variable
from airflow.utils.email import send_email

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


# get configuration from Airflow variables
def get_config():
    """Get configuration from Airflow variables with fallbacks"""
    config = {
        'SPARK_MASTER': Variable.get('SPARK_MASTER', 'spark://server:7077'),
        'S3_ENDPOINT': Variable.get('S3_ENDPOINT', 'http://server:9000'),
        'HOST_IP': Variable.get('HOST_IP', 'station'),
        'S3_BUCKET': Variable.get('S3_BUCKET', 's3a://local-vqe-results/experiments/'),
        'S3_WAREHOUSE': Variable.get('S3_WAREHOUSE', 's3a://local-features/warehouse/'),
        'SPARK_SUBMIT_ARGS': Variable.get('SPARK_SUBMIT_ARGS', ''),
        'PROCESSING_SCRIPT': Variable.get(
            'PROCESSING_SCRIPT', '/opt/airflow/dags/scripts/quantum_feature_processing.py'
        ),
        'DATA_PATH': Variable.get('DATA_PATH', '/opt/airflow/data/quantum'),
    }
    return config


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
    # get configuration
    config = get_config()

    check_for_data = FileSensor(
        task_id='check_for_data',
        filepath=config['DATA_PATH'],
        poke_interval=60,
        timeout=300,
        mode='reschedule',
    )

    # submit spark job
    run_quantum_processing = SparkSubmitOperator(
        task_id='run_quantum_processing',
        application=config['PROCESSING_SCRIPT'],
        conn_id='spark_default',
        name='quantum_feature_processing',
        conf={
            'spark.master': config['SPARK_MASTER'],
            'spark.driver.memory': '4g',
            'spark.executor.memory': '4g',
            'spark.executor.cores': '2',
            'spark.driver.host': config['HOST_IP'],
        },
        env_vars={
            'MINIO_ACCESS_KEY': '{{ var.value.MINIO_ACCESS_KEY }}',
            'MINIO_SECRET_KEY': '{{ var.value.MINIO_SECRET_KEY }}',
            'S3_ENDPOINT': config['S3_ENDPOINT'],
            'S3_BUCKET': config['S3_BUCKET'],
            'S3_WAREHOUSE': config['S3_WAREHOUSE'],
        },
        application_args=[
            '--s3_endpoint',
            config['S3_ENDPOINT'],
            '--s3_bucket',
            config['S3_BUCKET'],
            '--s3_warehouse',
            config['S3_WAREHOUSE'],
            '--spark_master',
            config['SPARK_MASTER'],
            '--host_ip',
            config['HOST_IP'],
        ],
        verbose=True,
    )

    # set task dependencies
    check_for_data >> run_quantum_processing
