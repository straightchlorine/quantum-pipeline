"""
Airflow DAG: Quantum ML Feature Materialization

Joins the 9 normalized Iceberg tables produced by quantum_feature_processing
into two ML-ready feature tables:

  - quantum_catalog.quantum_features.ml_iteration_features  (per-iteration)
  - quantum_catalog.quantum_features.ml_run_summary         (per-run aggregates)

Depends on: quantum_feature_processing DAG (waits via ExternalTaskSensor).
"""

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.log.logging_mixin import LoggingMixin

logger = LoggingMixin().log

try:
    from scripts.quantum_ml_feature_processing import DEFAULT_CONFIG
except ImportError:
    logger.error(
        'Could not import the quantum ML feature processing module. '
        'Make sure it is in the PYTHONPATH.'
    )
    sys.exit(1)

default_args = {
    'owner': 'quantum_pipeline',
    'depends_on_past': False,
    'email': ['quantum_alerts@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# seed Airflow Variables from DEFAULT_CONFIG on first load
for k, v in DEFAULT_CONFIG.items():
    if Variable.get(k, default_var=None) is None:
        Variable.set(k, v)

Variable.set('S3_ACCESS_KEY', os.getenv('S3_ACCESS_KEY'))
Variable.set('S3_SECRET_KEY', os.getenv('S3_SECRET_KEY'))

with DAG(
    'quantum_ml_feature_processing',
    default_args={**default_args, 'retries': 2, 'retry_delay': timedelta(minutes=15)},
    description='Materialize ML feature tables from normalized VQE Iceberg tables',
    schedule=timedelta(days=1),
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['quantum', 'ML', 'features', 'iceberg', 'Apache Spark'],
) as dag:

    # wait for the upstream raw-processing DAG to finish for the same logical date
    wait_for_raw_processing = ExternalTaskSensor(
        task_id='wait_for_quantum_feature_processing',
        external_dag_id='quantum_feature_processing',
        external_task_id='run_quantum_processing',
        execution_delta=timedelta(0),
        mode='poke',
        poke_interval=60,
        timeout=3600,
    )

    run_ml_feature_processing = SparkSubmitOperator(
        task_id='run_ml_feature_processing',
        application='/opt/airflow/dags/scripts/quantum_ml_feature_processing.py',
        conn_id='spark_default',
        name='quantum_ml_feature_processing',
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

    wait_for_raw_processing >> run_ml_feature_processing
