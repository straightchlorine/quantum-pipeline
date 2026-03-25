"""
Airflow DAG: Quantum ML Feature Materialization

Joins the 9 normalized Iceberg tables produced by quantum_feature_processing
into two ML-ready feature tables:

  - quantum_catalog.quantum_features.ml_iteration_features  (per-iteration)
  - quantum_catalog.quantum_features.ml_run_summary         (per-run aggregates)

Depends on: quantum_feature_processing DAG (waits via ExternalTaskSensor).
"""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.providers.standard.sensors.external_task import ExternalTaskSensor
from airflow.utils.log.logging_mixin import LoggingMixin

logger = LoggingMixin().log

default_args = {
    'owner': 'quantum_pipeline',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

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
        mode='reschedule',
        poke_interval=60,
        timeout=3600,
    )

    run_ml_feature_processing = SparkSubmitOperator(
        task_id='run_ml_feature_processing',
        application='/opt/airflow/dags/scripts/quantum_ml_feature_processing.py',
        conn_id='spark_default',
        name='quantum_ml_feature_processing',
        conf={
            'spark.jars.ivy': '/tmp/.ivy2',
        },
        env_vars={
            'S3_BUCKET_URL': os.getenv('S3_BUCKET_URL', 's3a://raw-results/experiments/'),
            'S3_WAREHOUSE_URL': os.getenv('S3_WAREHOUSE_URL', 's3a://features/warehouse/'),
        },
        execution_timeout=timedelta(hours=1),
        verbose=True,
    )

    wait_for_raw_processing >> run_ml_feature_processing
