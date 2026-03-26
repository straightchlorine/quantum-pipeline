"""
Airflow DAG: Quantum ML Feature Materialization

Joins the 9 normalized Iceberg tables produced by quantum_feature_processing
into two ML-ready feature tables:

  - quantum_catalog.quantum_features.ml_iteration_features  (per-iteration)
  - quantum_catalog.quantum_features.ml_run_summary         (per-run aggregates)

Depends on: quantum_feature_processing DAG (waits via ExternalTaskSensor).
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.providers.standard.sensors.external_task import ExternalTaskSensor
from airflow.utils.log.logging_mixin import LoggingMixin

from common.dag_defaults import make_default_args
from common.pipeline_config import S3_BUCKET_URL, S3_WAREHOUSE_URL

logger = LoggingMixin().log

with DAG(
    'quantum_ml_feature_processing',
    default_args=make_default_args(retries=2, retry_delay=timedelta(minutes=15)),
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
            'S3_BUCKET_URL': S3_BUCKET_URL,
            'S3_WAREHOUSE_URL': S3_WAREHOUSE_URL,
        },
        execution_timeout=timedelta(hours=1),
        sla=timedelta(minutes=45),
        verbose=True,
    )

    wait_for_raw_processing >> run_ml_feature_processing
