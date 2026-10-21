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
from airflow.sdk import Asset
from airflow.utils.log.logging_mixin import LoggingMixin
from common.dag_defaults import make_default_args
from common.pipeline_config import S3_BUCKET_URL, S3_WAREHOUSE_URL

logger = LoggingMixin().log

# emitted when a processing run finishes - Iceberg warehouse run writes to
# TODO: move it to a shared var; keep in sync with quantum_ml_feature_dag.py
QUANTUM_FEATURES_ASSET = Asset('s3://features/warehouse/quantum_features')


# create the DAG
with DAG(
    'quantum_feature_processing',
    default_args=make_default_args(retries=3, retry_delay=timedelta(minutes=20)),
    description='Process quantum experiment data into ML feature tables',
    schedule='@daily',
    start_date=datetime(2025, 1, 1),
    catchup=False,
    # Spark resolves jars into a shared ivy cache; run this one at a time.
    # NOTE: two concurrent runs of this DAG race on that cache
    # one of them fails with 'unresolved dependency'.
    max_active_runs=1,
    tags=['quantum', 'ML', 'features', 'processing', 'Apache Spark'],
) as dag:
    quantum_simulation_results_processing = SparkSubmitOperator(
        task_id='run_quantum_processing',
        application='/opt/airflow/dags/scripts/quantum_incremental_processing.py',
        conn_id='spark_default',
        name='quantum_feature_processing',
        conf={
            'spark.jars.ivy': '/tmp/.ivy2',  # noqa: S108
        },
        env_vars={
            'S3_BUCKET_URL': S3_BUCKET_URL,
            'S3_WAREHOUSE_URL': S3_WAREHOUSE_URL,
        },
        execution_timeout=timedelta(hours=2),
        # TODO: delete; Airflow 3 removed SLAs and only warns on sla= (Deadline Alerts replace it in 3.1+)
        sla=timedelta(hours=1, minutes=30),
        # asset updated only on success - schedules quantum_ml_feature_processing
        outlets=[QUANTUM_FEATURES_ASSET],
        verbose=True,
    )
