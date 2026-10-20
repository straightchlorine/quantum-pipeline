"""
Airflow DAG: Quantum ML Feature Tables

Joins the 5 normalized Iceberg tables produced by quantum_feature_processing
into two ML-ready feature tables:

  - quantum_catalog.quantum_features.ml_iteration_features  (per-iteration)
  - quantum_catalog.quantum_features.ml_run_summary         (per-run aggregates)

Depends on: quantum_feature_processing DAG, via the QUANTUM_FEATURES_ASSET it emits.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.sdk import Asset
from airflow.utils.log.logging_mixin import LoggingMixin
from common.dag_defaults import make_default_args
from common.pipeline_config import S3_BUCKET_URL, S3_WAREHOUSE_URL

logger = LoggingMixin().log

# emitted by SparkSubmitOperator on success.
# TODO: move it to a shared var; until then keep in sync with quantum_processing_dag.py
QUANTUM_FEATURES_ASSET = Asset('s3://features/warehouse/quantum_features')

# emitted when the ML feature tables are created - r2 schedules on this
# TODO: move it to a shared var; until then keep in sync with r2_sync_dag.py.
ML_FEATURES_ASSET = Asset('s3://features/warehouse/quantum_features/ml_features')

with DAG(
    'quantum_ml_feature_processing',
    default_args=make_default_args(retries=2, retry_delay=timedelta(minutes=15)),
    description='Build the ML feature tables from the normalized VQE Iceberg tables',
    # runs when feature_processing's run_quantum_processing succeeds and emits
    # QUANTUM_FEATURES_ASSET.
    schedule=[QUANTUM_FEATURES_ASSET],
    start_date=datetime(2025, 1, 1),
    catchup=False,
    # shared ivy cache as the upstream DAG
    max_active_runs=1,
    tags=['quantum', 'ML', 'features', 'iceberg', 'Apache Spark'],
) as dag:
    run_ml_feature_processing = SparkSubmitOperator(
        task_id='run_ml_feature_processing',
        application='/opt/airflow/dags/scripts/quantum_ml_feature_processing.py',
        conn_id='spark_default',
        name='quantum_ml_feature_processing',
        conf={
            'spark.jars.ivy': '/tmp/.ivy2',  # noqa: S108
        },
        env_vars={
            'S3_BUCKET_URL': S3_BUCKET_URL,
            'S3_WAREHOUSE_URL': S3_WAREHOUSE_URL,
        },
        execution_timeout=timedelta(hours=1),
        sla=timedelta(minutes=45),
        # asset updated only on success - schedules r2_sync
        outlets=[ML_FEATURES_ASSET],
        verbose=True,
    )
