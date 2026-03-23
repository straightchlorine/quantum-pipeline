"""
Airflow DAG: R2 Sync via rclone

Incrementally syncs processed ML feature tables from Garage (local S3) to
Cloudflare R2 using rclone:

  garage:features/warehouse/quantum_features/ml_iteration_features/
    → r2:qp-data/features/ml_iteration_features/

  garage:features/warehouse/quantum_features/ml_run_summary/
    → r2:qp-data/features/ml_run_summary/

Prerequisites:
  - rclone must be installed in the Airflow container image (Dockerfile.airflow).
  - Rclone remotes configured via environment variables (no rclone config file needed):

    Garage remote ("garage"):
      RCLONE_CONFIG_GARAGE_TYPE=s3
      RCLONE_CONFIG_GARAGE_PROVIDER=Other
      RCLONE_CONFIG_GARAGE_ENDPOINT=http://garage:3900
      RCLONE_CONFIG_GARAGE_ACCESS_KEY_ID=<garage-access-key>
      RCLONE_CONFIG_GARAGE_SECRET_ACCESS_KEY=<garage-secret-key>
      RCLONE_CONFIG_GARAGE_FORCE_PATH_STYLE=true
      RCLONE_CONFIG_GARAGE_NO_CHECK_BUCKET=true

    R2 remote ("r2"):
      RCLONE_CONFIG_R2_TYPE=s3
      RCLONE_CONFIG_R2_PROVIDER=Cloudflare
      RCLONE_CONFIG_R2_ENDPOINT=https://<account-id>.r2.cloudflarestorage.com
      RCLONE_CONFIG_R2_ACCESS_KEY_ID=<r2-access-key>
      RCLONE_CONFIG_R2_SECRET_ACCESS_KEY=<r2-secret-key>
      RCLONE_CONFIG_R2_NO_CHECK_BUCKET=true

  Set these as Airflow Variables or inject via docker-compose.ml.yaml env_file.

Schedule: manual trigger by default (set R2_SYNC_SCHEDULE Variable to override,
e.g. "@weekly"). Depends on quantum_ml_feature_processing completing first.
"""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.log.logging_mixin import LoggingMixin

logger = LoggingMixin().log

# ------------------------------------------------------------------
# Schedule: None (manual) unless overridden via Airflow Variable
# ------------------------------------------------------------------
_schedule = Variable.get('R2_SYNC_SCHEDULE', default_var=None)
if _schedule == 'None' or _schedule == '':
    _schedule = None

default_args = {
    'owner': 'quantum_pipeline',
    'depends_on_past': False,
    'email': ['quantum_alerts@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
}

# ------------------------------------------------------------------
# rclone environment variables (injected from Airflow Variables or
# the container environment — fall back to empty string so that
# missing credentials surface as a clear rclone error rather than a
# silent no-op).
# ------------------------------------------------------------------
def _rclone_env():
    """
    Build the rclone environment dict from Airflow Variables.
    Variables are populated from docker-compose.ml.yaml env_file (.env).
    """
    keys = [
        'RCLONE_CONFIG_GARAGE_TYPE',
        'RCLONE_CONFIG_GARAGE_PROVIDER',
        'RCLONE_CONFIG_GARAGE_ENDPOINT',
        'RCLONE_CONFIG_GARAGE_ACCESS_KEY_ID',
        'RCLONE_CONFIG_GARAGE_SECRET_ACCESS_KEY',
        'RCLONE_CONFIG_GARAGE_FORCE_PATH_STYLE',
        'RCLONE_CONFIG_GARAGE_NO_CHECK_BUCKET',
        'RCLONE_CONFIG_R2_TYPE',
        'RCLONE_CONFIG_R2_PROVIDER',
        'RCLONE_CONFIG_R2_ENDPOINT',
        'RCLONE_CONFIG_R2_ACCESS_KEY_ID',
        'RCLONE_CONFIG_R2_SECRET_ACCESS_KEY',
        'RCLONE_CONFIG_R2_NO_CHECK_BUCKET',
    ]
    return {k: Variable.get(k, default_var=os.getenv(k, '')) for k in keys}


_env = _rclone_env()

# ------------------------------------------------------------------
# Sync paths (configurable via Airflow Variables)
# ------------------------------------------------------------------
_garage_warehouse = Variable.get(
    'R2_SYNC_GARAGE_WAREHOUSE',
    default_var='garage:features/warehouse/quantum_features',
)
_r2_bucket = Variable.get(
    'R2_SYNC_R2_BUCKET',
    default_var='r2:qp-data/features',
)
_transfers = Variable.get('R2_SYNC_TRANSFERS', default_var='8')
_checkers = Variable.get('R2_SYNC_CHECKERS', default_var='4')


def _health_check_fn(**context):
    """
    Verify rclone can reach both Garage and R2 before attempting sync.
    Raises AirflowException on connectivity failure.
    """
    import subprocess  # noqa: PLC0415

    def _check(remote_name, path):
        result = subprocess.run(  # noqa: S603
            ['rclone', 'lsd', path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f'rclone connectivity check failed for {remote_name}:\n'
                f'stdout: {result.stdout}\nstderr: {result.stderr}'
            )
        logger.info(f'rclone connectivity OK for {remote_name}')

    _check('Garage (features)', f'{_garage_warehouse}/')
    _check('R2 (qp-data)', f'{_r2_bucket}/')


with DAG(
    'r2_sync',
    default_args=default_args,
    description='Sync ML feature Parquet files from Garage to Cloudflare R2 via rclone',
    schedule=_schedule,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['quantum', 'ML', 'r2', 'sync', 'rclone'],
) as dag:

    # wait for ML feature materialization to complete before syncing
    wait_for_ml_features = ExternalTaskSensor(
        task_id='wait_for_ml_feature_processing',
        external_dag_id='quantum_ml_feature_processing',
        external_task_id='run_ml_feature_processing',
        execution_delta=timedelta(0),
        mode='poke',
        poke_interval=60,
        timeout=7200,
    )

    # verify rclone connectivity to both remotes before committing to a sync
    health_check = PythonOperator(
        task_id='rclone_health_check',
        python_callable=_health_check_fn,
        env=_env,
    )

    sync_iteration_features = BashOperator(
        task_id='sync_ml_iteration_features',
        bash_command=(
            'rclone sync '
            f'{_garage_warehouse}/ml_iteration_features/ '
            f'{_r2_bucket}/ml_iteration_features/ '
            f'--transfers {_transfers} '
            f'--checkers {_checkers} '
            '--stats 30s '
            '--stats-one-line '
            '--log-level INFO'
        ),
        env=_env,
    )

    sync_run_summary = BashOperator(
        task_id='sync_ml_run_summary',
        bash_command=(
            'rclone sync '
            f'{_garage_warehouse}/ml_run_summary/ '
            f'{_r2_bucket}/ml_run_summary/ '
            f'--transfers {_transfers} '
            f'--checkers {_checkers} '
            '--stats 30s '
            '--stats-one-line '
            '--log-level INFO'
        ),
        env=_env,
    )

    wait_for_ml_features >> health_check >> [sync_iteration_features, sync_run_summary]
