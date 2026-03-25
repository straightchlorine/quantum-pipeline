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

  Inject via docker-compose.ml.yaml env_file, or override at runtime with
  Airflow Variables (R2_SYNC_GARAGE_WAREHOUSE, R2_SYNC_R2_BUCKET, etc.).

Schedule: manual trigger by default (set R2_SYNC_SCHEDULE Variable to override,
e.g. "@weekly"). Depends on quantum_ml_feature_processing completing first.
"""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.sensors.external_task import ExternalTaskSensor
from airflow.utils.log.logging_mixin import LoggingMixin

logger = LoggingMixin().log

# ------------------------------------------------------------------
# Schedule: None (manual) unless overridden via Airflow Variable.
# This is the only Variable.get() at module scope — acceptable for
# schedule since it's a single cheap DB call.
# ------------------------------------------------------------------
_schedule = Variable.get('R2_SYNC_SCHEDULE', default_var=None)
if _schedule == 'None' or _schedule == '':
    _schedule = None

# rclone env keys — read from container environment (cheap, no DB hit).
# Airflow Variable overrides are resolved at task runtime only.
_RCLONE_ENV_KEYS = [
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

default_args = {
    'owner': 'quantum_pipeline',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
}


def _resolve_rclone_env():
    """Build rclone env dict at task runtime (not parse time).

    Checks Airflow Variables first, falls back to container env vars.
    """
    return {k: Variable.get(k, default_var=os.getenv(k, '')) for k in _RCLONE_ENV_KEYS}


def _resolve_sync_config():
    """Resolve sync paths and tuning params at task runtime."""
    return {
        'garage_warehouse': Variable.get(
            'R2_SYNC_GARAGE_WAREHOUSE',
            default_var='garage:features/warehouse/quantum_features',
        ),
        'r2_bucket': Variable.get(
            'R2_SYNC_R2_BUCKET',
            default_var='r2:qp-data/features',
        ),
        'transfers': Variable.get('R2_SYNC_TRANSFERS', default_var='8'),
        'checkers': Variable.get('R2_SYNC_CHECKERS', default_var='4'),
    }


def _health_check_fn(**context):
    """Verify rclone can reach both Garage and R2 before attempting sync."""
    import subprocess  # noqa: PLC0415

    env = _resolve_rclone_env()
    cfg = _resolve_sync_config()

    def _check(remote_name, path):
        result = subprocess.run(  # noqa: S603
            ['rclone', 'lsd', path],
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, **env},
        )
        if result.returncode != 0:
            raise RuntimeError(
                f'rclone connectivity check failed for {remote_name}:\n'
                f'stdout: {result.stdout}\nstderr: {result.stderr}'
            )
        logger.info(f'rclone connectivity OK for {remote_name}')

    _check('Garage (features)', f'{cfg["garage_warehouse"]}/')
    _check('R2 (qp-data)', f'{cfg["r2_bucket"]}/')


def _sync_table_fn(table_name, **context):
    """Run rclone sync for a single table at task runtime."""
    import subprocess  # noqa: PLC0415

    env = _resolve_rclone_env()
    cfg = _resolve_sync_config()

    cmd = (
        f'rclone sync '
        f'{cfg["garage_warehouse"]}/{table_name}/ '
        f'{cfg["r2_bucket"]}/{table_name}/ '
        f'--transfers {cfg["transfers"]} '
        f'--checkers {cfg["checkers"]} '
        f'--stats 30s '
        f'--stats-one-line '
        f'--log-level INFO'
    )
    logger.info(f'Running: {cmd}')
    result = subprocess.run(  # noqa: S603
        cmd.split(),
        capture_output=True,
        text=True,
        env={**os.environ, **env},
    )
    if result.stdout:
        logger.info(result.stdout)
    if result.returncode != 0:
        logger.error(result.stderr)
        raise RuntimeError(f'rclone sync failed for {table_name}: {result.stderr}')


with DAG(
    'r2_sync',
    default_args=default_args,
    description='Sync ML feature Parquet files from Garage to Cloudflare R2 via rclone',
    schedule=_schedule,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['quantum', 'ML', 'r2', 'sync', 'rclone'],
) as dag:

    wait_for_ml_features = ExternalTaskSensor(
        task_id='wait_for_ml_feature_processing',
        external_dag_id='quantum_ml_feature_processing',
        external_task_id='run_ml_feature_processing',
        execution_delta=timedelta(0),
        mode='reschedule',
        poke_interval=60,
        timeout=7200,
    )

    health_check = PythonOperator(
        task_id='rclone_health_check',
        python_callable=_health_check_fn,
    )

    sync_iteration_features = PythonOperator(
        task_id='sync_ml_iteration_features',
        python_callable=_sync_table_fn,
        op_kwargs={'table_name': 'ml_iteration_features'},
    )

    sync_run_summary = PythonOperator(
        task_id='sync_ml_run_summary',
        python_callable=_sync_table_fn,
        op_kwargs={'table_name': 'ml_run_summary'},
    )

    wait_for_ml_features >> health_check >> [sync_iteration_features, sync_run_summary]
