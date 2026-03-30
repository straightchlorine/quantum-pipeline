"""
Airflow DAG: VQE batch generation wrapper (vqe_batch_generation_dag.py).

Wraps scripts/generate_ml_batch.py as a single BashOperator task so that
Airflow can provide scheduling, alerting, and ExternalTaskSensor chaining
to the downstream Spark processing DAG - without the LocalExecutor
scalability problems of orchestrating 3,200 individual VQE tasks.

Architecture: Option C1 (hybrid) from QUA-36 research.
  - Script handles all generation logic (3-lane parallel, JSON state, resume)
  - Airflow handles: scheduling, email-on-failure, ExternalTaskSensor trigger

Trigger: Manual only (schedule=None). Run via Airflow UI "Trigger DAG"
         with optional JSON conf: {"tier": 1}

Requires:
  - Docker socket mounted into the Airflow worker (/var/run/docker.sock)
  - Repo root mounted at /opt/quantum-pipeline (for scripts/, compose/, data/, gen/)
"""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from common.dag_defaults import make_default_args

# The repo root is mounted at this path inside the Airflow worker container.
# docker-compose.ml.yaml binds the host repo root here.
_REPO_ROOT = os.environ.get('QUANTUM_PIPELINE_ROOT', '/opt/quantum-pipeline')

default_args = make_default_args(
    email_on_failure=True,
    retry_delay=timedelta(minutes=30),
)

with DAG(
    'vqe_batch_generation',
    default_args=default_args,
    description=(
        'Run scripts/generate_ml_batch.py for one tier. '
        'Resumes automatically from last completed invocation. '
        "Pass {'tier': N} in the trigger conf to select tier (default: 1)."
    ),
    schedule=None,  # manual trigger only
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['quantum', 'batch', 'generation', 'VQE'],
) as dag:
    run_batch_generation = BashOperator(
        task_id='run_batch_generation',
        bash_command=(
            'cd {{ params.repo_root }} && '
            'python scripts/generate_ml_batch.py '
            "--tier {{ dag_run.conf.get('tier', 1) }} "
            '--log-level INFO'
        ),
        params={'repo_root': _REPO_ROOT},
        # Safety cap: Tier 1 is ~15-25h; allow 30h before Airflow kills the task.
        # The script's JSON state ensures generation resumes on next trigger.
        execution_timeout=timedelta(hours=30),
        # Surface the script's stdout in Airflow task logs
        append_env=True,
    )
