"""
Airflow DAG: VQE batch generation wrapper (vqe_batch_generation_dag.py).

Wraps scripts/generate_ml_batch.py as a single BashOperator task so that
Airflow can provide scheduling, alerting, and ExternalTaskSensor chaining
to the downstream Spark processing DAG — without the LocalExecutor
scalability problems of orchestrating 3,200 individual VQE tasks.

Architecture: Option C1 (hybrid) from QUA-36 research.
  - Script handles all generation logic (3-lane parallel, JSON state, resume)
  - Airflow handles: scheduling, email-on-failure, ExternalTaskSensor trigger

Trigger: Manual only (schedule=None). Run via Airflow UI "Trigger DAG"
         with optional JSON conf: {"tier": 1}
"""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

# The batch script path inside the Airflow container.
# docker-compose.ml.yaml mounts ../docker/airflow/ → /opt/airflow/dags,
# but the script lives in the repo root's scripts/ directory.
# Adjust REPO_ROOT below if the Airflow container mounts it differently.
_REPO_ROOT = os.environ.get("QUANTUM_PIPELINE_ROOT", "/home/zweiss/code/quantum-pipeline")

default_args = {
    "owner": "quantum_pipeline",
    "depends_on_past": False,
    "email": ["quantum_alerts@example.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=30),
}

with DAG(
    "vqe_batch_generation",
    default_args=default_args,
    description=(
        "Run scripts/generate_ml_batch.py for one tier. "
        "Resumes automatically from last completed invocation. "
        "Pass {'tier': N} in the trigger conf to select tier (default: 1)."
    ),
    schedule=None,                              # manual trigger only
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["quantum", "batch", "generation", "VQE"],
) as dag:

    run_batch_generation = BashOperator(
        task_id="run_batch_generation",
        bash_command=(
            "cd {{ params.repo_root }} && "
            "python scripts/generate_ml_batch.py "
            "--tier {{ dag_run.conf.get('tier', 1) }} "
            "--log-level INFO"
        ),
        params={"repo_root": _REPO_ROOT},
        # Safety cap: Tier 1 is ~15-25h; allow 30h before Airflow kills the task.
        # The script's JSON state ensures generation resumes on next trigger.
        execution_timeout=timedelta(hours=30),
        # Surface the script's stdout in Airflow task logs
        append_env=True,
    )
