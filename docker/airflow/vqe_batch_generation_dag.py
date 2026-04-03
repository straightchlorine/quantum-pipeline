"""
Airflow DAG: VQE batch generation wrapper (vqe_batch_generation_dag.py).

Wraps scripts/generate_ml_batch.py as a single BashOperator task so that
Airflow can provide scheduling, alerting, and ExternalTaskSensor chaining
to the downstream Spark processing DAG - without the LocalExecutor
scalability problems of orchestrating 3,200 individual VQE tasks.

Architecture: Option C1 (hybrid) from QUA-36 research.
  - Script handles all generation logic (3-lane parallel, JSON state, resume)
  - Airflow handles: scheduling, email-on-failure, ExternalTaskSensor trigger

Tasks:
  1. build_images         - builds quantum-pipeline:cpu and :gpu via Docker socket
  2. run_batch_generation - runs the batch script (3 parallel hardware lanes)

Docker images are built using the host's Docker daemon via the mounted
socket (/var/run/docker.sock). The .dockerignore whitelist keeps the
build context minimal. Docker layer cache makes rebuilds fast (~1-2s)
when nothing changed.

Trigger: Manual only (schedule=None). Run via Airflow UI "Trigger DAG"
         with optional JSON conf: {"tier": 1}

Requires:
  - Docker socket mounted into the Airflow worker (/var/run/docker.sock)
  - Repo root mounted at /opt/quantum-pipeline
  - Airflow user in a group matching the host's docker socket GID
"""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from common.dag_defaults import make_default_args

_REPO_ROOT = os.environ.get('QUANTUM_PIPELINE_ROOT', '/opt/quantum-pipeline')

default_args = make_default_args(
    email_on_failure=True,
    retry_delay=timedelta(minutes=30),
)

with DAG(
    'vqe_batch_generation',
    default_args=default_args,
    description=(
        'Build simulation images, then run scripts/generate_ml_batch.py for one tier. '
        'Resumes automatically from last completed invocation. '
        "Pass {'tier': N} in the trigger conf to select tier (default: 1)."
    ),
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['quantum', 'batch', 'generation', 'VQE'],
) as dag:
    # CUDA_ARCH: 6.1=Pascal(GTX10xx), 7.5=Turing(RTX20xx), 8.6=Ampere(RTX30xx), 8.9=Ada(RTX40xx)
    _CUDA_ARCH = os.environ.get('CUDA_ARCH', '6.1')

    build_images = BashOperator(
        task_id='build_images',
        bash_command=(
            'cd {{ params.repo_root }} && '
            'echo "Building quantum-pipeline:cpu ..." && '
            'docker build -f docker/Dockerfile.cpu -t quantum-pipeline:cpu . && '
            'echo "Building quantum-pipeline:gpu (CUDA_ARCH={{ params.cuda_arch }}) ..." && '
            'docker build -f docker/Dockerfile.gpu '
            '--build-arg CUDA_ARCH={{ params.cuda_arch }} '
            '-t quantum-pipeline:gpu .'
        ),
        params={'repo_root': _REPO_ROOT, 'cuda_arch': _CUDA_ARCH},
        execution_timeout=timedelta(hours=2),
        append_env=True,
    )

    run_batch_generation = BashOperator(
        task_id='run_batch_generation',
        bash_command=(
            'cd {{ params.repo_root }} && '
            'python scripts/generate_ml_batch.py '
            "--tier {{ dag_run.conf.get('tier', 1) }} "
            '--log-level INFO'
        ),
        params={'repo_root': _REPO_ROOT},
        execution_timeout=timedelta(hours=30),
        append_env=True,
    )

    build_images >> run_batch_generation
