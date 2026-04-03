"""Shared Airflow DAG default arguments.

Provides a factory to build default_args dicts with consistent baseline
settings across all DAGs in the quantum pipeline.
"""

from datetime import timedelta

from common.pipeline_config import AIRFLOW_ALERT_EMAIL

BASE_DEFAULT_ARGS = {
    'owner': 'quantum_pipeline',
    'depends_on_past': False,
    'email': [AIRFLOW_ALERT_EMAIL],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'retry_exponential_backoff': True,
    'max_retry_delay': timedelta(hours=1),
}


def make_default_args(**overrides):
    """Create default_args dict with project-wide baseline and DAG-specific overrides."""
    return {**BASE_DEFAULT_ARGS, **overrides}
