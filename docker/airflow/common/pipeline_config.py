"""Centralized configuration for the quantum data pipeline.

Single source of truth for S3 paths, catalog names, ML feature parameters,
and other shared settings. All values are read from environment variables
with sensible defaults.
"""

import os

# S3 storage paths
S3_BUCKET_URL = os.getenv('S3_BUCKET_URL', 's3a://raw-results/experiments/')
S3_WAREHOUSE_URL = os.getenv('S3_WAREHOUSE_URL', 's3a://features/warehouse/')

# Iceberg catalog
CATALOG_FQN = 'quantum_catalog.quantum_features'

# ML feature engineering parameters
ML_ROLLING_WINDOW = int(os.getenv('ML_ROLLING_WINDOW', '5'))
ML_TRAJECTORY_HEAD = int(os.getenv('ML_TRAJECTORY_HEAD', '10'))
ML_TRAJECTORY_TAIL = int(os.getenv('ML_TRAJECTORY_TAIL', '10'))

# Alert configuration
AIRFLOW_ALERT_EMAIL = os.getenv('AIRFLOW_ALERT_EMAIL', 'quantum_alerts@example.com')
