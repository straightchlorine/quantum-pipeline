#!/bin/bash
set -e

# Create additional databases on first Postgres startup.
# The default POSTGRES_DB (airflow) is created automatically.
# This script adds any extra databases needed by the stack.

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE mlflow;
    GRANT ALL PRIVILEGES ON DATABASE mlflow TO $POSTGRES_USER;
EOSQL
