#!/usr/bin/env bash
set -euo pipefail

# ML Pipeline first-time setup
# Generates .env, configures Garage (layout, keys, buckets) via the garage CLI.
# Usage: bash scripts/ml-setup.sh

COMPOSE="docker compose --env-file .env -f compose/docker-compose.ml.yaml"
GARAGE="docker exec ml-garage /garage"

# --- step 1: generate .env and garage.toml ---

echo "[ INFO ] Setting up .env..."

if [ ! -f .env ]; then
    cp .env.ml.example .env
    echo "[  OK  ] Created .env from .env.ml.example"
else
    echo "[ SKIP ] .env already exists"
fi

if grep -q "GARAGE_RPC_SECRET=CHANGE_ME" .env; then
    RPC_SECRET=$(openssl rand -hex 32)
    sed -i "s|GARAGE_RPC_SECRET=CHANGE_ME_32_BYTE_HEX|GARAGE_RPC_SECRET=${RPC_SECRET}|" .env
    echo "[  OK  ] Generated GARAGE_RPC_SECRET"
else
    echo "[ SKIP ] GARAGE_RPC_SECRET already set"
fi

if grep -q "GARAGE_ADMIN_TOKEN=CHANGE_ME" .env; then
    ADMIN_TOKEN=$(openssl rand -hex 32)
    sed -i "s|GARAGE_ADMIN_TOKEN=CHANGE_ME_32_BYTE_HEX|GARAGE_ADMIN_TOKEN=${ADMIN_TOKEN}|" .env
    echo "[  OK  ] Generated GARAGE_ADMIN_TOKEN"
else
    echo "[ SKIP ] GARAGE_ADMIN_TOKEN already set"
fi

if grep -q "AIRFLOW_POSTGRES_PASSWORD=airflow-password" .env; then
    PG_PASS=$(openssl rand -hex 16)
    sed -i "s#AIRFLOW_POSTGRES_PASSWORD=airflow-password#AIRFLOW_POSTGRES_PASSWORD=${PG_PASS}#" .env
    echo "[  OK  ] Generated AIRFLOW_POSTGRES_PASSWORD"
fi

if grep -q "AIRFLOW_ADMIN_PASSWORD=admin" .env; then
    ADMIN_PASS=$(openssl rand -hex 16)
    sed -i "s#AIRFLOW_ADMIN_PASSWORD=admin#AIRFLOW_ADMIN_PASSWORD=${ADMIN_PASS}#" .env
    echo "[  OK  ] Generated AIRFLOW_ADMIN_PASSWORD"
fi

if grep -q "AIRFLOW_FERNET_KEY=your-fernet-key-here" .env; then
    FERNET_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())" 2>/dev/null || openssl rand -base64 32)
    sed -i "s|AIRFLOW_FERNET_KEY=your-fernet-key-here|AIRFLOW_FERNET_KEY=${FERNET_KEY}|" .env
    echo "[  OK  ] Generated AIRFLOW_FERNET_KEY"
fi

if grep -q "AIRFLOW_JWT_SECRET=your-jwt-secret-here" .env; then
    JWT_KEY=$(openssl rand -hex 32)
    sed -i "s|AIRFLOW_JWT_SECRET=your-jwt-secret-here|AIRFLOW_JWT_SECRET=${JWT_KEY}|" .env
    echo "[  OK  ] Generated AIRFLOW_JWT_SECRET"
fi

if grep -q "AIRFLOW_WEBSERVER_SECRET_KEY=your-webserver-secret-here" .env; then
    WS_KEY=$(openssl rand -hex 32)
    sed -i "s|AIRFLOW_WEBSERVER_SECRET_KEY=your-webserver-secret-here|AIRFLOW_WEBSERVER_SECRET_KEY=${WS_KEY}|" .env
    echo "[  OK  ] Generated AIRFLOW_WEBSERVER_SECRET_KEY"
fi

# source generated values for envsubst
set -a; source .env; set +a

echo ""
echo "[ INFO ] Generating garage.toml from template..."
envsubst < compose/garage.toml.template > compose/garage.toml
echo "[  OK  ] compose/garage.toml generated"

# --- step 2: start garage for configuration ---

echo ""
echo "[ INFO ] Starting Garage to configure layout, access keys, and S3 buckets..."
$COMPOSE up -d garage
echo "[  OK  ] Garage container started"

echo "[ INFO ] Waiting for Garage to be ready..."
for i in $(seq 1 20); do
    if $GARAGE status > /dev/null 2>&1; then
        echo "[  OK  ] Garage is ready"
        break
    fi
    [ "$i" -eq 20 ] && echo "[ FAIL ] Garage not responding after 60s" && exit 1
    echo "         attempt $i/20, retrying in 3s..."
    sleep 3
done

# --- step 3: configure garage via CLI ---

echo ""
echo "[ INFO ] Configuring Garage S3 storage..."

NODE_ID=$($GARAGE status 2>/dev/null | grep -oP '^\S+' | tail -1)
[ -z "${NODE_ID}" ] && echo "[ FAIL ] Could not get node ID" && exit 1

# layout assign (skip if node already has a role)
LAYOUT=$($GARAGE layout show 2>/dev/null)
if echo "${LAYOUT}" | grep -q "${NODE_ID:0:12}"; then
    echo "[ SKIP ] Layout already assigned for ${NODE_ID:0:12}"
else
    echo "[ INFO ] Assigning cluster layout (node=${NODE_ID:0:12}, zone=dc1, capacity=10G)..."
    $GARAGE layout assign -z dc1 -c 10G "${NODE_ID}" > /dev/null 2>&1
    $GARAGE layout apply --version 1 > /dev/null 2>&1
    echo "[  OK  ] Layout applied"
fi

# key create (reuse existing if present)
KEY_INFO=$($GARAGE key info ml-pipeline 2>/dev/null)
if [ -n "${KEY_INFO}" ]; then
    echo "[ SKIP ] Key ml-pipeline already exists"
    KEY_ID=$(echo "${KEY_INFO}" | grep "Key ID:" | awk '{print $3}')
    KEY_SECRET=$(echo "${KEY_INFO}" | grep "Secret key:" | awk '{print $3}')
else
    echo "[ INFO ] Creating access key (ml-pipeline)..."
    KEY_OUTPUT=$($GARAGE key create ml-pipeline 2>/dev/null)
    KEY_ID=$(echo "${KEY_OUTPUT}" | grep "Key ID:" | awk '{print $3}')
    KEY_SECRET=$(echo "${KEY_OUTPUT}" | grep "Secret key:" | awk '{print $3}')
fi
[ -z "${KEY_ID}" ] && echo "[ FAIL ] Could not get access key" && exit 1
echo "[  OK  ] Key: ${KEY_ID}"

BUCKETS=("${S3_RAW_BUCKET:-raw-results}" "${S3_FEATURES_BUCKET:-features}" "${S3_ICEBERG_BUCKET:-warehouse}" "mlflow-artifacts")
echo "[ INFO ] Creating buckets: ${BUCKETS[*]}..."
for BUCKET in "${BUCKETS[@]}"; do
    $GARAGE bucket create "${BUCKET}" > /dev/null 2>&1 || true
    $GARAGE bucket allow --read --write "${BUCKET}" --key ml-pipeline > /dev/null 2>&1
    echo "[  OK  ] ${BUCKET}"
done

sed -i "s#S3_ACCESS_KEY=.*#S3_ACCESS_KEY=${KEY_ID}#" .env
sed -i "s#S3_SECRET_KEY=.*#S3_SECRET_KEY=${KEY_SECRET}#" .env
echo "[  OK  ] .env updated with S3 credentials"

# --- step 4: stop garage and print summary ---

echo ""
$COMPOSE down > /dev/null 2>&1
echo ""
echo "[  OK  ] ML pipeline setup complete."
echo ""
echo "         Configuration summary:"
echo "           Garage:  node=${NODE_ID:0:12} zone=dc1 capacity=10G"
echo "           Key:     ${KEY_ID} (ml-pipeline)"
echo "           Buckets: ${BUCKETS[*]}"
echo "           Spark:   ${SPARK_VERSION:-4.0.2} (Iceberg + S3 via spark-defaults.conf)"
echo "           Airflow: ${AIRFLOW_VERSION:-3.1.8} (CeleryExecutor + Redis)"
echo ""
echo "         Start the stack:"
echo "           docker compose --env-file .env -f compose/docker-compose.ml.yaml up -d"
