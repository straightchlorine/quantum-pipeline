#!/usr/bin/env bash
set -euo pipefail

# ML Pipeline first-time setup
# Generates .env secrets, starts the stack, and configures Garage (layout, keys, buckets).
# Usage: bash scripts/ml-setup.sh

# --- step 1: generate .env ---

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

if grep -q "AIRFLOW_WEBSERVER_SECRET_KEY=your-webserver-secret-here" .env; then
    WS_KEY=$(openssl rand -hex 32)
    sed -i "s|AIRFLOW_WEBSERVER_SECRET_KEY=your-webserver-secret-here|AIRFLOW_WEBSERVER_SECRET_KEY=${WS_KEY}|" .env
    echo "[  OK  ] Generated AIRFLOW_WEBSERVER_SECRET_KEY"
fi

# --- step 2: start the stack ---

echo ""
echo "[ INFO ] Starting ML stack..."
docker compose -f compose/docker-compose.ml.yaml up -d
echo "[  OK  ] ML stack started"

# --- step 3: configure garage ---

echo ""
echo "[ INFO ] Configuring Garage..."

export $(grep -E '^(GARAGE_ADMIN_TOKEN|GARAGE_S3_API_PORT|GARAGE_ADMIN_PORT|S3_RAW_BUCKET|S3_FEATURES_BUCKET|S3_ICEBERG_BUCKET)=' .env | xargs)
GARAGE_ADMIN_PORT="${GARAGE_ADMIN_PORT:-3903}"
GARAGE_API="http://localhost:${GARAGE_ADMIN_PORT}/v1"
AUTH="Authorization: Bearer ${GARAGE_ADMIN_TOKEN}"

echo "[ INFO ] Waiting for Garage admin API..."
for i in $(seq 1 20); do
    if curl -sf -H "${AUTH}" "${GARAGE_API}/health" > /dev/null 2>&1; then
        echo "[  OK  ] Garage is healthy"
        break
    fi
    [ "$i" -eq 20 ] && echo "[ FAIL ] Garage not responding after 60s" && exit 1
    echo "         attempt $i/20, retrying in 3s..."
    sleep 3
done

echo "[ INFO ] Getting node ID..."
NODE_RESP=$(curl -sf -H "${AUTH}" "${GARAGE_API}/node")
NODE_ID=$(echo "${NODE_RESP}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('id','')[:16])")
[ -z "${NODE_ID}" ] && echo "[ FAIL ] Could not get node ID" && exit 1
echo "[  OK  ] Node: ${NODE_ID}"

echo "[ INFO ] Assigning layout..."
curl -sf -X POST -H "${AUTH}" -H "Content-Type: application/json" \
    -d "[{\"id\": \"${NODE_ID}\", \"zone\": \"local\", \"capacity\": 10737418240, \"tags\": []}]" \
    "${GARAGE_API}/layout" > /dev/null
curl -sf -X POST -H "${AUTH}" -H "Content-Type: application/json" \
    -d "{\"version\": 1}" \
    "${GARAGE_API}/layout/apply" > /dev/null
echo "[  OK  ] Layout applied (zone=local, 10G)"

echo "[ INFO ] Creating access key..."
KEY_RESP=$(curl -sf -X POST -H "${AUTH}" -H "Content-Type: application/json" \
    -d "{\"name\": \"ml-pipeline\"}" \
    "${GARAGE_API}/key")
KEY_ID=$(echo "${KEY_RESP}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('accessKeyId',''))")
KEY_SECRET=$(echo "${KEY_RESP}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('secretAccessKey',''))")
[ -z "${KEY_ID}" ] && echo "[ FAIL ] Could not create access key" && exit 1
echo "[  OK  ] Key: ${KEY_ID}"

echo "[ INFO ] Creating buckets and granting access..."
for BUCKET in "${S3_RAW_BUCKET:-local-vqe-results}" "${S3_FEATURES_BUCKET:-local-features}" "${S3_ICEBERG_BUCKET:-iceberg}"; do
    curl -sf -X POST -H "${AUTH}" -H "Content-Type: application/json" \
        -d "{\"globalAlias\": \"${BUCKET}\"}" \
        "${GARAGE_API}/bucket" > /dev/null 2>&1 || true
    BUCKET_ID=$(curl -sf -H "${AUTH}" "${GARAGE_API}/bucket?globalAlias=${BUCKET}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('id',''))" 2>/dev/null || echo "")
    if [ -n "${BUCKET_ID}" ]; then
        curl -sf -X POST -H "${AUTH}" -H "Content-Type: application/json" \
            -d "{\"bucketId\": \"${BUCKET_ID}\", \"accessKeyId\": \"${KEY_ID}\", \"permissions\": {\"read\": true, \"write\": true, \"owner\": false}}" \
            "${GARAGE_API}/bucket/allowed-keys" > /dev/null
        echo "[  OK  ] Bucket: ${BUCKET}"
    else
        echo "[ WARN ] Could not configure bucket: ${BUCKET}"
    fi
done

sed -i "s#S3_ACCESS_KEY=.*#S3_ACCESS_KEY=${KEY_ID}#" .env
sed -i "s#S3_SECRET_KEY=.*#S3_SECRET_KEY=${KEY_SECRET}#" .env
echo "[  OK  ] .env updated with credentials"

# --- step 4: restart to pick up new creds ---

echo ""
echo "[ INFO ] Restarting stack with new credentials..."
docker compose -f compose/docker-compose.ml.yaml down
docker compose -f compose/docker-compose.ml.yaml up -d
echo ""
echo "[  OK  ] ML pipeline setup complete"
docker compose -f compose/docker-compose.ml.yaml ps
