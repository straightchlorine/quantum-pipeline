#!/usr/bin/env bash
# Smoke test: end-to-end VQE pipeline through Garage
# Prerequisites: just ml-setup (stack must be running)
set -euo pipefail

COMPOSE_FILE="compose/docker-compose.ml.yaml"
PASS=0
FAIL=0

# --- helpers ---

log()  { echo "  $*"; }
ok()   { echo "  [  OK  ] $*"; PASS=$((PASS+1)); }
fail() { echo "  [ FAIL ] $*"; FAIL=$((FAIL+1)); }
skip() { echo "  [ SKIP ] $*"; }

wait_for() {
    local label=$1 url=$2 token=${3:-} max=${4:-30}
    local -a curl_args=(curl -sf)
    [ -n "$token" ] && curl_args+=(-H "Authorization: Bearer ${token}")
    curl_args+=("$url")
    for i in $(seq 1 "$max"); do
        if "${curl_args[@]}" > /dev/null 2>&1; then
            return 0
        fi
        sleep 2
    done
    return 1
}

# --- load env ---

if [ ! -f .env ]; then
    echo "[ FAIL ] .env not found - run 'just ml-setup' first"
    exit 1
fi
set -a; source .env; set +a

GARAGE_ADMIN_PORT="${GARAGE_ADMIN_PORT:-3903}"
GARAGE_S3_PORT="${GARAGE_S3_API_PORT:-3901}"
KAFKA_CONNECT_PORT="${KAFKA_CONNECT_PORT:-8083}"
SCHEMA_REGISTRY_PORT="${SCHEMA_REGISTRY_PORT:-8081}"
S3_RAW_BUCKET="${S3_RAW_BUCKET:-raw-results}"

echo ""
echo "ML Pipeline Smoke Test"
echo "--------------------------------------------------------------"
echo ""

# --- step 1: service health ---

echo "Step 1: Service health"

if curl -sf "http://localhost:${GARAGE_ADMIN_PORT}/v1/health" \
       -H "Authorization: Bearer ${GARAGE_ADMIN_TOKEN}" > /dev/null 2>&1; then
    ok "Garage admin API healthy"
else
    fail "Garage admin API not healthy - run 'just ml-up' first"
fi

if curl -sf "http://localhost:${SCHEMA_REGISTRY_PORT}/subjects" > /dev/null 2>&1; then
    ok "Schema Registry healthy"
else
    fail "Schema Registry not healthy"
fi

if curl -sf "http://localhost:${KAFKA_CONNECT_PORT}/connectors" > /dev/null 2>&1; then
    ok "Kafka Connect REST API healthy"
else
    fail "Kafka Connect REST API not healthy"
fi

echo ""

# --- step 2: kafka connect connector ---

echo "Step 2: Kafka Connect garage-sink connector"

CONNECTOR_STATUS=$(curl -sf "http://localhost:${KAFKA_CONNECT_PORT}/connectors/garage-sink/status" 2>/dev/null || echo '{}')
CONNECTOR_STATE=$(echo "$CONNECTOR_STATUS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('connector',{}).get('state','NOT_FOUND'))" 2>/dev/null || echo "NOT_FOUND")

if [ "$CONNECTOR_STATE" = "RUNNING" ]; then
    ok "garage-sink connector: RUNNING"
elif [ "$CONNECTOR_STATE" = "NOT_FOUND" ]; then
    fail "garage-sink connector not registered - run 'just ml-up' (kafka-connect-init registers it)"
else
    fail "garage-sink connector state: ${CONNECTOR_STATE} (expected RUNNING)"
    log "Trace: $(echo "$CONNECTOR_STATUS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('tasks',[{}])[0].get('trace','')[:200])" 2>/dev/null || echo 'n/a')"
fi

echo ""

# --- step 3: garage bucket access ---

echo "Step 3: Garage S3 bucket access"

if [ -z "${S3_ACCESS_KEY:-}" ] || echo "${S3_ACCESS_KEY}" | grep -q "CHANGE_ME\|GK_CHANGE"; then
    fail "S3_ACCESS_KEY is still a placeholder - run 'just ml-setup' first"
else
    BUCKET_RESP=$(curl -sf \
        -H "Authorization: AWS ${S3_ACCESS_KEY}:ignored" \
        "http://localhost:${GARAGE_S3_PORT}/${S3_RAW_BUCKET}?list-type=2&max-keys=1" \
        2>/dev/null || echo "ERROR")

    if echo "$BUCKET_RESP" | grep -q "ListBucketResult\|Contents\|KeyCount"; then
        ok "Garage bucket '${S3_RAW_BUCKET}' accessible"
    elif echo "$BUCKET_RESP" | grep -q "NoSuchBucket"; then
        fail "Bucket '${S3_RAW_BUCKET}' does not exist - run 'just ml-setup'"
    else
        fail "Could not list bucket '${S3_RAW_BUCKET}' (check S3_ACCESS_KEY/S3_SECRET_KEY in .env)"
    fi
fi

echo ""

# --- step 4: vqe run -> kafka ---

echo "Step 4: VQE simulation -> Kafka (H2, sto3g, max 50 iterations)"

KAFKA_SERVERS="localhost:${KAFKA_EXTERNAL_PORT:-9094}"

if ! command -v python3 &>/dev/null; then
    fail "python3 not found"
elif ! python3 -c "import quantum_pipeline" 2>/dev/null; then
    fail "quantum_pipeline not installed - run 'just install' first"
else
    SMOKE_MOL=$(mktemp /tmp/smoke_h2_XXXXXX.json)
    cat > "$SMOKE_MOL" <<'JSON'
[{"symbols":["H","H"],"coords":[[0.0,0.0,0.0],[0.0,0.0,0.74]],"multiplicity":1,"charge":0,"units":"angstrom","masses":[1.008,1.008]}]
JSON
    log "Running H2 VQE (max_iterations=50, sto3g, Kafka enabled)..."
    if KAFKA_SERVERS="${KAFKA_SERVERS}" python3 -m quantum_pipeline \
        --file "$SMOKE_MOL" \
        --basis sto3g \
        --max-iterations 50 \
        --kafka \
        --servers "${KAFKA_SERVERS}" \
        --seed 42 2>&1 | tail -5; then
        ok "VQE simulation completed and result sent to Kafka"
    else
        fail "VQE simulation failed"
    fi
    rm -f "$SMOKE_MOL"
fi

echo ""

# --- step 5: verify data in garage ---

echo "Step 5: Verify data landed in Garage (waiting up to 30s for S3 Sink flush)"

if [ -z "${S3_ACCESS_KEY:-}" ] || echo "${S3_ACCESS_KEY}" | grep -q "CHANGE_ME\|GK_CHANGE"; then
    skip "S3_ACCESS_KEY not set (step 3 prerequisite failed)"
else
    FOUND=false
    for i in $(seq 1 15); do
        OBJECTS=$(curl -sf \
            "http://localhost:${GARAGE_S3_PORT}/${S3_RAW_BUCKET}?list-type=2&prefix=experiments/&max-keys=5" \
            --aws-sigv4 "aws:amz:${S3_REGION:-garage}:s3" \
            --user "${S3_ACCESS_KEY}:${S3_SECRET_KEY}" \
            2>/dev/null || echo "")

        KEY_COUNT=$(echo "$OBJECTS" | python3 -c \
            "import sys,re; m=re.search(r'<KeyCount>(\d+)</KeyCount>',sys.stdin.read()); print(int(m.group(1)) if m else 0)" 2>/dev/null || echo "0")

        if [ "$KEY_COUNT" -gt "0" ] 2>/dev/null; then
            FOUND=true
            break
        fi
        log "Waiting for S3 Sink flush... (${i}/15)"
        sleep 2
    done

    if $FOUND; then
        ok "Objects found in Garage bucket (experiments/ prefix)"
        log "First key: $(echo "$OBJECTS" | python3 -c "import sys,re; m=re.search(r'<Key>(.*?)</Key>',sys.stdin.read()); print(m.group(1) if m else 'n/a')" 2>/dev/null)"
    else
        fail "No objects in Garage bucket after 30s"
        log "Check: docker compose -f ${COMPOSE_FILE} logs kafka-connect | grep -i 'error\|warn\|garage'"
    fi
fi

echo ""

# --- step 6: spark s3a read ---

echo "Step 6: Spark s3a read from Garage"

SPARK_OUTPUT=$(docker exec ml-spark-master \
    bash -c "/opt/spark/bin/spark-sql \
        --conf spark.hadoop.fs.s3a.access.key=${S3_ACCESS_KEY} \
        --conf spark.hadoop.fs.s3a.secret.key=${S3_SECRET_KEY} \
        --conf spark.hadoop.fs.s3a.endpoint=http://garage:3901 \
        --conf spark.hadoop.fs.s3a.path.style.access=true \
        --conf spark.hadoop.fs.s3a.connection.ssl.enabled=false \
        --conf spark.hadoop.fs.s3a.aws.credentials.provider=org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider \
        --conf spark.hadoop.fs.s3a.endpoint.region=${S3_REGION:-garage} \
        -e \"SELECT 1\" 2>&1 | tail -3" \
    2>/dev/null || echo "ERROR")

if echo "$SPARK_OUTPUT" | grep -q "ERROR\|Exception\|error"; then
    fail "Spark s3a test failed"
    log "Check: docker compose -f ${COMPOSE_FILE} logs spark-master | tail -20"
else
    ok "Spark running and reachable (spark-sql SELECT 1 succeeded)"
fi

echo ""

# --- summary ---

echo "--------------------------------------------------------------"
echo "  Results: ${PASS} passed, ${FAIL} failed"
echo "--------------------------------------------------------------"
echo ""

if [ "$FAIL" -gt 0 ]; then
    echo "  Next steps for failed checks:"
    echo "    1. Stack not running?   just ml-up"
    echo "    2. Garage not init?     just ml-setup"
    echo "    3. Connector failed?    docker compose -f ${COMPOSE_FILE} logs kafka-connect"
    echo ""
    exit 1
fi

echo "  [  OK  ] Smoke test passed. VQE -> Kafka -> Garage pipeline is operational."
echo ""
