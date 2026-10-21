#!/usr/bin/env bash
# ML pipeline first-time setup.
#
# 1. Creates .env from .env.ml.example and fills in every generated secret.
# 2. Renders compose/spark-defaults.conf from its template.
# 3. Sets up the object store, one of two ways:
#    - S3_ENDPOINT in .env names an instance that will be used (any S3 service):
#      only the four buckets are created, using the key from .env.
#      That key must be allowed to create buckets. For garage it's:
#      `garage key allow --create-bucket <key>` once.
#    - otherwise the compose Garage: layout, the ml-pipeline key (written to .env), buckets.
#      If .env named an endpoint that does not answer, it is pointed back at the compose
#      Garage first, so the stack and the rendered files agree.
#
# Usage: bash scripts/ml-setup.sh        (from the repository root; safe to re-run)
set -euo pipefail

COMPOSE=(docker compose --env-file .env -f compose/docker-compose.ml.yaml)
GARAGE=(docker exec ml-garage /garage)
LOCAL_S3_ENDPOINT="http://garage:3901"
LOCAL_KEY_NAME="ml-pipeline"
LOCAL_ZONE="dc1"
LOCAL_CAPACITY="10G"

info() { echo "[ INFO ] $*"; }
ok()   { echo "[  OK  ] $*"; }
skip() { echo "[ SKIP ] $*"; }
warn() { echo "[ WARN ] $*"; }
die()  { echo "[ FAIL ] $*" >&2; exit 1; }

# --- .env and secrets -------------------------------------------------------------------

if [ ! -f .env ]; then
    cp .env.ml.example .env
    ok "Created .env from .env.ml.example"
else
    skip ".env already exists"
fi

# gen_secret VAR PLACEHOLDER COMMAND: replace VAR's placeholder value with COMMAND's output.
gen_secret() {
    local var=$1 placeholder=$2 value
    if grep -q "^${var}=${placeholder}" .env; then
        value=$(eval "$3")
        sed -i "s|^${var}=${placeholder}.*|${var}=${value}|" .env
        ok "Generated ${var}"
    fi
}
gen_secret GARAGE_RPC_SECRET            CHANGE_ME_32_BYTE_HEX        'openssl rand -hex 32'
gen_secret GARAGE_ADMIN_TOKEN           CHANGE_ME_32_BYTE_HEX        'openssl rand -hex 32'
gen_secret AIRFLOW_POSTGRES_PASSWORD    airflow-password             'openssl rand -hex 16'
gen_secret AIRFLOW_ADMIN_PASSWORD       admin                        'openssl rand -hex 16'
gen_secret AIRFLOW_FERNET_KEY           your-fernet-key-here \
    'python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())" 2>/dev/null || openssl rand -base64 32'
gen_secret AIRFLOW_JWT_SECRET           your-jwt-secret-here         'openssl rand -hex 32'
gen_secret AIRFLOW_WEBSERVER_SECRET_KEY your-webserver-secret-here   'openssl rand -hex 32'

set -a; source .env; set +a
S3_ENDPOINT="${S3_ENDPOINT:-${LOCAL_S3_ENDPOINT}}"
S3_REGION="${S3_REGION:-garage}"
BUCKETS=("${S3_RAW_BUCKET:-raw-results}" "${S3_FEATURES_BUCKET:-features}" "${S3_ICEBERG_BUCKET:-warehouse}" "mlflow-artifacts")

# --- which object store -------------------------------------------------------------------

REMOTE=false
if [ "${S3_ENDPOINT}" != "${LOCAL_S3_ENDPOINT}" ]; then
    # Garage answers an anonymous request with 403; any HTTP status means something is there.
    PROBE=$(curl -s -o /dev/null -m 5 -w '%{http_code}' "${S3_ENDPOINT}/" 2>/dev/null) || PROBE=000
    if [ "${PROBE}" != "000" ]; then
        ok "${S3_ENDPOINT} answers (HTTP ${PROBE}); using it, the compose Garage stays off"
        REMOTE=true
    else
        warn "${S3_ENDPOINT} did not answer; falling back to the compose Garage"
        sed -i "s#^S3_ENDPOINT=.*#S3_ENDPOINT=${LOCAL_S3_ENDPOINT}#" .env
        S3_ENDPOINT="${LOCAL_S3_ENDPOINT}"
        ok ".env now points S3_ENDPOINT at ${LOCAL_S3_ENDPOINT}"
    fi
fi

# The compose files mount the rendered copy; the template carries ${S3_ENDPOINT} and ${S3_REGION}.
S3_ENDPOINT="${S3_ENDPOINT}" S3_REGION="${S3_REGION}" \
    envsubst < compose/spark-defaults.conf.template > compose/spark-defaults.conf
ok "compose/spark-defaults.conf generated (S3_ENDPOINT=${S3_ENDPOINT})"

# --- external instance: buckets over S3 ---------------------------------------------------

setup_remote() {
    case "${S3_ACCESS_KEY:-}" in
        ""|GK_CHANGE_ME) die "S3_ACCESS_KEY is not set in .env" ;;
    esac
    local sigv4="aws:amz:${S3_REGION}:s3" auth="${S3_ACCESS_KEY}:${S3_SECRET_KEY}" body code bucket
    body=$(mktemp)
    trap 'rm -f "${body}"' RETURN

    info "Creating buckets on ${S3_ENDPOINT}: ${BUCKETS[*]}"
    for bucket in "${BUCKETS[@]}"; do
        code=$(curl -s -o "${body}" -w '%{http_code}' -X PUT --aws-sigv4 "${sigv4}" \
            --user "${auth}" "${S3_ENDPOINT}/${bucket}") || code=000
        case "${code}" in
            200) ok "${bucket}" ;;
            409) skip "${bucket} already exists" ;;
            403) echo "         $(grep -o '<Message>[^<]*' "${body}" | cut -c10-)"
                echo "         On the Garage node run: garage key allow --create-bucket ${S3_ACCESS_KEY}"
                die "${bucket}: the key may not create buckets" ;;
            000) die "${S3_ENDPOINT} unreachable" ;;
            *)   cat "${body}"; echo; die "${bucket}: HTTP ${code}" ;;
        esac
    done

    info "Buckets visible to ${S3_ACCESS_KEY}:"
    curl -s --aws-sigv4 "${sigv4}" --user "${auth}" "${S3_ENDPOINT}/" \
        | grep -o '<Name>[^<]*' | cut -c7- | sed 's/^/           /'
}

# --- compose Garage: layout, key, buckets -------------------------------------------------

setup_compose() {
    local i node_id key_output key_id key_secret bucket
    envsubst < compose/garage.toml.template > compose/garage.toml
    ok "compose/garage.toml generated"

    info "Starting Garage to configure layout, access key and buckets"
    "${COMPOSE[@]}" up -d garage
    for i in $(seq 1 20); do
        "${GARAGE[@]}" status > /dev/null 2>&1 && break
        [ "$i" -eq 20 ] && die "Garage not responding after 60 s"
        sleep 3
    done
    ok "Garage is ready"

    # `garage status` prints a header line, then one row per node; the id is the first column.
    node_id=$("${GARAGE[@]}" status 2>/dev/null | awk 'NR>1 && $1 != "" {id=$1} END {print id}')
    [ -n "${node_id}" ] || die "Could not read the node id from garage status"
    if "${GARAGE[@]}" layout show 2>/dev/null | grep -q "${node_id:0:12}"; then
        skip "Layout already assigned for ${node_id:0:12}"
    else
        "${GARAGE[@]}" layout assign -z "${LOCAL_ZONE}" -c "${LOCAL_CAPACITY}" "${node_id}" > /dev/null
        "${GARAGE[@]}" layout apply --version 1 > /dev/null
        ok "Layout applied (node=${node_id:0:12} zone=${LOCAL_ZONE} capacity=${LOCAL_CAPACITY})"
    fi

    # `key create` prints the secret once; `key info` redacts it.
    # If secret never reached the .env, it cannot be recovered; delete and create again.
    if "${GARAGE[@]}" key info "${LOCAL_KEY_NAME}" > /dev/null 2>&1 \
        && grep -q -E "^S3_ACCESS_KEY=(GK_CHANGE_ME|)$" .env; then
        info "Key ${LOCAL_KEY_NAME} exists but .env has no credentials; recreating it"
        "${GARAGE[@]}" key delete --yes "${LOCAL_KEY_NAME}" > /dev/null
    fi
    if "${GARAGE[@]}" key info "${LOCAL_KEY_NAME}" > /dev/null 2>&1; then
        skip "Key ${LOCAL_KEY_NAME} already exists with credentials in .env"
        key_id=$(grep '^S3_ACCESS_KEY=' .env | cut -d= -f2)
        key_secret=$(grep '^S3_SECRET_KEY=' .env | cut -d= -f2)
    else
        key_output=$("${GARAGE[@]}" key create "${LOCAL_KEY_NAME}")
        key_id=$(echo "${key_output}" | awk '/Key ID:/ {print $3}')
        key_secret=$(echo "${key_output}" | awk '/Secret key:/ {print $3}')
        ok "Created key ${LOCAL_KEY_NAME}"
    fi
    [ -n "${key_id}" ] || die "Could not get the access key id"
    [ -n "${key_secret}" ] && [ "${key_secret}" != "(redacted)" ] || die "Could not get the secret key"

    info "Creating buckets: ${BUCKETS[*]}"
    for bucket in "${BUCKETS[@]}"; do
        "${GARAGE[@]}" bucket create "${bucket}" > /dev/null 2>&1 || true
        "${GARAGE[@]}" bucket allow --read --write "${bucket}" --key "${LOCAL_KEY_NAME}" > /dev/null
        ok "${bucket}"
    done

    sed -i "s#^S3_ACCESS_KEY=.*#S3_ACCESS_KEY=${key_id}#; s#^S3_SECRET_KEY=.*#S3_SECRET_KEY=${key_secret}#" .env
    ok ".env updated with the S3 credentials"
    "${COMPOSE[@]}" down > /dev/null 2>&1
    echo "           Garage:  node=${node_id:0:12} zone=${LOCAL_ZONE} capacity=${LOCAL_CAPACITY}, key ${key_id}"
}

if [ "${REMOTE}" = true ]; then setup_remote; else setup_compose; fi

echo ""
ok "ML pipeline setup complete."
echo "           Object store: ${S3_ENDPOINT} (region ${S3_REGION})"
echo "           Buckets:      ${BUCKETS[*]}"
echo "           Spark:        ${SPARK_VERSION:-4.0.2} (Iceberg + S3 via spark-defaults.conf)"
echo "           Airflow:      ${AIRFLOW_VERSION:-3.1.8} (CeleryExecutor + Redis)"
echo ""
echo "         Start the stack:"
echo "           docker compose --env-file .env -f compose/docker-compose.ml.yaml up -d"
