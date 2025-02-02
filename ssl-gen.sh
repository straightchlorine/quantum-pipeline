#!/bin/bash
set -e

# -------------------------------
# Load .env file if present
# -------------------------------
ENV_FILE=".env"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
    echo "Loaded environment variables from $ENV_FILE."
fi

# -------------------------------
# Check dependencies
# -------------------------------
command -v openssl >/dev/null 2>&1 || { echo "Error: openssl is not installed." >&2; exit 1; }
command -v keytool >/dev/null 2>&1 || { echo "Error: keytool is not installed." >&2; exit 1; }

# -------------------------------
# Set output directory for secrets
# -------------------------------
OUTPUT_DIR="./secrets"
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Creating output directory at $OUTPUT_DIR..."
    mkdir -p "$OUTPUT_DIR"
fi

# -------------------------------
# Get hostnames
# -------------------------------
while [[ -z "$HOSTNAMES" ]]; do
    read -p "Enter the hostname(s) (comma separated, e.g., kafka,localhost): " HOSTNAMES
done

# -------------------------------
# Prompt for passwords if not provided in .env
# -------------------------------
# Use CERTIFICATE_PASSWORD from .env for the CA password if set.
if [ -z "$CERTIFICATE_PASSWORD" ]; then
    read -s -p "Enter CA (Certificate Authority) password: " CA_PASS
    echo
else
    CA_PASS="$CERTIFICATE_PASSWORD"
fi

# Use KEYSTORE_PASSWORD from .env for keystore if set.
if [ -z "$KEYSTORE_PASSWORD" ]; then
    read -s -p "Enter keystore password: " KEYSTORE_PASS
    echo
else
    KEYSTORE_PASS="$KEYSTORE_PASSWORD"
fi

# Use TRUSTSTORE_PASSWORD from .env for truststore if set.
if [ -z "$TRUSTSTORE_PASSWORD" ]; then
    read -s -p "Enter truststore password: " TRUSTSTORE_PASS
    echo
else
    TRUSTSTORE_PASS="$TRUSTSTORE_PASSWORD"
fi

# Always prompt for the client certificate export password
read -s -p "Enter client certificate export password (for PKCS12 file): " CLIENT_PASS
echo

# -------------------------------
# Define filenames (all files will be placed in the OUTPUT_DIR)
# -------------------------------
CA_KEY="${OUTPUT_DIR}/ca.key"
CA_CERT="${OUTPUT_DIR}/ca.crt"
CLIENT_KEY="${OUTPUT_DIR}/client.key"
CLIENT_CSR="${OUTPUT_DIR}/client.csr"
CLIENT_CERT="${OUTPUT_DIR}/client.crt"
PKCS12_FILE="${OUTPUT_DIR}/kafka.p12"
KEYSTORE="${OUTPUT_DIR}/kafka.keystore.jks"
TRUSTSTORE="${OUTPUT_DIR}/kafka.truststore.jks"
SAN_FILE="${OUTPUT_DIR}/san.cnf"

# -------------------------------
# Backup any existing files
# -------------------------------
for file in "$CA_KEY" "$CA_CERT" "$CLIENT_KEY" "$CLIENT_CSR" "$CLIENT_CERT" "$PKCS12_FILE" "$KEYSTORE" "$TRUSTSTORE"; do
    if [ -f "$file" ]; then
        mv "$file" "${file}.backup_$(date +%Y%m%d%H%M%S)"
        echo "Backed up existing file $file"
    fi
done

# -------------------------------
# Step 1: Generate CA Key and Self-Signed Certificate
# -------------------------------
echo "Generating CA key and self-signed certificate..."
openssl genrsa -aes256 -passout pass:"$CA_PASS" -out "$CA_KEY" 4096
openssl req -x509 -new -key "$CA_KEY" -days 365 -out "$CA_CERT" \
    -subj "/CN=Certificate Authority" -passin pass:"$CA_PASS"

# -------------------------------
# Step 2: Generate Client Key and CSR
# -------------------------------
PRIMARY_HOST=$(echo "$HOSTNAMES" | cut -d',' -f1 | xargs)
echo "Generating client private key and certificate signing request (CSR)..."
openssl genrsa -out "$CLIENT_KEY" 4096
openssl req -new -key "$CLIENT_KEY" -out "$CLIENT_CSR" -subj "/CN=${PRIMARY_HOST}"

# -------------------------------
# Step 3: Create a SAN (Subject Alternative Names) Config File
# -------------------------------
echo "Creating SAN configuration file..."
cat > "$SAN_FILE" <<EOF
[ req ]
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no

[ req_distinguished_name ]
CN = ${PRIMARY_HOST}

[ v3_req ]
subjectAltName = @alt_names

[ alt_names ]
EOF

# Append each hostname as a DNS entry
IFS=',' read -ra HOST_ARR <<< "$HOSTNAMES"
i=1
for host in "${HOST_ARR[@]}"; do
    host=$(echo "$host" | xargs)  # trim spaces
    echo "DNS.$i = $host" >> "$SAN_FILE"
    ((i++))
done

# -------------------------------
# Step 4: Sign the CSR with the CA Key to Generate the Client Certificate
# -------------------------------
echo "Signing the client CSR with the CA key to generate the client certificate..."
openssl x509 -req -in "$CLIENT_CSR" -CA "$CA_CERT" -CAkey "$CA_KEY" -CAcreateserial \
    -out "$CLIENT_CERT" -days 365 -extfile "$SAN_FILE" -passin pass:"$CA_PASS"

# -------------------------------
# Step 5: Create a PKCS12 File from Client Key and Certificate
# -------------------------------
echo "Creating PKCS12 file from client key and certificate..."
openssl pkcs12 -export -in "$CLIENT_CERT" -inkey "$CLIENT_KEY" -certfile "$CA_CERT" \
    -name kafka -out "$PKCS12_FILE" -passout pass:"$CLIENT_PASS"

# -------------------------------
# Step 6: Create the Java Keystore (.jks) Using keytool
# -------------------------------
echo "Importing PKCS12 file into Java keystore..."
keytool -importkeystore \
    -deststorepass "$KEYSTORE_PASS" -destkeypass "$KEYSTORE_PASS" \
    -destkeystore "$KEYSTORE" \
    -srckeystore "$PKCS12_FILE" -srcstoretype PKCS12 -srcstorepass "$CLIENT_PASS" \
    -alias kafka

# -------------------------------
# Step 7: Create the Java Truststore (.jks) and Import the CA Certificate
# -------------------------------
echo "Creating Java truststore and importing CA certificate..."
keytool -import -file "$CA_CERT" -alias CARoot \
    -keystore "$TRUSTSTORE" -storepass "$TRUSTSTORE_PASS" -noprompt

# -------------------------------
# Cleanup temporary files
# -------------------------------
rm -f "$SAN_FILE" "$CLIENT_CSR" "$PKCS12_FILE" "${OUTPUT_DIR}/ca.srl"

# -------------------------------
# Summary of Generated Files
# -------------------------------
echo "----------------------------------------------"
echo "Certificates and keystores created successfully in $OUTPUT_DIR!"
echo "Files generated:"
echo "  - CA certificate: $CA_CERT"
echo "  - Client certificate: $CLIENT_CERT"
echo "  - Client private key: $CLIENT_KEY"
echo "  - Java keystore: $KEYSTORE"
echo "  - Java truststore: $TRUSTSTORE"
echo "----------------------------------------------"
