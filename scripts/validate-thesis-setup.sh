#!/bin/bash

# Quantum Pipeline Thesis Setup Validation Script

echo "🔬 Quantum Pipeline Thesis Setup Validation"
echo "=========================================="

# Check system resources
echo "📊 System Resources:"
echo "CPU Cores: $(nproc)"
echo "Memory: $(free -h | awk '/^Mem:/ {print $2}')"
echo "GPU Count: $(nvidia-smi -L 2>/dev/null | wc -l || echo "0")"

if command -v nvidia-smi &> /dev/null; then
    echo "GPU Details:"
    nvidia-smi -L
else
    echo "⚠️  NVIDIA drivers not found - GPU experiments will not work"
fi

echo ""

# Check Docker and Docker Compose
echo "🐳 Docker Environment:"
if command -v docker &> /dev/null; then
    echo "✅ Docker: $(docker --version)"

    # Check if Docker daemon is running
    if docker info &> /dev/null; then
        echo "✅ Docker daemon is running"
    else
        echo "❌ Docker daemon is not running"
        exit 1
    fi
else
    echo "❌ Docker not found"
    exit 1
fi

if command -v docker-compose &> /dev/null; then
    echo "✅ Docker Compose: $(docker-compose --version)"
elif docker compose version &> /dev/null; then
    echo "✅ Docker Compose (plugin): $(docker compose version)"
else
    echo "❌ Docker Compose not found"
    exit 1
fi

echo ""

# Check required files
echo "📁 Required Files:"
files=(
    "docker-compose.thesis.yaml"
    "docker/Dockerfile.cpu"
    "docker/Dockerfile.gpu"
    "monitoring/prometheus.yml"
    "monitoring/grafana/provisioning/datasources/prometheus.yml"
    "data/molecules.json"
)

all_files_exist=true
for file in "${files[@]}"; do
    if [[ -f "$file" ]]; then
        echo "✅ $file"
    else
        echo "❌ $file (missing)"
        all_files_exist=false
    fi
done

if [[ "$all_files_exist" = false ]]; then
    echo "⚠️  Some required files are missing"
fi

echo ""

# Check environment configuration
echo "⚙️  Environment Configuration:"
if [[ -f ".env" ]]; then
    echo "✅ .env file exists"

    # Check critical variables
    if grep -q "MINIO_ROOT_USER=" .env && grep -q "MINIO_ROOT_PASSWORD=" .env; then
        echo "✅ MinIO credentials configured"
    else
        echo "⚠️  MinIO credentials not configured in .env"
    fi

    if grep -q "AIRFLOW_FERNET_KEY=" .env; then
        echo "✅ Airflow Fernet key configured"
    else
        echo "⚠️  Airflow Fernet key not configured in .env"
    fi

    if grep -q "EXTERNAL_PUSHGATEWAY_URL=" .env; then
        echo "✅ External Prometheus PushGateway URL configured"
    else
        echo "⚠️  External PushGateway URL not configured in .env"
    fi
else
    echo "⚠️  .env file not found (copy from .env.thesis.example)"
fi

echo ""

# Validate resource allocation
echo "🎯 Resource Allocation Analysis:"
total_cpu=$(nproc)
total_memory=$(free -m | awk '/^Mem:/ {print $2}')

echo "Total CPU cores: $total_cpu"
echo "Total Memory: ${total_memory}MB"

planned_cpu_usage=$((2 + 2 + 2 + 1 + 1))  # CPU container + GPU1 + GPU2 + Spark + Other services
planned_memory_usage=$((10240 + 10240 + 10240 + 4096 + 4096))  # In MB - three-way comparison

echo "Planned CPU usage: $planned_cpu_usage cores"
echo "Planned Memory usage: ${planned_memory_usage}MB ($(($planned_memory_usage / 1024))GB)"

if [[ $planned_cpu_usage -le $total_cpu ]]; then
    echo "✅ CPU allocation is feasible"
else
    echo "⚠️  CPU over-allocation detected - reduce container limits"
fi

if [[ $planned_memory_usage -le $total_memory ]]; then
    echo "✅ Memory allocation is feasible"
else
    echo "⚠️  Memory over-allocation detected - reduce container limits"
fi

echo ""

# Check if containers are already running
echo "🚀 Container Status:"
if docker-compose -f docker-compose.thesis.yaml ps | grep -q "Up"; then
    echo "✅ Some thesis containers are running:"
    docker-compose -f docker-compose.thesis.yaml ps
else
    echo "ℹ️  No thesis containers currently running"
fi

echo ""

# Final recommendations
echo "📋 Setup Recommendations:"
echo "1. Copy .env.thesis.example to .env and configure external monitoring URLs"
echo "2. Ensure Docker has sufficient resources allocated"
echo "3. Start with: docker-compose -f docker-compose.thesis.yaml up -d"
echo "4. Connect external Grafana to receive metrics from PushGateway"
echo "5. Connect external Dozzle to thesis-server:7007"
echo "6. Connect external Portainer to thesis-server:9001"

echo ""
echo "🎓 Ready for thesis experiments!"