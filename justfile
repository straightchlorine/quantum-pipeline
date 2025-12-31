# Quantum Pipeline Universal Justfile
# Development, testing, and deployment automation
# Run 'just' or 'just --list' to see all available commands

default:
    @just --list

# ============================================================================
# SETUP AND INSTALLATION
# ============================================================================

# Install all dependencies (core + dev + docs + airflow)
install:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Installing quantum-pipeline with all optional dependencies..."
    echo "Using PDM package manager"
    pdm install -G dev -G docs -G airflow
    echo ""
    echo "[SUCCESS] All dependencies installed successfully"
    pdm --version
    python --version

# Install only core dependencies (production)
install-core:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Installing core production dependencies only..."
    pdm install --no-dev
    echo "[SUCCESS] Core dependencies installed"

# Install with development tools
install-dev:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Installing with development tools..."
    pdm install -G dev
    echo "[SUCCESS] Development dependencies installed"

# Install with documentation tools
install-docs:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Installing documentation generation tools..."
    pdm install -G docs
    echo "[SUCCESS] Documentation dependencies installed"

# Install with Apache Airflow orchestration
install-airflow:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Installing with Apache Airflow orchestration..."
    pdm install -G airflow
    echo "[SUCCESS] Airflow dependencies installed"

# Install development + documentation (for CI/CD)
install-full-dev:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Installing with all development and documentation tools..."
    pdm install -G dev -G docs
    echo "[SUCCESS] Full development environment installed"

# ============================================================================
# DEVELOPMENT COMMANDS
# ============================================================================

# Run the quantum pipeline simulation (main entry point)
run *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Running quantum pipeline simulation..."
    pdm run python quantum_pipeline.py {{ARGS}}

# Start interactive Python shell with quantum_pipeline imported
shell:
    #!/usr/bin/env bash
    set -euo pipefail
    pdm run python -c "from quantum_pipeline import *; import IPython; IPython.embed()"

# ============================================================================
# TESTING AND COVERAGE
# ============================================================================

# Run all tests with pytest
test *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Running pytest test suite..."
    pdm run pytest tests/ {{ARGS}} -v
    echo ""
    echo "✓ Tests completed"

# Run tests with coverage report
test-coverage *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Running tests with coverage analysis..."
    pdm run pytest tests/ {{ARGS}} \
        --cov=quantum_pipeline \
        --cov-report=term-missing \
        --cov-report=html:htmlcov \
        -v
    echo ""
    echo "✓ Coverage report generated in htmlcov/index.html"

# Run tests matching a pattern
test-match PATTERN *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Running tests matching pattern: {{PATTERN}}"
    pdm run pytest tests/ -k "{{PATTERN}}" {{ARGS}} -v

# Run specific test file
test-file FILE *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Running test file: {{FILE}}"
    pdm run pytest "tests/{{FILE}}" {{ARGS}} -v

# Run tests in specific module
test-module MODULE *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Running tests for module: {{MODULE}}"
    pdm run pytest "tests/{{MODULE}}/" {{ARGS}} -v

# Run tests with detailed output and debugging
test-debug *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Running tests in debug mode..."
    pdm run pytest tests/ {{ARGS}} -vv -s --tb=long

# Run integration tests only
test-integration *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Running integration tests..."
    pdm run pytest tests/integration/ {{ARGS}} -v

# Run unit tests only
test-unit *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Running unit tests..."
    pdm run pytest tests/ {{ARGS}} \
        --ignore=tests/integration/ \
        -v

# Run a quick test smoke-check (fail-fast)
test-quick *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Running quick smoke-check tests..."
    pdm run pytest tests/ {{ARGS}} \
        -x \
        --tb=short \
        -q

# Watch tests and rerun on file changes (requires pytest-watch)
test-watch:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Starting test watcher..."
    echo "Tests will rerun automatically when files change"
    pdm run ptw tests/ -- -v

# ============================================================================
# CODE QUALITY AND FORMATTING
# ============================================================================

# Format code with ruff and black
format:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Formatting code..."
    echo "  Running ruff formatter..."
    pdm run ruff format quantum_pipeline/ tests/
    echo "  Running black formatter..."
    pdm run black quantum_pipeline/ tests/ --line-length=99
    echo ""
    echo "✓ Code formatted successfully"

# Run linting checks (ruff + flake8)
lint:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Running linting checks..."
    echo ""
    echo "  Ruff linting..."
    pdm run ruff check quantum_pipeline/ tests/ --show-fixes
    echo ""
    echo "  Flake8 linting..."
    pdm run flake8 quantum_pipeline/ tests/
    echo ""
    echo "✓ Linting checks complete"

# Run type checking with mypy
type-check:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Running type checking with mypy..."
    pdm run mypy quantum_pipeline/
    echo ""
    echo "✓ Type checking complete"

# Run all quality checks (lint, type-check, format-check)
quality:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Running comprehensive quality checks..."
    echo ""

    echo "1. Ruff linting..."
    pdm run ruff check quantum_pipeline/ tests/

    echo ""
    echo "2. Flake8 linting..."
    pdm run flake8 quantum_pipeline/ tests/

    echo ""
    echo "3. Type checking with mypy..."
    pdm run mypy quantum_pipeline/

    echo ""
    echo "4. Checking code formatting with ruff..."
    pdm run ruff format --check quantum_pipeline/ tests/

    echo ""
    echo "4. Checking code formatting with black..."
    pdm run black --check quantum_pipeline/ tests/ --line-length=99

    echo ""
    echo "✓ All quality checks passed!"

# ============================================================================
# TESTING + QUALITY COMBO
# ============================================================================

# Run all tests with coverage + quality checks
check:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Running comprehensive checks: tests + quality"
    echo ""

    echo "1. Running tests with coverage..."
    pdm run pytest tests/ \
        --cov=quantum_pipeline \
        --cov-report=term-missing \
        -v

    echo ""
    echo "2. Running linting checks..."
    pdm run ruff check quantum_pipeline/ tests/
    pdm run flake8 quantum_pipeline/ tests/

    echo ""
    echo "3. Running type checks..."
    pdm run mypy quantum_pipeline/

    echo ""
    echo "✓ All checks passed!"

# Quick check (fast quality + tests)
check-quick:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Running quick checks..."
    pdm run pytest tests/ -q --tb=short
    pdm run ruff check quantum_pipeline/ tests/ --quiet
    echo "✓ Quick checks passed!"

# ============================================================================
# DOCUMENTATION
# ============================================================================

# Build documentation with mkdocs
docs-build:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Building documentation..."
    pdm run mkdocs build
    echo ""
    echo "✓ Documentation built to ./site/"

# Serve documentation locally with live reload
docs-serve:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Serving documentation on http://localhost:8000"
    echo "Press Ctrl+C to stop"
    pdm run mkdocs serve

# Deploy documentation to GitHub Pages
docs-deploy:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Deploying documentation to GitHub Pages..."
    pdm run mkdocs gh-deploy --force
    echo "✓ Documentation deployed"

# ============================================================================
# DOCKER AND CONTAINERIZATION
# ============================================================================

# Build CPU-only Docker image
docker-build-cpu:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Building CPU-only Docker image..."
    docker build -f docker/Dockerfile.cpu -t quantum-pipeline:latest .
    echo "✓ CPU image built: quantum-pipeline:latest"

# Build GPU Docker image
docker-build-gpu:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Building GPU Docker image..."
    docker build -f docker/Dockerfile.gpu -t quantum-pipeline:gpu .
    echo "✓ GPU image built: quantum-pipeline:gpu"

# Build Spark integration Docker image
docker-build-spark:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Building Spark integration Docker image..."
    docker build -f docker/Dockerfile.spark -t quantum-pipeline:spark .
    echo "✓ Spark image built: quantum-pipeline:spark"

# Build Airflow integration Docker image
docker-build-airflow:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Building Airflow integration Docker image..."
    docker build -f docker/Dockerfile.airflow -t quantum-pipeline:airflow .
    echo "✓ Airflow image built: quantum-pipeline:airflow"

# Build all Docker images
docker-build-all: docker-build-cpu docker-build-gpu docker-build-spark docker-build-airflow
    #!/usr/bin/env bash
    echo "✓ All Docker images built successfully"

# Run full Docker Compose stack (development)
docker-up:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Starting Docker Compose services..."
    echo "Services: Quantum Pipeline, Kafka, MinIO, Spark, Prometheus, Grafana"
    echo ""
    docker compose up -d
    echo ""
    echo "✓ Services started"
    echo ""
    docker compose ps

# Run Docker Compose with build
docker-up-build:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Building and starting Docker Compose services..."
    docker compose up -d --build
    echo "✓ Services built and started"
    docker compose ps

# Stop Docker Compose services
docker-down:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Stopping Docker Compose services..."
    docker compose down
    echo "✓ Services stopped"

# Stop and remove volumes (WARNING: deletes data)
docker-clean:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "⚠ Removing Docker Compose services and volumes..."
    docker compose down -v
    echo "✓ Cleanup complete"

# View Docker Compose logs
docker-logs SERVICE="":
    #!/usr/bin/env bash
    if [ -z "{{SERVICE}}" ]; then
        docker compose logs -f
    else
        docker compose logs -f {{SERVICE}}
    fi

# Show Docker Compose service status
docker-status:
    #!/usr/bin/env bash
    docker compose ps

# ============================================================================
# BUILD AND DISTRIBUTION
# ============================================================================

# Build distribution packages (wheel + sdist)
build:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Building distribution packages..."
    pdm build
    echo ""
    echo "✓ Distributions built to dist/"
    ls -lh dist/

# Clean build artifacts
build-clean:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Cleaning build artifacts..."
    rm -rf build/ dist/ *.egg-info
    echo "✓ Build artifacts cleaned"

# ============================================================================
# DEVELOPMENT UTILITIES
# ============================================================================

# Clean all generated files and caches
clean:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Cleaning generated files and caches..."

    echo "  Removing pytest cache..."
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true

    echo "  Removing coverage data..."
    rm -rf .coverage htmlcov/ .mypy_cache/

    echo "  Removing build artifacts..."
    rm -rf build/ dist/ *.egg-info

    echo "  Removing generated outputs..."
    rm -rf gen/ run_configs/

    echo "✓ Cleanup complete"

# Reset development environment (dangerous!)
clean-all: clean docker-clean
    #!/usr/bin/env bash
    set -euo pipefail
    echo "⚠ Removing all generated files, caches, and Docker data..."
    pdm venv remove in-project --force 2>/dev/null || true
    echo "✓ Full reset complete. Run 'just install' to rebuild"

# Show project statistics
stats:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Project Statistics"
    echo "=================="
    echo ""
    echo "Source Code:"
    find quantum_pipeline -name "*.py" | wc -l | xargs echo "  Python files:"
    find quantum_pipeline -name "*.py" -exec wc -l {} + | tail -1 | awk '{print "  Lines of code: " $1}'
    echo ""
    echo "Tests:"
    find tests -name "test_*.py" | wc -l | xargs echo "  Test files:"
    find tests -name "*.py" -exec wc -l {} + | tail -1 | awk '{print "  Lines of test code: " $1}'
    echo ""
    echo "Coverage Summary:"
    pdm run pytest tests/ --cov=quantum_pipeline --cov-report=term-missing -q 2>/dev/null | tail -10 || echo "  (Run 'just test-coverage' to generate)"

# Generate requirements file from pdm.lock
requirements-freeze:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Generating requirements files from pdm.lock..."
    pdm export -o requirements.txt --no-hashes
    pdm export -d -o requirements-dev.txt --no-hashes
    echo "✓ Requirements files updated"

# ============================================================================
# MONITORING AND PROFILING
# ============================================================================

# Run with performance profiling
profile:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Running with performance profiling..."
    pdm run python -m cProfile -o profile.stats quantum_pipeline.py
    echo "✓ Profile saved to profile.stats"
    echo "View with: python -m pstats profile.stats"

# Run with memory profiling (requires memory-profiler)
memprofile:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Running with memory profiling..."
    pdm run python -m memory_profiler quantum_pipeline.py
    echo "✓ Memory profile complete"

# ============================================================================
# GIT AND VERSIONING
# ============================================================================

# Show current version
version:
    #!/usr/bin/env bash
    python -c "import quantum_pipeline; print(f'quantum-pipeline v{quantum_pipeline.__version__}')"

# Create git tag for release
tag VERSION:
    #!/usr/bin/env bash
    set -euo pipefail
    git tag -a "v{{VERSION}}" -m "Release v{{VERSION}}"
    echo "✓ Tag created: v{{VERSION}}"
    echo "Push with: git push origin v{{VERSION}}"

# ============================================================================
# HELP AND INFORMATION
# ============================================================================

# Show detailed help for all commands
help:
    @echo "Quantum Pipeline - Universal Justfile"
    @echo ""
    @echo "Available command categories:"
    @echo ""
    @echo "Setup & Installation:"
    @echo "  just install              - Install all dependencies"
    @echo "  just install-core         - Install production only"
    @echo "  just install-airflow      - Install with Airflow"
    @echo "  just install-docs         - Install docs tools"
    @echo ""
    @echo "Testing:"
    @echo "  just test                 - Run all tests"
    @echo "  just test-coverage        - Run tests with coverage report"
    @echo "  just test-quick           - Run quick smoke-check"
    @echo "  just test-module MODULE   - Run tests for specific module"
    @echo ""
    @echo "Code Quality:"
    @echo "  just format               - Format code automatically"
    @echo "  just lint                 - Run linting checks"
    @echo "  just type-check           - Run type checking"
    @echo "  just quality              - Run all quality checks"
    @echo "  just check                - Tests + quality checks"
    @echo ""
    @echo "Docker:"
    @echo "  just docker-build-cpu     - Build CPU image"
    @echo "  just docker-up            - Start Docker Compose"
    @echo "  just docker-down          - Stop Docker Compose"
    @echo ""
    @echo "Documentation:"
    @echo "  just docs-build           - Build docs"
    @echo "  just docs-serve           - Serve docs locally"
    @echo ""
    @echo "Run 'just --list' for complete command list"
