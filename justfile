set shell := ["bash", "-euo", "pipefail", "-c"]

default:
    @just --list

# --- setup ---

# Install all dev dependencies (dev + docs)
install:
    pdm install -G dev -G docs
    echo "[  OK  ] Installed ($(pdm --version), $(python --version 2>&1))"

# Install core dependencies only (no dev tools)
install-core:
    pdm install -G core
    echo "[  OK  ] Core dependencies installed"

# --- development ---

# Run the quantum pipeline simulation
run *ARGS:
    pdm run python quantum_pipeline.py {{ARGS}}

# Start IPython shell with quantum_pipeline imported
shell:
    pdm run python -c "from quantum_pipeline import *; import IPython; IPython.embed()"

# --- testing ---

# Run tests - pass a path for targeting: just test tests/solvers
test *ARGS:
    #!/usr/bin/env bash
    set -- {{ARGS}}
    if [ $# -eq 0 ] || [[ "$1" == -* ]]; then
        pdm run pytest tests/ "$@" -v
    else
        pdm run pytest "$@" -v
    fi
    echo "[  OK  ] Tests passed"

# Run tests with coverage
test-cov *ARGS:
    #!/usr/bin/env bash
    set -- {{ARGS}}
    if [ $# -eq 0 ] || [[ "$1" == -* ]]; then
        pdm run pytest tests/ "$@" --cov=quantum_pipeline --cov-report=term-missing --cov-report=html:htmlcov -v
    else
        pdm run pytest "$@" --cov=quantum_pipeline --cov-report=term-missing --cov-report=html:htmlcov -v
    fi
    echo "[  OK  ] Coverage report: htmlcov/index.html"

# Quick smoke test (fail-fast, minimal output)
test-quick *ARGS:
    pdm run pytest tests/ {{ARGS}} -x --tb=short -q

# Debug tests (verbose, no capture, long tracebacks)
test-debug *ARGS:
    #!/usr/bin/env bash
    set -- {{ARGS}}
    if [ $# -eq 0 ] || [[ "$1" == -* ]]; then
        pdm run pytest tests/ "$@" -vv -s --tb=long
    else
        pdm run pytest "$@" -vv -s --tb=long
    fi

# Watch tests and rerun on changes
test-watch:
    pdm run ptw tests/ -- -v

# --- code quality ---

# Format and fix lint (ruff format + check --fix)
fmt:
    pdm run ruff format .
    pdm run ruff check . --fix
    echo "[  OK  ] Formatted and lint-fixed"

# Run all quality checks (lint + type-check + format-check)
quality:
    #!/usr/bin/env bash
    echo "[ INFO ] Ruff lint..."
    pdm run ruff check .
    echo "[ INFO ] Mypy type-check..."
    pdm run mypy quantum_pipeline/
    echo "[ INFO ] Ruff format check..."
    pdm run ruff format --check .
    echo "[  OK  ] All quality checks passed"

# --- documentation ---

# Build documentation
docs-build:
    pdm run mkdocs build
    echo "[  OK  ] Docs built to ./site/"

# Serve documentation with live reload
docs-serve:
    @echo "[ INFO ] Serving docs on http://localhost:8000"
    pdm run mkdocs serve

# --- docker ---

# Build Docker image (target: cpu, gpu)
# For GPU: set CUDA_ARCH env var to match your GPU (default: 8.6/Ampere)
#   CUDA_ARCH=6.1 just docker-build gpu   # Pascal (GTX 10xx)
#   CUDA_ARCH=7.5 just docker-build gpu   # Turing (RTX 20xx)
#   CUDA_ARCH=8.9 just docker-build gpu   # Ada Lovelace (RTX 40xx)
docker-build TARGET="cpu":
    #!/usr/bin/env bash
    VERSION=$(python -c "import quantum_pipeline; print(quantum_pipeline.__version__)")
    case "{{TARGET}}" in
        cpu)
            echo "[ INFO ] Building CPU image..."
            docker build -f docker/Dockerfile.cpu \
                -t quantum-pipeline:cpu \
                -t straightchlorine/quantum-pipeline:cpu \
                -t straightchlorine/quantum-pipeline:cpu-${VERSION} \
                .
            echo "[  OK  ] CPU image built (v${VERSION})"
            ;;
        gpu)
            ARCH="${CUDA_ARCH:-8.6}"
            echo "[ INFO ] Building GPU image (CUDA_ARCH=${ARCH})..."
            docker build -f docker/Dockerfile.gpu \
                --build-arg CUDA_ARCH="${ARCH}" \
                -t quantum-pipeline:gpu \
                -t straightchlorine/quantum-pipeline:gpu \
                -t straightchlorine/quantum-pipeline:gpu-${VERSION} \
                .
            echo "[  OK  ] GPU image built (v${VERSION}, CUDA_ARCH=${ARCH})"
            ;;
        *)
            echo "[ FAIL ] Unknown target: {{TARGET}} (expected: cpu, gpu)"
            exit 1
            ;;
    esac

# Start docker compose stack
docker-up *ARGS:
    docker compose -f compose/docker-compose.yaml up -d {{ARGS}}
    echo "[  OK  ] Stack started"
    docker compose -f compose/docker-compose.yaml ps

# Build and start docker compose stack
docker-up-build:
    docker compose -f compose/docker-compose.yaml up -d --build
    echo "[  OK  ] Stack built and started"
    docker compose -f compose/docker-compose.yaml ps

# Stop docker compose stack
docker-down:
    docker compose -f compose/docker-compose.yaml down
    echo "[  OK  ] Stack stopped"

# Tail docker compose logs (optional: service name)
docker-logs SERVICE="":
    #!/usr/bin/env bash
    if [ -z "{{SERVICE}}" ]; then
        docker compose -f compose/docker-compose.yaml logs -f
    else
        docker compose -f compose/docker-compose.yaml logs -f {{SERVICE}}
    fi

# --- ml pipeline ---

# First-time setup (secrets, Garage config) - run once
ml-setup:
    bash scripts/ml-setup.sh

# Start the ML pipeline stack
ml-up:
    #!/usr/bin/env bash
    [ ! -f .env ] && echo "[ FAIL ] .env not found - run 'just ml-setup' first" && exit 1
    docker compose -f compose/docker-compose.ml.yaml up -d
    echo "[  OK  ] ML stack started"
    docker compose -f compose/docker-compose.ml.yaml ps

# Stop the ML pipeline stack
ml-down:
    docker compose -f compose/docker-compose.ml.yaml down
    echo "[  OK  ] ML stack stopped"

# End-to-end smoke test (VQE → Kafka → Garage → Spark)
ml-smoke-test:
    bash scripts/smoke_test_ml_pipeline.sh

# --- build ---

# Build distribution packages (wheel + sdist)
build:
    pdm build
    echo "[  OK  ] Built to dist/"
    @ls -lh dist/

# --- utilities ---

# Clean all generated files and caches
clean:
    #!/usr/bin/env bash
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
    rm -rf .coverage htmlcov/ .mypy_cache/ build/ dist/ *.egg-info gen/ run_configs/
    echo "[  OK  ] Cleaned"

# Generate requirements.txt from pdm.lock
requirements:
    pdm export -o requirements.txt --no-hashes
    pdm export -d -o requirements-dev.txt --no-hashes
    echo "[  OK  ] Requirements files updated"

# Run with CPU profiler
profile:
    pdm run python -m cProfile -o profile.stats quantum_pipeline.py
    echo "[  OK  ] Profile saved to profile.stats"

# Run with memory profiler
memprofile:
    pdm run python -m memory_profiler quantum_pipeline.py

# --- help ---

# Show help
help:
    @echo "Quantum Pipeline Justfile"
    @echo ""
    @echo "Setup:"
    @echo "  just install          Install dev + docs dependencies"
    @echo "  just install-core     Install core only (no dev tools)"
    @echo ""
    @echo "Testing (pass any path for targeting, e.g. tests/solvers):"
    @echo "  just test             Run all tests"
    @echo "  just test <path>      Run tests at path"
    @echo "  just test-cov         Run with coverage"
    @echo "  just test-quick       Fail-fast smoke test"
    @echo "  just test-debug       Verbose with full tracebacks"
    @echo "  just test-watch       Rerun on file changes"
    @echo ""
    @echo "Code Quality:"
    @echo "  just fmt              Format + lint fix (ruff)"
    @echo "  just quality          Lint + type-check + format-check"
    @echo ""
    @echo "Docker:"
    @echo "  just docker-build         Build CPU image (default)"
    @echo "  just docker-build gpu     Build GPU image"
    @echo "  just docker-up            Start compose stack"
    @echo "  just docker-down          Stop compose stack"
    @echo "  just docker-logs [svc]    Tail logs"
    @echo ""
    @echo "ML Pipeline:"
    @echo "  just ml-setup         First-time setup (run once)"
    @echo "  just ml-up / ml-down  Start / stop ML stack"
    @echo "  just ml-smoke-test    End-to-end smoke test"
    @echo ""
    @echo "Docs:"
    @echo "  just docs-build       Build mkdocs site"
    @echo "  just docs-serve       Serve with live reload"
    @echo ""
    @echo "Other:"
    @echo "  just build            Build wheel + sdist"
    @echo "  just clean            Remove caches and artifacts"
    @echo "  just requirements     Export requirements.txt"
    @echo "  just profile          CPU profiler"
    @echo "  just memprofile       Memory profiler"
