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

# First-time stack setup (secrets, Garage config) - run once
setup:
    bash scripts/ml-setup.sh

# --- development ---

# Run the quantum pipeline simulation
run *ARGS:
    python -m quantum_pipeline {{ARGS}}

# Start IPython shell with quantum_pipeline imported
shell:
    python -c "from quantum_pipeline import *; import IPython; IPython.embed()"

# --- testing ---

# Run tests: just test | just test ml | just test integration | just test cov
#             just test quick | just test debug [path] | just test <path>
test *ARGS:
    #!/usr/bin/env bash
    set -- {{ARGS}}
    case "${1:-}" in
        ml)
            shift
            echo "[ INFO ] ML tests (sequential)..."
            pdm run pytest tests/ml/ "$@" -v --timeout=300
            ;;
        integration)
            shift
            echo "[ INFO ] Integration tests (sequential, requires Docker)..."
            pdm run pytest tests/integration/ "$@" -v -m integration --timeout=300
            ;;
        cov)
            shift
            echo "[ INFO ] Unit tests with coverage..."
            pdm run coverage erase
            pdm run pytest tests/ "$@" -n auto -m "not integration and not slow" --cov=quantum_pipeline --cov-report= -v
            pdm run coverage combine 2>/dev/null || true
            pdm run coverage report --show-missing
            pdm run coverage html -d htmlcov
            echo "[  OK  ] Coverage report: htmlcov/index.html"
            ;;
        quick)
            shift
            pdm run pytest tests/ "$@" -x --tb=short -q -n auto -m "not integration and not slow"
            ;;
        debug)
            shift
            if [ $# -eq 0 ] || [[ "$1" == -* ]]; then
                pdm run pytest tests/ "$@" -vv -s --tb=long --timeout=0
            else
                pdm run pytest "$@" -vv -s --tb=long --timeout=0
            fi
            ;;
        ""|-*)
            pdm run pytest tests/ "$@" -v -n auto -m "not integration and not slow"
            ;;
        *)
            pdm run pytest "$@" -v -n auto
            ;;
    esac
    echo "[  OK  ] Tests passed"

# Watch tests and rerun on changes
test-watch:
    pdm run ptw tests/ -- -v -n auto -m "not integration and not slow"

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

# Build Docker image (target: cpu, gpu, all)
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
        all)
            just docker-build cpu
            just docker-build gpu
            ;;
        *)
            echo "[ FAIL ] Unknown target: {{TARGET}} (expected: cpu, gpu, all)"
            exit 1
            ;;
    esac

# Start the compose stack (requires .env - run 'just setup' first)
up *ARGS:
    #!/usr/bin/env bash
    [ ! -f .env ] && echo "[ FAIL ] .env not found - run 'just setup' first" && exit 1
    docker compose --env-file .env -f compose/docker-compose.ml.yaml up -d {{ARGS}}
    echo "[  OK  ] Stack started"
    docker compose --env-file .env -f compose/docker-compose.ml.yaml ps

# Stop the compose stack (including batch containers and state)
down:
    #!/usr/bin/env bash
    docker rm -f $(docker ps -q --filter "name=ml-quantum-pipeline" 2>/dev/null) 2>/dev/null || true
    docker compose --env-file .env -f compose/docker-compose.ml.yaml --profile batch down
    rm -f gen/ml_batch_state.json
    echo "[  OK  ] Stack stopped, batch state cleared"

# Tail compose logs (optional: service name)
logs SERVICE="":
    #!/usr/bin/env bash
    if [ -z "{{SERVICE}}" ]; then
        docker compose --env-file .env -f compose/docker-compose.ml.yaml logs -f
    else
        docker compose --env-file .env -f compose/docker-compose.ml.yaml logs -f {{SERVICE}}
    fi

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
    rm -rf .coverage .coverage.* htmlcov/ .mypy_cache/ build/ dist/ *.egg-info gen/ run_configs/
    echo "[  OK  ] Cleaned"

# Generate requirements.txt from pdm.lock
requirements:
    pdm export -o requirements.txt --no-hashes
    pdm export -d -o requirements-dev.txt --no-hashes
    echo "[  OK  ] Requirements files updated"

# --- help ---

# Show help
help:
    @echo "Quantum Pipeline Justfile"
    @echo ""
    @echo "Setup:"
    @echo "  just install          Install dev + docs dependencies"
    @echo "  just install-core     Install core only (no dev tools)"
    @echo "  just setup            First-time stack setup (run once)"
    @echo ""
    @echo "Development:"
    @echo "  just run [ARGS]       Run quantum-pipeline with arguments"
    @echo ""
    @echo "Testing:"
    @echo "  just test               Run unit tests (parallel)"
    @echo "  just test ml            Run ML tests (sequential)"
    @echo "  just test integration   Run integration tests (Docker)"
    @echo "  just test cov           Unit tests with coverage"
    @echo "  just test quick         Fail-fast smoke test"
    @echo "  just test debug [path]  Verbose, no capture, no timeout"
    @echo "  just test <path>        Run tests at specific path"
    @echo "  just test-watch         Rerun on file changes"
    @echo ""
    @echo "Code Quality:"
    @echo "  just fmt              Format + lint fix (ruff)"
    @echo "  just quality          Lint + type-check + format-check"
    @echo ""
    @echo "Docker:"
    @echo "  just docker-build         Build CPU image (default)"
    @echo "  just docker-build gpu     Build GPU image"
    @echo "  just docker-build all     Build CPU + GPU images"
    @echo ""
    @echo "Stack:"
    @echo "  just setup            First-time setup (secrets, Garage)"
    @echo "  just up               Start the compose stack"
    @echo "  just down             Stop the compose stack"
    @echo "  just logs [service]   Tail compose logs"
    @echo ""
    @echo "Docs:"
    @echo "  just docs-build       Build mkdocs site"
    @echo "  just docs-serve       Serve with live reload"
    @echo ""
    @echo "Other:"
    @echo "  just build            Build wheel + sdist"
    @echo "  just clean            Remove caches and artifacts"
    @echo "  just requirements     Export requirements.txt"
