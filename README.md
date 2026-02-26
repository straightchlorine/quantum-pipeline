# Quantum Pipeline


<div align="center">

**Repository:** [GitHub](https://github.com/straightchlorine/quantum-pipeline) (primary) · [Codeberg](https://codeberg.org/piotrkrzysztof/quantum-pipeline) (mirror)

[![PyPI version](https://badge.fury.io/py/quantum-pipeline.svg)](https://pypi.org/project/quantum-pipeline/)
[![Total Downloads](https://static.pepy.tech/badge/quantum-pipeline)](https://pepy.tech/project/quantum-pipeline)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/quantum-pipeline)](https://pypi.org/project/quantum-pipeline/)
[![Docker Pulls](https://img.shields.io/docker/pulls/straightchlorine/quantum-pipeline.svg)](https://hub.docker.com/r/straightchlorine/quantum-pipeline)
</div>

## Overview

The Quantum Pipeline project is an extensible framework designed for exploring Variational Quantum Eigensolver (VQE) algorithms. It combines quantum and classical computing to estimate the ground-state energy of molecular systems with a comprehensive data engineering pipeline.

The framework provides modules to handle algorithm orchestration, parametrising it as well as monitoring and data visualization. Data is organised in extensible dataclasses, which can be streamed via Kafka for real-time processing, transformed into ML features using Apache Spark, and stored in Apache Iceberg tables for scalable analytics.

Currently, it offers VQE as its primary algorithm with production-grade data processing capabilities, including automated workflow orchestration via Apache Airflow, and aims to evolve into a convenient platform for running various quantum algorithms at scale.

---

## Features

### Core Quantum Computing
- **Molecule Loading:** Load and validate molecular data from files.
- **Hamiltonian Preparation:** Generate second-quantized Hamiltonians for molecular systems.
- **Quantum Circuit Construction:** Create parameterized ansatz circuits with customizable repetitions.
- **VQE Execution:** Solve Hamiltonians using the VQE algorithm with support for various optimizers.
- **Advanced Backend Options:** Customize simulation parameters such as qubit count, shot count, and optimization levels.

### Data Engineering Pipeline
- **Real-time Streaming:** Stream simulation results to Apache Kafka with Avro serialization for real-time data processing.
- **ML Feature Engineering:** Transform quantum experiment data into ML features using Apache Spark with incremental processing.
- **Data Lake Storage:** Store processed data in Apache Iceberg tables with versioning and time-travel capabilities.
- **Object Storage:** Persist data using MinIO S3-compatible storage with automated backup and retention.
- **Workflow Orchestration:** Automate data processing workflows using Apache Airflow with monitoring and alerting.

### Analytics and Visualization
- **Visualization Tools:** Plot molecular structures, energy convergence, and operator coefficients.
- **Report Generation:** Automatically generate detailed reports for each processed molecule.
- **Scientific Reference Validation:** Compare VQE results against experimentally verified and high-level theoretical ground state energies from peer-reviewed literature for 8+ molecules (H₂, HeH⁺, LiH, BeH₂, H₂O, NH₃, CO₂, N₂) with accuracy metrics and chemical accuracy tracking.
- **Feature Tables:** Access structured data through 9 specialized ML feature tables (molecules, iterations, parameters, etc.).
- **Processing Metadata:** Track data lineage and processing history with comprehensive metadata management.

### Production Deployment
- **Containerized Execution:** Deploy as multi-service Docker containers with GPU support.
- **CI/CD Pipeline:** Automated testing, building, and publishing of Docker images via GitHub Actions.
- **Scalable Architecture:** Distributed processing with Spark clusters and horizontal scaling capabilities.
- **Security:** Comprehensive secrets management and secure communication between services.

---

## Directory Structure

```
quantum_pipeline/
├── configs/              # Configuration settings and argument parsers
├── drivers/              # Molecule loading and basis set validation
├── features/             # Quantum circuit and Hamiltonian features
├── mappers/              # Fermionic-to-qubit mapping implementations
├── monitoring/           # Performance monitoring (Prometheus/Grafana integration)
├── report/               # Report generation utilities
├── runners/              # VQE execution logic
├── solvers/              # VQE solver implementations
├── stream/               # Kafka streaming and messaging utilities
├── structures/           # Quantum and classical data structures
├── utils/                # Utility functions (logging, visualization, etc.)
├── visual/               # Visualization tools for molecules and operators
├── docker/               # Docker configurations and deployment files (see docker/README.md)
│   ├── airflow/          # Airflow DAGs and Spark processing scripts
│   ├── connectors/       # Kafka Connect configurations
│   ├── Dockerfile.cpu    # CPU-optimized container
│   ├── Dockerfile.gpu    # GPU-accelerated container
│   ├── Dockerfile.spark  # Spark cluster container
│   └── Dockerfile.airflow # Airflow services container
├── notebooks/            # Jupyter notebooks for data analysis and exploration
├── .github/              # CI/CD workflows and automation
└── quantum_pipeline.py   # Main entry point
```

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/quantum_pipeline.git
   cd quantum_pipeline
   ```

2. **Set Up a Virtual Environment** (optional but recommended):
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Deploy Full Platform with Docker Compose**:
   ```bash
   docker-compose up --build
   ```

   This launches the complete data platform including:
   - Quantum Pipeline (CPU/GPU)
   - Apache Kafka with Schema Registry
   - Apache Spark cluster (master + workers)
   - Apache Airflow (webserver, scheduler, triggerer)
   - MinIO object storage
   - PostgreSQL database
   - Prometheus & Grafana monitoring (optional)

   **For detailed Docker configuration, environment variables, and troubleshooting, see [docker/README.md](docker/README.md).**

5. **(Alternative) Build Individual Containers**:

   Available Dockerfiles:
   - `docker/Dockerfile.cpu` - CPU-optimized quantum simulation
   - `docker/Dockerfile.gpu` - GPU-accelerated with CUDA support (requires NVIDIA Docker)
   - `docker/Dockerfile.spark` - Apache Spark cluster nodes
   - `docker/Dockerfile.airflow` - Apache Airflow workflow orchestration

   ```bash
   # CPU-optimized container
   docker build -f docker/Dockerfile.cpu -t quantum-pipeline:cpu .

   # GPU-accelerated container (requires NVIDIA Docker)
   docker build -f docker/Dockerfile.gpu -t quantum-pipeline:gpu .
   ```

6. **(Production) Use Pre-built Images**:
   Docker images are automatically built and published via GitHub Actions:
   ```bash
   # Latest stable release
   docker pull straightchlorine/quantum-pipeline:latest

   # GPU-enabled version
   docker pull straightchlorine/quantum-pipeline:latest-gpu
   ```

---

## Usage

### 1. Prepare Input Data

Molecules should be defined like this:

```json
[
    {
        "symbols": ["H", "H"],
        "coords": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        "multiplicity": 1,
        "charge": 0,
        "units": "angstrom",
        "masses": [1.008, 1.008]
    },
    {
        "symbols": ["O", "H", "H"],
        "coords": [[0.0, 0.0, 0.0], [0.0, 0.757, 0.586], [0.0, -0.757, 0.586]],
        "multiplicity": 1,
        "charge": 0,
        "units": "angstrom",
        "masses": [15.999, 1.008, 1.008]
    }
]
```

### 2. Run the Pipeline

Run the main script to process molecules:

```bash
python quantum_pipeline.py -f data/molecule.json -b sto-3g --max-iterations 100 --optimizer COBYLA --report
```

Defaults for each option can be found in `configs/defaults.py` and the help message (`python quantum_pipeline.py -h`). Other available parameters include:

**Core Parameters:**
- `-f FILE, --file FILE`: Path to the molecule data file (required).
- `-b BASIS, --basis BASIS`: Specify the basis set for the simulation (sto-3g, 6-31g, cc-pvdz).
- `--ibm`: Use IBM Quantum backend (default is local simulator).
- `--min-qubits MIN_QUBITS`: Specify the minimum number of qubits required.
- `--max-iterations MAX_ITERATIONS`: Set the maximum number of VQE iterations (mutually exclusive with `--convergence`).
- `--convergence THRESHOLD`: Enable convergence-based optimization with specified threshold (mutually exclusive with `--max-iterations`).
- `--optimizer OPTIMIZER`: Choose from 16+ optimization algorithms (see Optimizer Configuration section).
- `-ar, --ansatz-reps REPS`: Number of ansatz repetitions (default: 3).
- `--output-dir OUTPUT_DIR`: Specify the directory for storing output files.
- `--log-level {DEBUG,INFO,WARNING,ERROR}`: Set the logging level.

**Simulation Configuration:**
- `--simulation-method METHOD`: Choose backend method (automatic, statevector, density_matrix, stabilizer, extended_stabilizer, matrix_product_state, unitary, superop, tensor_network).
- `--shots SHOTS`: Number of shots for quantum circuit execution.
- `--optimization-level {0,1,2,3}`: Circuit optimization level.
- `--noise MODEL`: Choose noise model for simulation.
- `--gpu`: Enable GPU acceleration (requires CUDA-enabled backend).

**Data & Reporting:**
- `--report`: Generate a PDF report after simulation.
- `--kafka`: Stream data to Apache Kafka for real-time processing.
- `--dump FILE`: Save current configuration to JSON file.
- `--load FILE`: Load configuration from JSON file.

**Performance Monitoring:**
- `--enable-performance-monitoring`: Enable comprehensive performance metrics collection.
- `--performance-interval SECONDS`: Metrics collection interval (default: 30s).
- `--performance-pushgateway URL`: Prometheus PushGateway URL for metrics export.
- `--performance-export-format {json,prometheus,both}`: Metrics export format.

### Example Configurations

Basic configuration (utilizes the `defaults.py` config) emphasizes performance over accuracy:
```bash
python quantum_pipeline.py -f data/molecules.json
```

Configuration with custom parameters:
```bash
python quantum_pipeline.py -f data/molecule.json -b cc-pvdz --max-iterations 200 --optimizer L-BFGS-B --shots 2048 --report
```

### 3. Data Platform Integration

**Kafka Streaming**: Enable real-time streaming to Apache Kafka with Avro serialization:
```bash
python quantum_pipeline.py -f data/molecule.json --kafka
```

Data is serialized using **Avro** format with Schema Registry integration for type-safe messaging:
- Binary Avro encoding for efficient transmission
- Automatic numpy type conversion (float64 → double, int64 → long)
- Schema versioning and evolution support
- Deserialization interface: `quantum_pipeline.stream.serialization.interfaces.vqe`

**Full Platform Deployment**: Launch with complete data processing pipeline:
```bash
# Start all services
docker-compose up -d

# Run quantum pipeline with data streaming
docker-compose exec quantum-pipeline python quantum_pipeline.py -f data/molecules.json --kafka --gpu
```

**Airflow Orchestration**: Access the Airflow web interface at `http://airflow:8084` to:
- Monitor automated daily processing workflows (DAG: `quantum_processing_dag`)
- View data processing logs and metrics
- Manage DAG schedules and configurations
- Track incremental loading to Iceberg tables
- Receive email alerts on success/failure

The `quantum_processing_dag` DAG (`docker/airflow/quantum_processing_dag.py`) runs daily and:
- Ingests VQE results from Kafka topics
- Transforms data using Spark into ML features
- Loads processed data into Apache Iceberg tables with time-travel support
- Manages configuration via Airflow Variables

**Spark Analytics**: Process and analyze quantum experiment data:
```bash
# Access Spark master UI at http://spark-master:8080
# MinIO console at http://minio:9001
# Kafka UI available through connect APIs
```

---

## Optimizer Configuration

The pipeline supports multiple optimizers with configurable parameters. The optimizer behavior is controlled by two **mutually exclusive** parameters:

- `--max-iterations MAX_ITERATIONS`: Sets a hard limit on optimization iterations
- `--convergence THRESHOLD`: Enables convergence-based optimization with specified threshold

**Supported Optimizers** (16 total):

**Gradient-Based (Recommended):**
- `L-BFGS-B` (default) - Limited-memory BFGS with bounds; recommended for GPU acceleration and accuracy
- `BFGS` - Broyden-Fletcher-Goldfarb-Shanno algorithm
- `CG` - Conjugate gradient method
- `Newton-CG` - Newton conjugate gradient
- `TNC` - Truncated Newton with bounds

**Trust-Region Methods:**
- `trust-constr` - Trust-region constrained optimization
- `trust-ncg` - Trust-region Newton conjugate gradient
- `trust-exact` - Trust-region exact Hessian
- `trust-krylov` - Trust-region Krylov method
- `dogleg` - Dog-leg trust-region algorithm

**Derivative-Free:**
- `COBYLA` - Constrained optimization by linear approximation
- `COBYQA` - Constrained optimization by quadratic approximation
- `Powell` - Powell's method
- `Nelder-Mead` - Simplex algorithm

**Sequential Methods:**
- `SLSQP` - Sequential least squares programming

**Custom:**
- `custom` - User-defined optimization function

**Best Practices**:
- Use `L-BFGS-B` for most cases - best balance of speed and accuracy
- Use `--max-iterations` for controlled runtime (e.g., `--max-iterations 100`)
- Use `--convergence` for accuracy-focused optimization (e.g., `--convergence 1e-6`)
- For larger molecules, use high `--max-iterations` values (200-500+) as they require more calculations
- Gradient-based optimizers (L-BFGS-B, BFGS, CG) converge faster on smooth energy landscapes
- Derivative-free optimizers (COBYLA, Powell) are more robust to noisy cost functions
- **Never use both `--max-iterations` and `--convergence` simultaneously** - see Troubleshooting below

**Configuration is handled in** `quantum_pipeline/solvers/optimizer_config.py:18-24`

---

## Performance Monitoring

The pipeline includes comprehensive performance monitoring with Prometheus and Grafana integration, providing deep insights into quantum simulations and system performance.

**Monitoring Capabilities:**
- **System Metrics:** CPU, GPU, memory, disk I/O, and container resource usage
- **VQE Metrics:** Energy convergence, iteration timing, optimization progress, parameter evolution
- **Scientific Accuracy:** Real-time validation against reference database with error tracking
- **Efficiency Metrics:** Iterations per second, overhead ratio, time per iteration
- **Background Collection:** Non-blocking thread-based metrics gathering
- **Multi-Format Export:** Prometheus PushGateway integration and JSON file export

**For detailed monitoring setup and configuration, see [monitoring/README.md](monitoring/README.md).**

Quick enable:
```bash
# Enable monitoring via environment variable
export QUANTUM_PERFORMANCE_ENABLED=true
export QUANTUM_PERFORMANCE_PUSHGATEWAY_URL=http://localhost:9091

# Or via CLI parameters
python quantum_pipeline.py -f data/molecules.json \
  --enable-performance-monitoring \
  --performance-interval 30 \
  --performance-pushgateway http://localhost:9091 \
  --performance-export-format both

# Or via docker-compose with monitoring stack
docker-compose -f docker-compose.yaml -f docker-compose.monitoring.yaml up
```

Access dashboards:
- **Grafana:** http://grafana:3000 (comprehensive VQE and system metrics)
- **Prometheus:** http://prometheus:9090 (raw metrics and queries)

---

## Troubleshooting

### Optimizer Configuration Issues

**Problem**: Calculations freeze or stop silently with no CPU/GPU usage

**Symptoms**:
- VQE optimization appears to hang
- CPU/GPU usage drops to 0% during optimization
- PerformanceMonitor (if enabled) still sends metrics to Prometheus, but no progress
- No error messages in logs

**Common Causes**:
1. **Using both `--max-iterations` and `--convergence` simultaneously**
   - These parameters are mutually exclusive
   - The optimizer configuration will raise a `ValueError` if both are set
   - If this check is bypassed, it can cause silent freezing

2. **Insufficient `--max-iterations` for molecule complexity**
   - Larger molecules require more iterations to converge
   - Too few iterations can cause premature termination
   - Recommendation: Start with 200-500 iterations for complex molecules

**Solutions**:
```bash
# Good: Use only max-iterations
python quantum_pipeline.py -f data/molecule.json --max-iterations 200 --optimizer L-BFGS-B

# Good: Use only convergence threshold
python quantum_pipeline.py -f data/molecule.json --convergence 1e-6 --optimizer L-BFGS-B

# Bad: Don't use both (will raise ValueError)
python quantum_pipeline.py -f data/molecule.json --max-iterations 100 --convergence 1e-6  # ❌ ERROR
```

**Recommendations**:
- For production runs: Use `--max-iterations` with a generous limit
- For research/accuracy: Use `--convergence` with appropriate threshold (1e-6 typical)
- Monitor logs for optimizer warnings about parameter recommendations
- Enable performance monitoring to detect silent failures early

---

## Examples

### Python API

The framework can be used programmatically:

```python
from quantum_pipeline.runners.vqe_runner import VQERunner

backend = VQERunner.default_backend()
runner = VQERunner(
    filepath='data/molecules.json',
    basis_set='sto3g',
    max_iterations=1,
    convergence_threshold=1e-6,
    optimizer='COBYLA',
    ansatz_reps=3
)
runner.run(backend)
```

### Docker Examples

**Single Container Execution**:
```bash
# CPU version
docker run --rm straightchlorine/quantum-pipeline:latest --file /app/data/molecule.json --basis sto-3g --max-iterations 10

# GPU version (requires NVIDIA Docker)
docker run --rm --gpus all straightchlorine/quantum-pipeline:latest-gpu --file /app/data/molecule.json --basis sto-3g --gpu
```

**Platform Deployment**:
```bash
# Deploy complete data platform
docker-compose up -d

# Execute quantum simulation with full data processing
docker-compose exec quantum-pipeline python quantum_pipeline.py \
  -f data/molecules.json \
  --kafka \
  --gpu \
  --max-iterations 150 \
  --report
```

### Example KafkaConsumer

You can test the Kafka integration with a simple consumer like this:

```python
from kafka import KafkaConsumer
from quantum_pipeline.stream.serialization.interfaces.vqe import VQEDecoratedResultInterface

class KafkaMessageConsumer:
    def __init__(self, topic='vqe_results', bootstrap_servers='localhost:9092'):
        self.deserializer = VQEDecoratedResultInterface()
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            value_deserializer=self.deserializer.from_avro_bytes,
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            group_id='vqe_consumer_group'
        )

    def consume_messages(self):
        try:
            for message in self.consumer:
                try:
                    # Process the message
                    decoded_message = message.value
                    yield decoded_message
                except Exception as e:
                    print(f"Error processing message: {str(e)}")
                    continue
        except Exception as e:
            print(f"Error in consumer: {str(e)}")
        finally:
            self.consumer.close()
```

Then you can use the consumer like this:
```python
consumer = KafkaMessageConsumer()
for msg in consumer.consume_messages():
    print(f"Received message: {msg}")
```

### Data Analytics with Spark

Access processed quantum data through Iceberg tables:
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Quantum Data Analytics") \
    .config("spark.sql.catalog.quantum_catalog", "org.apache.iceberg.spark.SparkCatalog") \
    .getOrCreate()

# Query VQE results
vqe_results = spark.sql("""
    SELECT molecule_id, basis_set, minimum_energy, total_iterations
    FROM quantum_catalog.quantum_features.vqe_results
    WHERE processing_date >= '2025-01-01'
""")

# Analyze convergence patterns
convergence = spark.sql("""
    SELECT experiment_id, iteration_step, iteration_energy
    FROM quantum_catalog.quantum_features.vqe_iterations
    ORDER BY experiment_id, iteration_step
""")
```
---

## Architecture Overview

The platform follows a modern data architecture with the following components:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Quantum        │───▶│  Apache Kafka    │───▶│  Apache Spark   │
│  Pipeline       │    │  (Streaming)     │    │  (Processing)   │
│  (VQE Runner)   │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Apache Airflow │    │  Schema Registry │    │  Apache Iceberg │
│  (Orchestration)│    │  (Avro Schemas)  │    │  (Data Lake)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌──────────────────┐
                    │  MinIO Storage   │
                    │  (Object Store)  │
                    └──────────────────┘
```

## CI/CD and Deployment

The project includes comprehensive CI/CD pipelines via GitHub Actions (`.github/` folder):

- **Automated Testing**: Python tests with flake8 linting on every PR
- **Docker Image Building**: Automatic builds for CPU and GPU variants
- **Security Scanning**: Trivy vulnerability scans for all container images
- **DockerHub Publishing**: Automated daily and tag-based releases
- **Image Signing**: Cosign-based container signing for security

Available Docker images:
- `straightchlorine/quantum-pipeline:latest` (CPU optimized)
- `straightchlorine/quantum-pipeline:latest-gpu` (GPU accelerated)
- `straightchlorine/quantum-pipeline:nightly-cpu` (Development builds)
- `straightchlorine/quantum-pipeline:nightly-gpu` (Development builds)

## Contributing

For now, this project is not open for contributions since it is a university project, but feel free to fork it and make your own version.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## Contact

For questions or support, please reach out to:
- **Email:** piotrkrzysztoflis@pm.me
- **GitHub:** [straightchlorine](https://github.com/straightchlorine)
