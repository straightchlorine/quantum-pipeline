# Quantum Pipeline

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
├── report/               # Report generation utilities
├── runners/              # VQE execution logic
├── solvers/              # VQE solver implementations
├── stream/               # Kafka streaming and messaging utilities
├── structures/           # Quantum and classical data structures
├── utils/                # Utility functions (logging, visualization, etc.)
├── visual/               # Visualization tools for molecules and operators
├── docker/               # Docker configurations and deployment files
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

5. **(Alternative) Build Individual Containers**:
   ```bash
   # CPU-optimized container
   docker build -f docker/Dockerfile.cpu .

   # GPU-accelerated container (requires NVIDIA Docker)
   docker build -f docker/Dockerfile.gpu .

   # Spark processing container
   docker build -f docker/Dockerfile.spark .
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

- `-f FILE, --file FILE`: Path to the molecule data file (required).
- `-b BASIS, --basis BASIS`: Specify the basis set for the simulation.
- `--local`: Use a local quantum simulator instead of IBM Quantum.
- `--min-qubits MIN_QUBITS`: Specify the minimum number of qubits required.
- `--max-iterations MAX_ITERATIONS`: Set the maximum number of VQE iterations.
- `--optimizer OPTIMIZER`: Choose from a variety of optimization algorithms.
- `--output-dir OUTPUT_DIR`: Specify the directory for storing output files.
- `--log-level {DEBUG,INFO,WARNING,ERROR}`: Set the logging level.
- `--shots SHOTS`: Number of shots for quantum circuit execution.
- `--optimization-level {0,1,2,3}`: Circuit optimization level.
- `--report`: Generate a PDF report after simulation.
- `--kafka`: Stream data to Apache Kafka for real-time processing.

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

**Kafka Streaming**: Enable real-time streaming to Apache Kafka:
```bash
python quantum_pipeline.py -f data/molecule.json --kafka
```

**Full Platform Deployment**: Launch with complete data processing pipeline:
```bash
# Start all services
docker-compose up -d

# Run quantum pipeline with data streaming
docker-compose exec quantum-pipeline python quantum_pipeline.py -f data/molecules.json --kafka --gpu
```

**Airflow Orchestration**: Access the Airflow web interface at `http://localhost:8084` to:
- Monitor automated daily processing workflows
- View data processing logs and metrics
- Manage DAG schedules and configurations

**Spark Analytics**: Process and analyze quantum experiment data:
```bash
# Access Spark master UI at http://localhost:8080
# MinIO console at http://localhost:9001
# Kafka UI available through connect APIs
```

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
- **Email:** piotrlis555@gmail.com
- **GitHub:** [straightchlorine](https://github.com/straightchlorine)
