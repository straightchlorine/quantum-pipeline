# Frequently Asked Questions

This page addresses the most common questions about the Quantum Pipeline, organized by topic.
Click on any question to expand the answer.

---

## General

??? question "What is Quantum Pipeline?"
    Quantum Pipeline is an extensible framework for exploring Variational Quantum
    Eigensolver (VQE) algorithms with production-grade data engineering. It combines
    quantum chemistry simulations using Qiskit with a streaming data platform built
    on Apache Kafka, Spark, Airflow, and Iceberg.

    The system supports both CPU and GPU-accelerated simulation, and all results are
    automatically serialized, streamed, and stored for analysis. It was developed as
    part of a thesis project to demonstrate the integration of quantum computing
    workloads with modern data engineering practices.

??? question "What is the Variational Quantum Eigensolver (VQE)?"
    VQE is a hybrid quantum-classical algorithm that iteratively optimizes a parameterized quantum circuit to find the ground state energy of a molecular system. See the [VQE Algorithm](../scientific/vqe-algorithm.md) page for details on how Quantum Pipeline implements it.

??? question "What molecules are supported?"
    The system supports any molecule that can be described by the PySCF molecular
    specification format. Pre-configured molecules include:

    - **H2** (Hydrogen) - 4 qubits with sto-3g
    - **HeH+** (Helium hydride cation) - 4 qubits with sto-3g
    - **LiH** (Lithium hydride) - 8 qubits with sto-3g
    - **BeH2** (Beryllium hydride) - 10 qubits with sto-3g
    - **H2O** (Water) - 12 qubits with sto-3g
    - **NH3** (Ammonia) - 12 qubits with sto-3g

    You can define custom molecules by specifying atomic symbols and 3D coordinates.
    See [Basic Usage](../getting-started/basic-usage.md) for examples of custom molecule
    definitions.

??? question "Can Quantum Pipeline connect to real quantum hardware?"
    The system was designed with the capability to integrate with real quantum computers
    via IBM Quantum backends. However, due to access costs, the current implementation
    uses the Qiskit Aer simulator exclusively.

    The architecture allows for swapping in real quantum backends with minimal code
    changes - the backend configuration is abstracted behind a consistent interface.

??? question "What license is Quantum Pipeline released under?"
    Quantum Pipeline is open-source software. Check the repository on
    [GitHub](https://github.com/straightchlorine/quantum-pipeline) for the current
    license terms and contribution guidelines.

??? question "Who developed Quantum Pipeline?"
    Quantum Pipeline was developed by Piotr Krzysztof Lis as part of a thesis project.
    The project demonstrates the integration of quantum computing simulations with
    production-grade data engineering using modern streaming and batch processing
    technologies. The source code is available on
    [GitHub](https://github.com/straightchlorine/quantum-pipeline) and [Codeberg](https://codeberg.org/piotrkrzysztof/quantum-pipeline).

---

## Installation

??? question "Which Python version is required?"
    Quantum Pipeline requires Python 3.10, 3.11, or 3.12. Python 3.13 and later
    are not yet supported due to dependency constraints in Qiskit and related
    libraries.

    During development `pyenv` was extensively used to ensure proper versioning.
    Alternatives include `conda`, `uv` nad similar tools to ensure correct Python version.
    See the [Installation](../getting-started/installation.md) guide for detailed setup instructions.

??? question "Can I use Quantum Pipeline on Windows?"
    The recommended environment is Linux. Windows users can run simulations through WSL2 with Docker Desktop. macOS supports CPU-only simulations (no GPU acceleration due to the NVIDIA CUDA requirement). Native Windows execution is not officially supported.

??? question "How do I install GPU support?"
    Install NVIDIA drivers and the NVIDIA Container Toolkit on the host, then use the GPU image (`straightchlorine/quantum-pipeline:latest-gpu`). The image includes CUDA 11.8.0 and cuDNN 8. See [GPU Acceleration](../deployment/gpu-acceleration.md) for complete setup instructions.

??? question "Can I install without Docker?"
    Yes. The core simulation module can be installed as a Python package using
    `pip install quantum-pipeline`. This allows you to run VQE simulations directly
    on your machine.

??? question "What are the required system dependencies?"
    Python 3.10-3.12, Docker, and Docker Compose. For GPU support: NVIDIA drivers and the NVIDIA Container Toolkit. All scientific computing dependencies are managed through pip.

---

## Usage

??? question "How do I choose the right optimizer?"
    The default is **L-BFGS-B** (gradient-based, fast optimization). Use **COBYLA** for noisy landscapes. See the [Optimizers](../usage/optimizers.md) page for a complete comparison of all available optimizers.

??? question "What basis set should I use?"
    Start with **sto-3g** for rapid prototyping (fewest qubits, fastest execution). Use **cc-pvdz** for high-accuracy results (GPU speedup up to 4x). See [Basis Sets](../scientific/basis-sets.md) for a detailed comparison.

??? question "How do I run multiple simulations in parallel?"
    Deploy multiple simulation containers using Docker Compose. Each container
    operates independently and publishes results to Kafka. You can scale by:

    1. Adjusting the `replicas` count in Docker Compose.
    2. Launching additional `docker compose run` instances with different
       molecule configurations.
    3. Using different optimizer or basis set parameters per container.

    Kafka handles the parallel message streams, and Kafka Connect automatically
    persists all results to MinIO regardless of how many producers are active.

??? question "What is the EfficientSU2 ansatz?"
    EfficientSU2 is a hardware-efficient parameterized quantum circuit from Qiskit using alternating RY/RZ rotation and CX entangling gates. The [`--ansatz-reps`](https://github.com/straightchlorine/quantum-pipeline/blob/master/quantum_pipeline/configs/parsing/argparser.py#L76) parameter controls the number of repeated layers (default: 2). More repetitions increase expressiveness at the cost of more parameters to optimize.

??? question "How do I interpret the simulation output?"
    The key output values are:

    - **Minimum energy**: The ground state energy found, measured in Hartree (Ha).
      Lower values indicate a better approximation of the true ground state.
    - **Total iterations**: Number of optimizer steps taken to complete optimization.
    - **Timing breakdown**: Hamiltonian construction time, qubit mapping time,
      VQE optimization time, and total execution time.
    - **Convergence status**: Whether the optimizer reached the convergence
      threshold or hit the iteration limit.

---

## Data Platform

??? question "Do I need Kafka to run simulations?"
    No. The simulation module can run in standalone mode without Kafka. Results will
    be printed to the console or saved to local files.

??? question "Can I use the pipeline without Spark?"
    Yes. Kafka and MinIO will still store your simulation results as Avro files in
    the `s3://local-vqe-results/experiments/` bucket. Spark is only needed for the
    incremental processing step that transforms raw results into structured feature
    tables in Apache Iceberg format.

    You can query the raw Avro files directly using any Avro-compatible tool,
    including Python's `fastavro` library or Apache Spark in standalone mode.

??? question "How do I access stored simulation data?"
    There are several ways to access the data:

    - **MinIO Console**: Browse files at `http://localhost:9001` (default port; thesis config uses 9002) (credentials from `.env`)
    - **MinIO Client (mc)**: Command-line access using the MinIO client tool
    - **S3 SDK**: Any S3-compatible SDK (boto3, etc.) can read from MinIO
    - **Spark SQL**: Query Iceberg feature tables with SQL through Spark

    Raw data is stored at `s3://local-vqe-results/experiments/` and processed
    feature tables at `s3://local-features/warehouse/`.

??? question "What serialization format is used?"
    Apache Avro for binary serialization, with schemas managed by the Confluent Schema Registry. See [Avro Serialization](../architecture/avro-serialization.md) for the schema definition and rationale.

??? question "How does schema evolution work?"
    New Avro schemas are registered in the Schema Registry and published to versioned Kafka topics (e.g., `vqe_decorated_result_v2`). Kafka Connect subscribes via [`topics.regex`](https://github.com/straightchlorine/quantum-pipeline/blob/master/docker/connectors/minio-sink-config.json#L7), so multiple schema versions coexist without downtime. See [Kafka Streaming](../data-platform/kafka-streaming.md) for details.

??? question "What happens if Kafka or MinIO goes down during a simulation?"
    The Kafka producer is configured with [`retries: 3`](https://github.com/straightchlorine/quantum-pipeline/blob/master/quantum_pipeline/configs/defaults.py#L35) and `acks: all` for
    durability. If Kafka is temporarily unavailable, the producer will [retry
    message delivery](https://github.com/straightchlorine/quantum-pipeline/blob/master/quantum_pipeline/stream/kafka_interface.py#L103). If MinIO is down, Kafka Connect will buffer messages
    in Kafka and retry storage when MinIO recovers.

    Simulation results are not lost as long as Kafka retains the messages
    (default retention: 7 days).

---

## Deployment

??? question "Can I run Quantum Pipeline on a cloud provider?"
    Yes. The Docker Compose deployment runs on any cloud VM with Docker installed.

??? question "How do I scale the processing layer?"
    The Spark processing layer scales horizontally by adding more worker nodes.
    Update the Docker Compose file to add additional `spark-worker` services, or
    deploy on a Spark cluster with a resource manager (YARN, Kubernetes).

    Kafka scales by adding partitions to topics and additional broker nodes.
    MinIO can be deployed in distributed mode across multiple nodes for storage
    scaling.

??? question "How do I back up my data?"
    MinIO data is stored in Docker volumes. Use `docker volume` commands or host-mounted volumes for backups. Iceberg's metadata layer provides point-in-time snapshots with time-travel query support as built-in versioning.

??? question "How do I update to a newer version?"
    Pull the latest Docker images and restart the services:

    ```bash
    docker compose pull
    docker compose down
    docker compose up -d
    ```

    For pip installations, upgrade with `pip install --upgrade quantum-pipeline`.

---

## Performance

??? question "What speedup can I expect with GPU acceleration?"
    Speedup depends on the basis set and molecule complexity:

    - **sto-3g basis set**: 1.7-1.8x average speedup, up to 2.1x for medium
      molecules (8-10 qubits)
    - **cc-pVDZ basis set**: 3.5-4.1x speedup due to larger matrix operations

    Small molecules (4 qubits) may see no benefit or slight slowdown due to GPU
    overhead. The greatest speedup is observed for molecules of medium complexity
    where the computation-to-overhead ratio is most favorable. See
    [Benchmarking](../scientific/benchmarking.md) for detailed results.

??? question "How do I speed up optimization?"
    Use L-BFGS-B optimizer (default), increase `--ansatz-reps` for expressiveness, run multiple simulations to avoid local minima, use GPU acceleration, start with sto-3g, and adjust `--convergence` tolerance for faster results.

??? question "What are the memory requirements for different molecules?"
    Memory usage scales exponentially with qubit count due to statevector simulation:

    | Molecule | Qubits (sto-3g) | Approx. Memory |
    |----------|----------------|----------------|
    | H2       | 4              | < 1 GB         |
    | HeH+     | 4              | < 1 GB         |
    | LiH      | 8              | 1-2 GB        |
    | BeH2     | 10             | 2-4 GB        |
    | H2O      | 12             | 4-8 GB        |
    | NH3      | 12             | 4-8 GB        |

    Using the cc-pVDZ basis set significantly increases qubit count and memory
    requirements for all molecules.

??? question "Why are my energy values different from reference literature?"
    Common causes: random parameter initialization converging on local minima, limited ansatz expressiveness, or early termination from convergence tolerance. Run multiple simulations and select the best result. See [Benchmarking](../scientific/benchmarking.md) for detailed analysis.

---

For issues not covered here, see the [Troubleshooting](troubleshooting.md) guide or
open an issue on [GitHub](https://github.com/straightchlorine/quantum-pipeline/issues).
