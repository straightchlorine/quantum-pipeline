# Frequently Asked Questions

This page addresses the most common questions about the Quantum Pipeline, organized by topic.
Click on any question to expand the answer.

## General

??? question "What is Quantum Pipeline?"
    Quantum Pipeline is an extensible framework for exploring Variational Quantum
    Eigensolver (VQE) algorithms backed by a modern data stack. It combines
    quantum chemistry simulations using Qiskit with a streaming data platform built
    on Apache Kafka, Spark, Airflow, and Iceberg.

    The system supports both CPU and GPU-accelerated simulation, and all results are
    automatically serialized, streamed, and stored for analysis. It was developed as
    part of a thesis project to demonstrate effects of GPU acceleration on
    quantum computing workloads.

??? question "What is the Variational Quantum Eigensolver (VQE)?"
    VQE is a hybrid quantum-classical algorithm that iteratively optimizes a parameterized quantum circuit to find the ground state energy of a molecular system. See the [VQE Algorithm](../scientific/vqe-algorithm.md) page for details on how Quantum Pipeline implements it.

??? question "What molecules are supported?"
    The system supports any molecule that can be described by the PySCF molecular
    specification format. The thesis benchmark set (`data/molecules.thesis.json`)
    includes:

    - **H\(_2\)** (Hydrogen) - 4 qubits with sto-3g
    - **HeH\(^+\)** (Helium hydride cation) - 4 qubits with sto-3g
    - **LiH** (Lithium hydride) - 8 qubits with sto-3g
    - **BeH\(_2\)** (Beryllium hydride) - 10 qubits with sto-3g
    - **H\(_2\)O** (Water) - 12 qubits with sto-3g
    - **NH\(_3\)** (Ammonia) - 12 qubits with sto-3g
    - **CO\(_2\)** (Carbon dioxide)
    - **N\(_2\)** (Nitrogen)

    The general-purpose file (`data/molecules.json`) contains 9 molecules
    including He2, BeH, BH, and CH\(_4\). You can define custom molecules by
    specifying atomic symbols and 3D coordinates. See
    [Basic Usage](../getting-started/basic-usage.md) for examples of custom
    molecule definitions.

??? question "Can Quantum Pipeline connect to real quantum hardware?"
    The system was designed with the capability to integrate with real quantum computers
    via IBM Quantum backends. However, due to access costs, testing uses the Qiskit
    Aer simulator exclusively.

    The architecture allows for swapping in real quantum backends with minimal code
    changes - the backend configuration is abstracted behind a consistent interface.

??? question "What license is Quantum Pipeline released under?"
    Quantum Pipeline is open-source software under MIT license.

    [GitHub](https://github.com/straightchlorine/quantum-pipeline) &middot;
    [Codeberg](https://codeberg.org/piotrkrzysztof/quantum-pipeline)

??? question "Who developed Quantum Pipeline?"
    Quantum Pipeline was developed by Piotr Krzysztof Lis as part of a thesis project.
    The project demonstrates the integration of quantum computing simulations with
    data engineering using streaming and batch processing technologies.

    [GitHub](https://github.com/straightchlorine/quantum-pipeline) &middot;
    [Codeberg](https://codeberg.org/piotrkrzysztof/quantum-pipeline)

## Installation

??? question "Which Python version is required?"
    Quantum Pipeline requires Python 3.10, 3.11, or 3.12.

    During development `uv` was extensively used to ensure proper versioning.
    Alternatives include `conda`, `pyenv` and similar tools to ensure correct Python version.
    See the [Installation](../getting-started/installation.md) guide for detailed setup instructions.

??? question "Can I use Quantum Pipeline on Windows?"
    The recommended environment is Linux. Windows users can run simulations through WSL2 with Docker Desktop.
    MacOS supports CPU-only simulations (no GPU acceleration due to the NVIDIA CUDA requirement).
    Native Windows execution is not officially supported and only Linux is tested.

??? question "How do I install GPU support?"
    Install NVIDIA drivers and the NVIDIA Container Toolkit on the host, then use the GPU image (`straightchlorine/quantum-pipeline:gpu`).
    The image includes CUDA 12.6 and cuDNN. You need to set the `CUDA_ARCH` build argument to match
    your GPU architecture when building locally (e.g. `6.1` for GTX 10xx, `7.5` for RTX 20xx, `8.6` for RTX 30xx). 

    See [GPU Acceleration](../deployment/gpu-acceleration.md) for complete setup instructions.

??? question "Can I install without Docker?"
    Yes. The core simulation module can be installed as a Python package using
    `pip install quantum-pipeline`. This allows you to run VQE simulations directly
    on your machine.

??? question "What are the required system dependencies?"
    Python 3.10-3.12, Docker, and Docker Compose. For GPU support: NVIDIA drivers and the NVIDIA Container Toolkit.
    All scientific computing dependencies are managed through pip.

## Usage

??? question "How do I choose the right optimizer?"
    The most commonly used are **COBYLA** and **L-BFGS-B**.

    The pipeline supports 8 optimizers used in batch generation:

    - **L-BFGS-B** - limited-memory BFGS, good general default
    - **COBYLA** - gradient-free, handles noisy landscapes
    - **SLSQP** - sequential least squares, gradient-based
    - **Nelder-Mead** - simplex method, gradient-free
    - **Powell** - directional set method, gradient-free
    - **BFGS** - full quasi-Newton method
    - **CG** - conjugate gradient
    - **TNC** - truncated Newton with bounds

    Additional optimizers (Newton-CG, COBYQA, trust-constr, dogleg,
    trust-ncg, trust-exact, trust-krylov) are available but not tested.
    See the [Optimizers](../usage/optimizers.md) page for a comparison.

??? question "What basis set should I use?"
    Start with **sto-3g** for rapid prototyping (fewest qubits, fastest execution). Use **cc-pvdz** for high-accuracy results (GPU speedup up to 4x). See [Basis Sets](../scientific/basis-sets.md) for a detailed comparison.

??? question "Which ansatz should I use?"
    Three ansatze are available via the `--ansatz` flag:

    - **EfficientSU2** (default) - hardware-efficient with RY/RZ rotations and
      CX entangling gates. Good general choice.
    - **RealAmplitudes** - real-valued amplitude ansatz. Simpler circuit, fewer
      parameters per layer.
    - **ExcitationPreserving** - preserves particle number. Physically motivated
      but more constrained.

    Control the circuit depth with `--ansatz-reps` (default: 2). More repetitions
    increase expressiveness at the cost of more parameters to optimize.

??? question "What initialization strategies are available?"
    Two strategies are available via the `--init-strategy` flag:

    - **random** (default) - uniform random parameters in [0, 2pi). Simple but
      prone to local minima, especially for larger molecules or higher basis sets.
    - **hf** (Hartree-Fock) - starts from a classically pre-optimized state that
      approximates the Hartree-Fock solution through the chosen ansatz. Generally
      converges faster and avoids barren plateaus, though pre-optimization fidelity
      decreases for larger molecules.

    For data collection or benchmarking, running both strategies across multiple
    seeds gives the most useful comparison.

??? question "How do I run multiple simulations in parallel?"
    The batch generation system is the recommended way to run many simulations.
    It defines 4 tiers of increasing complexity and distributes work across
    3 hardware lanes (two GPU lanes and one CPU lane) that run concurrently:

    ```bash
    # Run the full batch (resumes from where it left off)
    just ml-generate
    ```

    Each tier varies the basis set, optimizer set, init strategy, seeds, and
    ansatz. Tier 1 (sto-3g, all 8 optimizers, 25 seeds) is the broadest sweep;
    Tier 4 (cc-pvdz, 3 optimizers, 10 seeds) targets the hardest configurations.

    For simpler parallel runs, you can also launch multiple `docker compose run`
    instances with different molecule or optimizer configurations.

??? question "How do I interpret the simulation output?"
    The key output values are:

    - **Minimum energy**: The ground state energy found, measured in Hartree (Ha).
      Lower values indicate a better approximation of the true ground state.
    - **Total iterations**: Number of optimizer steps taken to complete optimization.
    - **Timing breakdown**: Hamiltonian construction time, qubit mapping time,
      VQE optimization time, and total execution time.
    - **Convergence status**: Whether the optimizer reached the convergence
      threshold or hit the iteration limit.

## Data Platform

??? question "Do I need Kafka to run simulations?"
    No. The simulation module can run in standalone mode without Kafka. Results will
    be printed to the console or saved to local files.

??? question "Can I use the pipeline without Spark?"
    Yes. Kafka and Garage will still store your simulation results as Avro files in
    the object storage bucket. Spark is only needed for the incremental processing
    step that transforms raw results into structured feature tables in Apache
    Iceberg format.

    You can query the raw Avro files directly using any Avro-compatible tool,
    including Python's `fastavro` library or Apache Spark in standalone mode.

??? question "How do I access stored simulation data?"
    There are several ways to access the data:

    - **S3-compatible client**: Any S3 SDK (boto3, rclone, etc.) can read from
      Garage using the `S3_ENDPOINT`, `S3_ACCESS_KEY`, and `S3_SECRET_KEY`
      environment variables.
    - **Spark SQL**: Query Iceberg feature tables with SQL through Spark.

    Feature tables are stored under `features/warehouse/quantum_features/` in
    Garage, and synced to Cloudflare R2 (`qp-data/features/`) via the R2 sync DAG.

    You can also use configured `aws` tool to browse through the S3 storage.

??? question "What serialization format is used?"
    Apache Avro for wire serialization between the producer and Kafka, with schemas managed by the Confluent Schema Registry. Data at rest is JSON (via Redpanda Connect) or Avro (via Kafka Connect). See [Serialization](../architecture/serialization.md) for details.

??? question "How does schema evolution work?"
    New Avro schemas are registered in the Schema Registry and published to the `experiment.vqe` Kafka topic. Redpanda Connect (configured in `compose/redpanda-connect.yaml`) consumes from this topic and writes to Garage object storage, so multiple schema versions coexist without downtime. See [Kafka Streaming](../data-platform/kafka-streaming.md) for details.

??? question "What happens if Kafka goes down during a simulation?"
    The Kafka producer is configured with [`retries: 3`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/configs/defaults.py#L38) and `acks: all` for
    durability. If Kafka is temporarily unavailable, the producer will [retry
    message delivery](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/stream/kafka_interface.py#L101).

    Simulation results are not lost as long as Kafka retains the messages
    (default retention: 7 days).

## Monitoring

??? question "How do I enable monitoring?"
    Set the `MONITORING_ENABLED` environment variable to `true`, or pass
    `--enable-performance-monitoring` on the command line. Metrics are pushed
    to the Prometheus PushGateway at the URL specified by `PUSHGATEWAY_URL`
    (default: `http://localhost:9091`).

    Other monitoring environment variables:

    - `MONITORING_INTERVAL` - collection interval in seconds (default: 10)
    - `MONITORING_EXPORT_FORMAT` - export format (`prometheus`, `json`, or `both`)
    - `CONTAINER_TYPE` - label for the container (e.g. `GPU_GTX1060_6GB`, `CPU`)

??? question "What metrics are collected?"
    Metrics use the `qp_` prefix and fall into three groups:

    - **`qp_vqe_*`** - per-experiment VQE metrics (timing, energy, accuracy,
      iterations, convergence)
    - **`qp_sys_*`** - system resource metrics (CPU, memory, uptime)
    - **`qp_batch_*`** - batch generation progress (total, done, failed,
      in-progress, pending per tier and lane)

    See [Performance Metrics](../monitoring/performance-metrics.md) for the full
    list and label descriptions.

## Deployment

??? question "Can I run Quantum Pipeline on a cloud provider?"
    Yes. The Docker Compose deployment runs on any cloud VM with Docker installed.

??? question "How do I scale the processing layer?"
    The Spark processing layer scales horizontally by adding more worker nodes.
    Update the Docker Compose file to add additional `spark-worker` services.

    Kafka scales by adding partitions to topics and additional broker nodes.
    Garage can be deployed in a multi-node layout for storage scaling.

??? question "How do I back up my data?"
    Garage data is stored in Docker volumes. Use `docker volume` commands or
    host-mounted volumes for backups. The R2 sync DAG automatically replicates
    feature tables to Cloudflare R2 as an off-site backup. Iceberg's metadata
    layer provides point-in-time snapshots with time-travel query support.

??? question "How do I update to a newer version?"
    Pull the latest Docker images and restart the services:

    ```bash
    docker compose pull
    docker compose down
    docker compose up -d
    ```

    For pip installations, upgrade with `pip install --upgrade quantum-pipeline`.

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
    Try `--init-strategy hf` to start from a better initial point,
    increase `--ansatz-reps` for expressiveness, run multiple
    simulations to avoid local minima, use GPU acceleration, start with sto-3g,
    and adjust `--convergence` tolerance for faster results.

??? question "Why are my energy values different from reference literature?"
    Common causes: random parameter initialization converging on local minima, limited ansatz expressiveness, or early termination from convergence tolerance. Run multiple simulations and select the best result, or try `--init-strategy hf` for a better starting point. See [Benchmarking](../scientific/benchmarking.md) for detailed analysis.

For issues not covered here, see the [Troubleshooting](troubleshooting.md) guide or
open an issue on [GitHub](https://github.com/straightchlorine/quantum-pipeline).
