# Deployment

Three ways to run Quantum Pipeline: install the PyPI package directly, run a
single Docker container, or bring up the full platform with Docker Compose.

<div class="grid cards" markdown>

-   :material-docker:{ .lg .middle } **Docker Basics**

    ---

    Container images (CPU, GPU, Spark, Airflow), building from source,
    and running individual containers.

    [:octicons-arrow-right-24: Docker Basics](docker-basics.md)

-   :material-layers-triple:{ .lg .middle } **Docker Compose**

    ---

    Full platform deployment with Kafka, Spark, Airflow, Garage, MLflow,
    and monitoring exporters.

    [:octicons-arrow-right-24: Docker Compose](docker-compose.md)

-   :material-expansion-card:{ .lg .middle } **GPU Acceleration**

    ---

    NVIDIA setup, CUDA build args, Qiskit Aer GPU backend, and
    performance benchmarks from the thesis experiments.

    [:octicons-arrow-right-24: GPU Acceleration](gpu-acceleration.md)

-   :material-variable:{ .lg .middle } **Environment Variables**

    ---

    Reference for all `.env` variables used across services in the
    Docker Compose deployment.

    [:octicons-arrow-right-24: Environment Variables](environment-variables.md)

</div>
