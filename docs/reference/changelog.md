# Changelog

All notable changes to this project will be documented in this page.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.3.1] - 2025-06-15

### Added

- Comprehensive MkDocs documentation site with Material theme.
- Privacy plugin integration for self-hosted assets.
- Custom figure styling for documentation images.
- Git revision date plugin for page modification tracking.
- Documentation sections: Getting Started, Usage, Architecture, Data Platform, Deployment, Monitoring, Scientific, and Reference.
- Mirror notice in project README.

### Changed

- Increased test coverage across core modules with additional unit and integration tests.
- Fixed several issues identified during test expansion.

### Fixed

- Corrected behavior in edge cases discovered through expanded test coverage.

---

## [1.3.0] - 2025-05-20

### Added

- Justfile for convenient execution of common development tasks (testing, linting, building).
- Expanded test suite with increased coverage for VQE simulation and data serialization modules.

### Changed

- Removed `.txt` requirement files in favor of consolidated dependency management in `pyproject.toml`.
- Updated and cleaned up reference values, removing uncertain reference entries.

### Fixed

- Several test failures and edge cases in monitoring and simulation modules.

---

## [1.2.0] - 2025-04-10

### Added

- Grafana thesis analysis dashboard (`quantum-pipeline-thesis.json`) with 16 panels across 4 sections.
- Docker Compose optimized settings for thesis experiment execution.
- Comprehensive monitoring and Docker deployment documentation in README.
- Convergence analysis graphs and performance visualization support.

### Changed

- Updated `pyproject.toml` with monitoring dashboard configuration.
- Improved Docker Compose configuration for running large iteration counts.

### Fixed

- Convergence graph generation corrected to avoid downsampling artifacts.
- Monitoring shutdown behavior now properly cleans up background threads.
- Temporarily switched to COBYLA optimizer for Grafana testing, then restored L-BFGS-B.

---

## [1.1.0] - 2025-03-15

### Added

- GPU performance monitoring with NVIDIA SMI integration.
- Additional Prometheus metrics for GPU utilization and memory.
- Timezone information to GPU container images.
- Verbose logging for monitoring subsystem.

### Changed

- Updated monitoring defaults for improved metric accuracy.
- Simplified monitoring code and removed deprecated methods.
- Disabled active GPU polling in favor of event-driven metric collection.

### Fixed

- NVIDIA SMI path resolution in containerized environments.
- Prometheus metric timestamps corrected from milliseconds to seconds.
- Prometheus metric exports now accept both HTTP 200 and 202 status codes.
- GPU name escaping in Prometheus metric labels.
- Removed timestamps from Prometheus metrics (PushGateway handles timestamps).
- Fixed GPU metrics export for multi-GPU configurations (gpu1, gpu2).

---

## [1.0.0] - 2025-02-01

### Added

- Core VQE simulation module using Qiskit Aer backend.
- Support for multiple optimizers via SciPy (L-BFGS-B, COBYLA, SLSQP, and others).
- EfficientSU2 ansatz with configurable repetitions.
- GPU acceleration using CUDA via `qiskit-aer-gpu`.
- Apache Kafka integration for streaming VQE results.
- Avro serialization with Confluent Schema Registry for type-safe data exchange.
- Dynamic topic creation with automatic schema versioning.
- Kafka Connect S3 Sink Connector for automatic data persistence to MinIO.
- Apache Spark incremental processing with master-worker configuration.
- Apache Iceberg integration for versioned feature tables with time-travel support.
- Nine specialized feature tables: molecules, ansatz\_info, performance\_metrics, vqe\_results, initial\_parameters, optimal\_parameters, vqe\_iterations, iteration\_parameters, and hamiltonian\_terms.
- Apache Airflow DAG for daily scheduled processing (`quantum_feature_processing`).
- SparkSubmitOperator with retry logic and email notifications.
- Prometheus PushGateway integration for container metrics export.
- System metrics collection: CPU usage, memory usage, iteration count, energy values, execution time.
- Docker Compose deployment with full service stack.
- Support for molecules: H2, HeH+, LiH, BeH2, H2O, NH3.
- Configurable convergence thresholds and maximum iteration limits.
- Statevector simulation method for both CPU and GPU backends.
- MinIO S3-compatible object storage integration.
- Environment variable configuration via `.env` files.
- Performance monitoring with `--enable-performance-monitoring` flag.

---

## [0.2.0] - 2025-01-15

### Added

- Apache Kafka integration for streaming VQE results.
- Avro serialization with basic schema support.
- MinIO integration for object storage.
- Docker Compose configuration for multi-service deployment.
- Schema Registry for Avro schema management.

### Changed

- Refactored simulation module to support pluggable output backends.

---

## [0.1.0] - 2025-01-01

### Added

- Initial project structure and repository setup.
- Basic VQE simulation capability for H2 molecule.
- Command-line interface for running simulations.
- Preliminary Kafka producer for result publishing.
- Support for sto-3g basis set.
- EfficientSU2 ansatz implementation.
- Basic convergence detection.

---

[1.3.1]: https://github.com/straightchlorine/quantum-pipeline/compare/v1.3.0...v1.3.1
[1.3.0]: https://github.com/straightchlorine/quantum-pipeline/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/straightchlorine/quantum-pipeline/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/straightchlorine/quantum-pipeline/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/straightchlorine/quantum-pipeline/compare/v0.2.0...v1.0.0
[0.2.0]: https://github.com/straightchlorine/quantum-pipeline/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/straightchlorine/quantum-pipeline/releases/tag/v0.1.0
