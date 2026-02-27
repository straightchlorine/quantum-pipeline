<p align="center">
  <img src="docs/assets/banner.svg" alt="Quantum Pipeline" width="350">
</p>

<div align="center">

**Repository:** [GitHub](https://github.com/straightchlorine/quantum-pipeline) (primary) · [Codeberg](https://codeberg.org/piotrkrzysztof/quantum-pipeline) (mirror)

[![PyPI version](https://badge.fury.io/py/quantum-pipeline.svg)](https://pypi.org/project/quantum-pipeline/)
[![Total Downloads](https://static.pepy.tech/badge/quantum-pipeline)](https://pepy.tech/project/quantum-pipeline)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/quantum-pipeline)](https://pypi.org/project/quantum-pipeline/)
[![Docker Pulls](https://img.shields.io/docker/pulls/straightchlorine/quantum-pipeline.svg)](https://hub.docker.com/r/straightchlorine/quantum-pipeline)

[Documentation](https://docs.qp.piotrkrzysztof.dev) · [Quick Start](https://docs.qp.piotrkrzysztof.dev/getting-started/quick-start/) · [Examples](https://docs.qp.piotrkrzysztof.dev/usage/examples/)

</div>

---

An extensible framework for running quantum algorithms backed by a modern data stack.
Combines quantum simulation with Kafka streaming, Spark-based feature engineering, Iceberg storage, and Airflow orchestration.

Currently implements the Variational Quantum Eigensolver (VQE) for ground-state energy estimation of molecular systems.

## Quick Start

```bash
pip install quantum-pipeline
quantum-pipeline -f molecules.json -b sto-3g --max-iterations 100 --optimizer L-BFGS-B
```

Or with Docker:

```bash
docker pull straightchlorine/quantum-pipeline:latest
docker run --rm straightchlorine/quantum-pipeline:latest -f /app/data/molecule.json -b sto-3g
```

See the [installation guide](https://docs.qp.piotrkrzysztof.dev/getting-started/installation/) for detailed setup, including GPU acceleration and full platform deployment.

## Features

**Quantum Computing** — VQE execution with 16+ optimizers, configurable ansatz circuits, multiple basis sets (sto-3g, 6-31g, cc-pVDZ), and GPU acceleration via CUDA. [Learn more](https://docs.qp.piotrkrzysztof.dev/scientific/vqe-algorithm/)

**Data Platform** — Real-time Kafka streaming with Avro serialization, Spark-based ML feature engineering, Iceberg data lake with time-travel, Airflow workflow orchestration. [Architecture overview](https://docs.qp.piotrkrzysztof.dev/architecture/)

**Production Deployment** — Multi-service Docker Compose stack, CI/CD with GitHub Actions, Trivy security scanning, signed container images. [Deployment guide](https://docs.qp.piotrkrzysztof.dev/deployment/)

**Monitoring** — Prometheus metrics, Grafana dashboards, energy convergence tracking, scientific reference validation against peer-reviewed data for 8+ molecules. [Monitoring setup](https://docs.qp.piotrkrzysztof.dev/monitoring/)

## Python API

```python
from quantum_pipeline.runners.vqe_runner import VQERunner

backend = VQERunner.default_backend()
runner = VQERunner(
    filepath='data/molecules.json',
    basis_set='sto3g',
    max_iterations=100,
    convergence_threshold=1e-6,
    optimizer='L-BFGS-B',
    ansatz_reps=3,
)
runner.run(backend)
```

## Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌───────────────┐
│  Quantum        │───>│ Apache Kafka │───>│ Apache Spark  │
│  Pipeline (VQE) │    │ (Streaming)  │    │ (Processing)  │
└─────────────────┘    └──────────────┘    └───────────────┘
                              │                     │
                              v                     v
┌─────────────────┐    ┌──────────────┐    ┌───────────────┐
│ Apache Airflow  │    │   Schema     │    │    Apache     │
│ (Orchestration) │    │   Registry   │    │    Iceberg    │
└─────────────────┘    └──────────────┘    └───────────────┘
         │                    │                     │
         └────────────────────┼─────────────────────┘
                              v
                    ┌──────────────────┐
                    │  MinIO Storage   │
                    └──────────────────┘
```

For detailed architecture documentation, see the [system design](https://docs.qp.piotrkrzysztof.dev/architecture/system-design/) and [data flow](https://docs.qp.piotrkrzysztof.dev/architecture/data-flow/) pages.

## Contributing

This project is not currently open for contributions as it is a university project. Feel free to fork it and make your own version.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

- **Email:** [piotr@codextechnologies.org](mailto:piotr@codextechnologies.org)
- **GitHub:** [straightchlorine](https://github.com/straightchlorine)
- **Codeberg:** [piotrkrzysztof](https://codeberg.org/piotrkrzysztof/quantum-pipeline)
