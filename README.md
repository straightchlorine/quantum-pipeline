<p align="center">
  <img src="https://qp-docs.codextechnologies.org/mkdocs/banner.svg" alt="Quantum Pipeline" width="350">
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

A framework for running quantum algorithms, with optional Kafka streaming, Spark processing, and Airflow orchestration. Currently implements the Variational Quantum Eigensolver (VQE) for ground-state energy estimation. Built as an engineering thesis project.

## Quick Start

```bash
pip install quantum-pipeline
quantum-pipeline -f molecules.json -b sto3g --max-iterations 100 --optimizer L-BFGS-B
```

Or with Docker:

```bash
docker pull straightchlorine/quantum-pipeline:cpu
docker run --rm straightchlorine/quantum-pipeline:cpu -f data/molecules.json -b sto3g --max-iterations 100
```

See the [installation guide](https://docs.qp.piotrkrzysztof.dev/getting-started/installation/) for detailed setup, including GPU acceleration and full platform deployment.

## Features

**Quantum Computing** -- VQE execution with multiple optimizers (L-BFGS-B, COBYLA, SLSQP, and others), configurable ansatz circuits (EfficientSU2, RealAmplitudes, ExcitationPreserving), parameter initialization strategies (random or Hartree-Fock based), multiple basis sets (sto-3g, 6-31g, cc-pVDZ), and GPU acceleration via CUDA. [Learn more](https://docs.qp.piotrkrzysztof.dev/scientific/vqe-algorithm/)

**Data Platform** -- Real-time Kafka streaming with Avro serialization, Spark-based ML feature engineering, Airflow workflow orchestration. [Architecture overview](https://docs.qp.piotrkrzysztof.dev/architecture/)

**ML Pipeline** -- Convergence prediction and energy estimation models trained on VQE experiment data. Includes preprocessing, experiment tracking, and a dedicated Docker Compose stack (`just ml-up` / `just ml-down`).

**Monitoring** -- Prometheus metrics export, Grafana dashboards, resource tracking. Configurable via environment variables (`MONITORING_ENABLED`, `PUSHGATEWAY_URL`, `MONITORING_INTERVAL`, `MONITORING_EXPORT_FORMAT`) or CLI flags. [Monitoring setup](https://docs.qp.piotrkrzysztof.dev/monitoring/)

**Deployment** -- Docker images for CPU and GPU (`quantum-pipeline:cpu`, `quantum-pipeline:gpu`). The GPU image on Docker Hub is built for Ampere (RTX 30xx) - rebuild with `CUDA_ARCH=6.1` for Pascal or `8.9` for Ada Lovelace. Sample molecule files are included in the image under `data/`. Multi-service Docker Compose stack for the full platform. [Deployment guide](https://docs.qp.piotrkrzysztof.dev/deployment/)

## Python API

```python
from quantum_pipeline.runners.vqe_runner import VQERunner

runner = VQERunner(
    filepath='data/molecules.json',
    basis_set='sto3g',
    max_iterations=100,
    optimizer='L-BFGS-B',
    ansatz_reps=3,
)
runner.run()
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

- **Email:** [piotr@codextechnologies.org](mailto:piotr@codextechnologies.org)
- **GitHub:** [straightchlorine](https://github.com/straightchlorine)
- **Codeberg:** [piotrkrzysztof](https://codeberg.org/piotrkrzysztof/quantum-pipeline)
