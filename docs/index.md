# Quantum Pipeline

<p align="center">
  <img src="assets/banner.svg" alt="Quantum Pipeline" width="420">
</p>

<p align="center">
  <a href="https://pypi.org/project/quantum-pipeline/">
    <img src="https://badge.fury.io/py/quantum-pipeline.svg" alt="PyPI version">
  </a>
  <a href="https://pypi.org/project/quantum-pipeline/">
    <img src="https://img.shields.io/pypi/dm/quantum-pipeline" alt="PyPI Downloads per Month">
  </a>
  <a href="https://pepy.tech/project/quantum-pipeline">
    <img src="https://static.pepy.tech/badge/quantum-pipeline" alt="PyPI Total Downloads">
  </a>
  <a href="https://pypi.org/project/quantum-pipeline/">
    <img src="https://img.shields.io/pypi/pyversions/quantum-pipeline.svg" alt="Python Versions">
  </a>
  <br>
  <a href="https://hub.docker.com/r/straightchlorine/quantum-pipeline">
    <img src="https://img.shields.io/docker/pulls/straightchlorine/quantum-pipeline.svg" alt="Docker Pulls">
  </a>
  <a href="https://hub.docker.com/r/straightchlorine/quantum-pipeline">
    <img src="https://img.shields.io/docker/image-size/straightchlorine/quantum-pipeline/latest" alt="Docker Image Size">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
  </a>
  <a href="https://github.com/straightchlorine/quantum-pipeline">
    <img src="https://img.shields.io/github/stars/straightchlorine/quantum-pipeline.svg" alt="GitHub Stars">
  </a>
</p>

## Overview

Quantum Pipeline is a framework for running quantum algorithms. Currently, only
the **Variational Quantum Eigensolver (VQE)** is implemented. It combines quantum
and classical computing to estimate the ground-state energy of molecular systems.

The framework handles algorithm orchestration, parametrization, monitoring, and
data visualization. Simulation results can be streamed via **Apache Kafka** for
real-time processing and transformed into ML features using **Apache Spark**.

It started as a Bachelor of Engineering thesis project at the DSW University of
Lower Silesia and is continued as a Master of Engineering thesis project. It is
still a work in progress.

<figure>
  <img src="https://qp-docs.codextechnologies.org/mkdocs/gpu_quantum_pipeline_service.png"
       alt="GPU-accelerated quantum pipeline service architecture">
  <figcaption>Overview of the GPU-accelerated quantum pipeline service architecture.</figcaption>
</figure>

## Quick Links

<div class="grid cards" markdown>

-   **Getting Started**

    ---

    Install Quantum Pipeline and run your first VQE simulation in minutes

    [Installation Guide →](getting-started/installation.md)

-   **Configuration**

    ---

    Learn about optimizers, ansatz types, initialization strategies, and parameter tuning

    [Usage Guide →](usage/index.md)

-   **Architecture**

    ---

    Understand the system design and data flow

    [Architecture Docs →](architecture/index.md)

-   **Deployment**

    ---

    Deploy with Docker, enable GPU acceleration, configure environments

    [Deployment Guide →](deployment/index.md)

</div>

## Links related to the project

- **GitHub**: [straightchlorine/quantum-pipeline](https://github.com/straightchlorine/quantum-pipeline)
- **Codeberg (mirror)**: [piotrkrzysztof/quantum-pipeline](https://codeberg.org/piotrkrzysztof/quantum-pipeline)
- **Docker Hub**: [straightchlorine/quantum-pipeline](https://hub.docker.com/r/straightchlorine/quantum-pipeline)
- **PyPI**: [quantum-pipeline](https://pypi.org/project/quantum-pipeline/)
- **Issues**: [Report bugs or request features](https://github.com/straightchlorine/quantum-pipeline/issues)

!!! info "Thesis project"
    This project began as a Bachelor of Engineering thesis at the **DSW University of
    Lower Silesia**, focusing on GPU-accelerated quantum simulation and data engineering
    for quantum computing workflows. It is continued as a Master of Engineering thesis.
