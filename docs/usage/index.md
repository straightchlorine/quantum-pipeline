# Usage

This section covers configuring and running VQE simulations with the Quantum
Pipeline. The CLI entry point is `quantum-pipeline`.

```mermaid
graph LR
    A[CLI Flags] -->|Override| C[Defaults]

    style A fill:#1a73e8,color:#fff,stroke:#1557b0
    style C fill:#34a853,color:#fff,stroke:#2d8f47
```

CLI flags override the built-in defaults in
[`defaults.py`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/configs/defaults.py).
A config file passed with `--load` is read and validated but its values are not
currently merged into the run, so the run still uses CLI flags and defaults. See
[Configuration](configuration.md) for details.

## Section Guide

<div class="grid cards" markdown>

-   :material-cog-outline:{ .lg .middle } **Configuration**

    ---

    CLI flag reference, parameter categories, validation rules, and the
    save/load workflow for every flag `quantum-pipeline` accepts.

    [:octicons-arrow-right-24: Configuration](configuration.md)

-   :material-tune-vertical:{ .lg .middle } **Optimizers**

    ---

    The 8 configured classical optimizers (L-BFGS-B, COBYLA, SLSQP, and
    5 generic-config ones), their behavior modes, default iteration
    budgets, and selection guidelines.

    [:octicons-arrow-right-24: Optimizers](optimizers.md)

-   :material-chip:{ .lg .middle } **Simulation Methods**

    ---

    Qiskit Aer backend methods - statevector, density_matrix,
    tensor_network, and others. GPU compatibility, memory scaling,
    and selection guidance.

    [:octicons-arrow-right-24: Simulation Methods](simulation-methods.md)

-   :material-code-tags:{ .lg .middle } **Examples**

    ---

    Copy-paste CLI recipes for common workflows: quick tests, GPU runs,
    Kafka streaming, convergence mode, config save/load, and thesis
    reproduction.

    [:octicons-arrow-right-24: Examples](examples.md)

</div>
