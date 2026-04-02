# Usage

This section covers configuring and running VQE simulations with the Quantum
Pipeline. The CLI entry point is `quantum-pipeline`.

```mermaid
graph LR
    A[CLI Arguments] -->|Override| B[Config File --load]
    B -->|Fallback| C[Defaults]

    style A fill:#1a73e8,color:#fff,stroke:#1557b0
    style B fill:#e8710a,color:#fff,stroke:#c45e08
    style C fill:#34a853,color:#fff,stroke:#2d8f47
```

Configuration resolves in three layers. CLI flags override loaded config files,
which override the built-in defaults in
[`defaults.py`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/configs/defaults.py).

## Section Guide

<div class="grid cards" markdown>

-   :material-cog-outline:{ .lg .middle } **Configuration**

    ---

    Complete CLI flag reference, parameter categories, validation rules,
    and save/load workflows. The single source of truth for every flag
    `quantum-pipeline` accepts.

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
