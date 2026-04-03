# Configuration Reference

Complete reference for all CLI flags accepted by `quantum-pipeline`.

## Configuration Layers

The pipeline resolves settings in three layers (highest priority first):

| Layer | Source | Purpose |
|-------|--------|---------|
| CLI arguments | Flags passed to `quantum-pipeline` | Runtime overrides |
| Config file | Loaded via `--load config.json` | Saved experiment configs |
| Defaults | [`defaults.py`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/configs/defaults.py) | Built-in fallback values |

Static settings (supported optimizers, basis sets, simulation methods, output
directories) live in
[`settings.py`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/configs/settings.py)
and are not meant to be changed at runtime.

Parsed arguments are assembled into typed dataclasses
([`BackendConfig`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/configs/module/backend.py),
[`ProducerConfig`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/configs/module/producer.py),
[`SecurityConfig`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/configs/module/security.py))
by
[`ConfigurationManager`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/configs/parsing/configuration_manager.py).

!!! note "Environment Variables"
    Some environment variables are supported for performance monitoring
    (`MONITORING_*`) and Docker deployments (`CONTAINER_TYPE`, `KAFKA_SERVERS`),
    but they do not participate in the three-layer hierarchy.

## Quick Reference

| Category | Flag | Type | Default | Description |
|----------|------|------|---------|-------------|
| Required | `--file` / `-f` | path | - | Molecule data file (JSON) |
| Required | `--molecule-index` | int | `None` | Process single molecule by 0-based index |
| Simulation | `--basis` / `-b` | choice | `sto3g` | Basis set (`sto3g`, `6-31g`, `cc-pvdz`) |
| Simulation | `--ansatz` | choice | `EfficientSU2` | Ansatz type (`EfficientSU2`, `RealAmplitudes`, `ExcitationPreserving`) |
| Simulation | `--ansatz-reps` / `-ar` | int | `2` | Ansatz circuit repetitions |
| Simulation | `--ibm` | flag | `false` | Use IBM Quantum backend (disables local Aer) |
| Simulation | `--min-qubits` | int | `None` | Minimum qubit count (IBM only) |
| VQE | `--max-iterations` | int | `100` | Maximum VQE iterations |
| VQE | `--convergence` | flag | `false` | Enable convergence-based stopping |
| VQE | `--threshold` | float | `1e-6` | Convergence threshold (Hartree) |
| VQE | `--optimizer` | choice | `L-BFGS-B` | Optimization algorithm |
| VQE | `--init-strategy` | choice | `random` | Parameter init (`random`, `hf`) |
| VQE | `--seed` | int | `None` | Random seed for reproducibility |
| Output | `--output-dir` | path | `./gen` | Output directory |
| Output | `--log-level` | choice | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| Backend | `--shots` | int | `1024` | Circuit execution shots (must be positive) |
| Backend | `--optimization-level` | int | `3` | Circuit optimization level (0-3) |
| Features | `--report` | flag | `false` | Generate PDF report |
| Features | `--dump` | flag | `false` | Save config to JSON |
| Features | `--load` | path | - | Load config from JSON |
| Features | `--gpu` | flag | `false` | Enable GPU acceleration |
| Features | `--simulation-method` | choice | `statevector` | Aer simulation method |
| Features | `--noise` | string | `None` | Noise model backend name |
| Monitoring | `--enable-performance-monitoring` | flag | `false` | Enable resource monitoring |
| Monitoring | `--performance-interval` | int | `30` | Metrics collection interval (seconds) |
| Monitoring | `--performance-pushgateway` | string | `None` | Prometheus PushGateway URL |
| Monitoring | `--performance-export-format` | choice | `both` | Export format (`json`, `prometheus`, `both`) |
| Kafka | `--kafka` | flag | `false` | Enable Kafka streaming |
| Kafka | `--servers` | string | `localhost:9092` | Bootstrap servers |
| Kafka | `--topic` | string | `experiment.vqe` | Topic name |
| Kafka | `--retries` | string | `3` | Send retry attempts |
| Kafka | `--retry-delay` | string | `2` | Retry delay (seconds) |
| Kafka | `--internal-retries` | int | `0` | Kafka-internal retries (risk of duplicates) |
| Kafka | `--acks` | choice | `all` | Ack level (`0`, `1`, `all`) |
| Kafka | `--timeout` | int | `10` | Request timeout (seconds) |
| Security | `--ssl` | flag | `false` | Enable SSL/TLS for Kafka |
| Security | `--disable-ssl-check-hostname` | flag | `false` | Disable hostname verification (testing only) |
| Security | `--sasl-ssl` | flag | `false` | Enable SASL_SSL |
| Security | `--ssl-password` | string | `None` | SSL private key password |
| Security | `--ssl-dir` | path | `./secrets/` | SSL certificates directory |
| Security | `--ssl-cafile` | path | `None` | CA certificate file |
| Security | `--ssl-certfile` | path | `None` | Client certificate file |
| Security | `--ssl-keyfile` | path | `None` | Client private key file |
| Security | `--ssl-crlfile` | path | `None` | Certificate revocation list |
| Security | `--ssl-ciphers` | string | `None` | SSL cipher suite |
| Security | `--sasl-mechanism` | choice | - | SASL method (`PLAIN`, `GSSAPI`, `SCRAM-SHA-256`, `SCRAM-SHA-512`) |
| Security | `--sasl-plain-username` | string | `None` | SASL username |
| Security | `--sasl-plain-password` | string | `None` | SASL password |
| Security | `--sasl-kerberos-service-name` | string | `kafka` | Kerberos service name |
| Security | `--sasl-kerberos-domain-name` | string | `None` | Kerberos domain |

## Simulation Flags

### `--file` / `-f` (required)

Path to the molecule data file in JSON format. The file must contain an array
of molecule objects with `symbols`, `coords`, `multiplicity`, `charge`, `units`,
and `masses` fields.

```bash
quantum-pipeline --file data/molecules.json
```

### `--molecule-index`

Process only the molecule at this 0-based index. Without this flag, all
molecules in the file are processed sequentially.

```bash
quantum-pipeline -f molecules.json --molecule-index 1
```

### `--basis` / `-b`

Basis set for quantum chemistry calculations.

| Basis Set | Accuracy | Speed | Qubits (H\(_2\)O) |
|-----------|----------|-------|---------------|
| `sto3g` | Low | Very fast | 14 |
| `6-31g` | Medium | Medium | 26 |
| `cc-pvdz` | High | Slow | 58 |

### `--ansatz`

Ansatz circuit type for the parameterized quantum circuit.

| Ansatz | Description |
|--------|-------------|
| `EfficientSU2` | General-purpose, supports all init strategies (default) |
| `RealAmplitudes` | Real-valued rotations (Ry) only |
| `ExcitationPreserving` | Preserves particle number |

!!! warning "HF Initialization Compatibility"
    `--init-strategy hf` only works with `EfficientSU2`. Other ansatze fall
    back to random initialization with a warning.

### `--ansatz-reps` / `-ar`

Number of repetitions for the ansatz circuit. More reps means more parameters
and a more expressive circuit, but longer optimization. Default: `2`.

### `--ibm`

Switch from local Aer simulator to IBM Quantum backend. Requires
`IBM_RUNTIME_TOKEN` and `IBM_RUNTIME_INSTANCE` environment variables.
Cannot be used with `--gpu`.

### `--min-qubits`

Minimum qubit count for IBM backend selection. Only valid with `--ibm`.

## VQE Flags

### `--max-iterations`

Maximum number of VQE optimization iterations. Default: `100`.

The limit is enforced at two levels: the optimizer config sets `maxiter` in
the scipy options dict, and the `compute_energy()` callback raises
`MaxFunctionEvalsReachedError` as a hard stop.

### `--convergence` and `--threshold`

Enable convergence-based stopping. When `--convergence` is set, the optimizer
runs until the energy change between iterations falls below `--threshold`
(default `1e-6` Hartree).

| Threshold | Precision | Typical use |
|-----------|-----------|-------------|
| `1.6e-3` | Chemical accuracy (~1 kcal/mol) | Quick prototyping |
| `1e-6` | 1 microHartree | Standard runs |
| `1e-8` | 10 nanoHartree | High-precision research |

!!! note "Interaction with --max-iterations"
    If `--convergence` is enabled and `--max-iterations` is left at the default
    (100), the entry point sets `max_iterations` to `None` and uses the
    threshold instead. Passing both non-default values raises a `ValueError` at
    the solver level via
    [`OptimizerConfig`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/solvers/optimizer_config.py#L27).

### `--optimizer`

Optimization algorithm. The pipeline has 8 optimizers with dedicated configs
via `OptimizerConfigFactory` and 8 more accepted by the CLI but without custom
handling. See [Optimizers](optimizers.md) for the full comparison.

### `--init-strategy`

| Strategy | Description |
|----------|-------------|
| `random` | Uniform random in [0, 2pi] (default) |
| `hf` | Hartree-Fock pre-optimized starting point |

The `hf` strategy can reduce iterations by 30-40% and helps avoid local minima,
especially with `cc-pvdz`. Only works with `EfficientSU2`.

### `--seed`

Random seed for reproducible parameter initialization. Default: `None` (random).

## Backend Flags

### `--shots`

Number of measurement samples per circuit execution. Must be a positive
integer. Default: `1024`.

### `--optimization-level`

Qiskit transpilation optimization level. Higher levels produce shorter circuits
at the cost of transpile time. Default: `3`.

| Level | Description |
|-------|-------------|
| `0` | No optimization |
| `1` | Light optimization |
| `2` | Medium optimization |
| `3` | Heavy optimization (recommended) |

### `--gpu`

Enable GPU acceleration. Works with `statevector`, `density_matrix`, `unitary`,
and `tensor_network` simulation methods. GPU options (device, cuStateVec,
memory limits) are configured in
[`defaults.py`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/configs/defaults.py#L14).

### `--simulation-method`

Aer backend simulation method. Default: `statevector`. See
[Simulation Methods](simulation-methods.md) for the full comparison.

| Method | GPU | Memory | Accuracy |
|--------|-----|--------|----------|
| `automatic` | Partial | Varies | Varies |
| `statevector` | Yes | High | Exact |
| `density_matrix` | Yes | Very high | Exact |
| `stabilizer` | No | Low | Exact (Clifford only) |
| `extended_stabilizer` | No | Medium | Approximate |
| `matrix_product_state` | No | Low | Approximate |
| `unitary` | Yes | Very high | Exact |
| `superop` | No | Extreme | Exact |
| `tensor_network` | Yes (required) | Medium | Exact |

### `--noise`

Name of a noise model backend (e.g., `ibmq_manila`). Best used with
`density_matrix` simulation method.

## Output Flags

### `--output-dir`

Directory for output files, graphs, and reports. Default: `./gen`.

### `--log-level`

Logging verbosity. Choices: `DEBUG`, `INFO`, `WARNING`, `ERROR`. Default: `INFO`.

### `--report`

Generate a PDF report with molecular structure visualizations, Hamiltonian
coefficients, energy convergence plots, and final results. Output goes to
`gen/`. See an
[example report](https://qp-docs.codextechnologies.org/mkdocs/quantum_report.pdf).
Ansatz circuit diagrams are saved separately in `gen/graphs/` because they are
often too large for the PDF
([example](https://qp-docs.codextechnologies.org/mkdocs/ansatz_H_H.png),
[decomposed](https://qp-docs.codextechnologies.org/mkdocs/ansatz_decomposed_H_H.png)).

### `--dump`

Save the current configuration to a timestamped JSON file in `run_configs/`.
Cannot be used with `--load`.

### `--load`

Load configuration from a previously saved JSON file. CLI flags override loaded
values. Cannot be used with `--dump`.

```bash
# Load config, override optimizer
quantum-pipeline --load run_configs/config_20250615.json --optimizer COBYLA
```

## Monitoring Flags

### `--enable-performance-monitoring`

Enable CPU, GPU, memory, and VQE-specific metrics collection.

### `--performance-interval`

Collection interval in seconds. Default: `30`.

### `--performance-pushgateway`

Prometheus PushGateway URL for metrics export.

### `--performance-export-format`

Export format: `json`, `prometheus`, or `both` (default).

Monitoring defaults are also defined in
[`settings.py`](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/configs/settings.py#L69):

| Setting | Default |
|---------|---------|
| `MONITORING_ENABLED` | `False` |
| `PUSHGATEWAY_URL` | `http://localhost:9091` |
| `MONITORING_INTERVAL` | `10` seconds |
| `MONITORING_EXPORT_FORMAT` | `['prometheus']` |

## Kafka Flags

All Kafka flags require `--kafka` to be set. Without it, Kafka parameters are
rejected by the argument parser.

### `--kafka`

Enable streaming VQE results to Kafka. Results are serialized in Avro format
with automatic Schema Registry integration.

### `--servers`

Bootstrap servers. Default: `localhost:9092`. Comma-separated for multiple
brokers.

### `--topic`

Topic name. Default: `experiment.vqe`. The pipeline appends experiment metadata
(molecule, iterations, basis set, backend) to form the full topic name.

### `--acks`

Acknowledgment level. Default: `all`.

| Level | Durability | Speed |
|-------|------------|-------|
| `0` | None | Fastest |
| `1` | Leader only | Fast |
| `all` | All replicas | Safest |

### `--retries`, `--retry-delay`, `--internal-retries`, `--timeout`

Retry and timeout settings. `--internal-retries` controls Kafka's own retry
mechanism, which introduces risk of duplicate messages if set above `0`.

## Security Flags

Security flags require `--kafka` to be enabled.

### SSL/TLS

Enable with `--ssl`. Provide certificates in one of two ways:

- `--ssl-dir` pointing to a directory with `ca.crt`, `client.crt`, `client.key`
- Individual paths via `--ssl-cafile`, `--ssl-certfile`, `--ssl-keyfile`

These two approaches are mutually exclusive. `--ssl-crlfile` and
`--ssl-ciphers` are optional. `--ssl-password` provides the private key
password if encrypted.

`--disable-ssl-check-hostname` disables hostname verification. Testing only.

### SASL

Enable with `--sasl-ssl`. Requires `--sasl-mechanism`.

| Mechanism | Required flags |
|-----------|----------------|
| `PLAIN` | `--sasl-plain-username`, `--sasl-plain-password` |
| `SCRAM-SHA-256` | `--sasl-plain-username`, `--sasl-plain-password` |
| `SCRAM-SHA-512` | `--sasl-plain-username`, `--sasl-plain-password` |
| `GSSAPI` | `--sasl-kerberos-service-name`, optionally `--sasl-kerberos-domain-name` |

PLAIN/SCRAM credentials cannot be mixed with GSSAPI options.

## Validation Rules

The
[argument parser](https://codeberg.org/piotrkrzysztof/quantum-pipeline/src/branch/master/quantum_pipeline/configs/parsing/argparser.py#L398)
enforces these rules:

| Rule | Condition |
|------|-----------|
| `--dump` and `--load` are mutually exclusive | Cannot use together |
| `--min-qubits` requires `--ibm` | Local simulator ignores it |
| `--convergence` uses `--threshold` | Threshold defaults to `1e-6` if not set |
| `tensor_network` requires `--gpu` | CPU-only methods: statevector, automatic, etc. |
| Kafka params require `--kafka` | Changing servers/topic/acks without `--kafka` is an error |
| `--ssl` requires `--kafka` | SSL is for Kafka connections |
| `--sasl-mechanism` requires `--kafka` | SASL is for Kafka connections |
| `--ssl-dir` and individual SSL files are mutually exclusive | Choose one approach |

## Environment Variables

```bash
# Performance monitoring
export MONITORING_ENABLED=true
export PUSHGATEWAY_URL=http://localhost:9091
export MONITORING_INTERVAL=10
export MONITORING_EXPORT_FORMAT=json,prometheus

# IBM Quantum
export IBM_RUNTIME_TOKEN=your_token_here
export IBM_RUNTIME_INSTANCE=crn:v1:bluemix:public:quantum-computing:...
```

In Docker deployments, `KAFKA_SERVERS`, `CONTAINER_TYPE`, and `MAX_ITERATIONS`
are set by the container entrypoint. See
[Docker Compose](../deployment/docker-compose.md).
