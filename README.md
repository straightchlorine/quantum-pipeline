# Quantum Pipeline

## Overview

The Quantum Pipeline project is an extensible framework designed for exploring Variational Quantum Eigensolver (VQE) algorithms. It combines quantum and classical computing to estimate the ground-state energy of molecular systems. The framework provides modularity to handle molecule loading, Hamiltonian generation, quantum circuit design, and result visualization.

Currently, it offers VQE as its primary algorithm with basic functionality, but aims to evolve into a convenient tool for running various quantum algorithms.

---

## Features

- **Molecule Loading:** Load and validate molecular data from files.
- **Hamiltonian Preparation:** Generate second-quantized Hamiltonians for molecular systems.
- **Quantum Circuit Construction:** Create parameterized ansatz circuits with customizable repetitions.
- **VQE Execution:** Solve Hamiltonians using the VQE algorithm with support for various optimizers.
- **Visualization Tools:** Plot molecular structures, energy convergence, and operator coefficients.
- **Report Generation:** Automatically generate detailed reports for each processed molecule.
- **Kafka Integration:** Stream simulation results to Apache Kafka for real-time data processing.
- **Advanced Backend Options:** Customize simulation parameters such as qubit count, shot count, and optimization levels.
- **Contenerised:** Deployment as a Docker container for easy setup and execution.


---

## Directory Structure

```
quantum_pipeline/
├── configs/              # Configuration settings and argument parsers
├── drivers/              # Molecule loading and basis set validation
├── features/             # Quantum circuit and Hamiltonian features
├── mappers/              # Fermionic-to-qubit mapping implementations
├── report/               # Report generation utilities
├── runners/              # VQE execution logic
├── solvers/              # VQE solver implementations
├── utils/                # Utility functions (logging, visualization, etc.)
├── visual/               # Visualization tools for molecules and operators
├── Dockerfile            # Dockerfile for containerized execution
├── docker-compose.yaml   # Docker Compose file for multi-container setup
├── pyproject.toml        # Project configuration
├── requirements.txt      # Python dependencies
└── quantum_pipeline.py   # Main entry point
```

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/quantum_pipeline.git
   cd quantum_pipeline
   ```

2. **Set Up a Virtual Environment** (optional but recommended):
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Run in Docker**:
   ```bash
   docker-compose up --build
   ```

---

## Usage

### 1. Prepare Input Data
Place molecule file in the `data/` directory. The file should look like this:
```json
[
    {
        "symbols": ["H", "H"],
        "coords": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        "multiplicity": 1,
        "charge": 0,
        "units": "angstrom",
        "masses": [1.008, 1.008]
    },
    {
        "symbols": ["O", "H", "H"],
        "coords": [[0.0, 0.0, 0.0], [0.0, 0.757, 0.586], [0.0, -0.757, 0.586]],
        "multiplicity": 1,
        "charge": 0,
        "units": "angstrom",
        "masses": [15.999, 1.008, 1.008]
    }
]

```
### 2. Run the Pipeline
Run the main script to process molecules:
```bash
python quantum_pipeline.py -f data/molecule.json -b sto-3g --max-iterations 100 --optimizer COBYLA --report
```

Defaults for each option can be found in `configs/defaults.py`, other available parameters include:

`-f FILE, --file FILE`: Path to the molecule data file (required).
`-b BASIS, --basis BASIS`: Specify the basis set for the simulation.
`--local`: Use a local quantum simulator instead of IBM Quantum.
`--min-qubits MIN_QUBITS`: Specify the minimum number of qubits required.
`--max-iterations MAX_ITERATIONS`: Set the maximum number of VQE iterations.
`--optimizer OPTIMIZER`: Choose from a variety of optimization algorithms.
`--output-dir OUTPUT_DIR`: Specify the directory for storing output files.
`--log-level {DEBUG,INFO,WARNING,ERROR}`: Set the logging level.
`--shots SHOTS`: Number of shots for quantum circuit execution.
`--optimization-level {0,1,2,3}`: Circuit optimization level.
`--report`: Generate a PDF report after simulation.
`--kafka`: Stream data to Apache Kafka for real-time processing.

Help message can be displayed with:

```bash
python quantum_pipeline.py -h
```

### Example configurations

Basic configuration (utilises the `defaults.py` config) emphasises performance over accuracy:
```bash
python quantum_pipeline.py -f data/molecules.json
```

Configuration with custom parameters:
```bash
python quantum_pipeline.py -f data/molecule.json -b cc-pvdz --max-iterations 200 --optimizer L-BFGS-B --shots 2048 --report
```


### 3. Kafka integration
Enable Apache Kafka for streaming simulation results:
```bash
python quantum_pipeline.py -f data/molecule.json --kafka
```

---

## Examples

### Python API
The framework can be used programmatically:
```python
from quantum_pipeline.runners.vqe_runner import VQERunner
from quantum_pipeline.configs.argparser import BackendConfig

backend_config = BackendConfig(backend_type='qasm_simulator', shots=1024)

runner = VQERunner(
    filepath='data/molecule.json',
    basis_set='sto-3g',
    max_iterations=200,
    convergence_threshold=1e-6,
    optimizer='COBYLA',
    ansatz_reps=3
)

runner.run(backend_config)
```

### Docker Example
Run the pipeline in a Docker container:
```bash
docker run -v $(pwd)/data:/app/data quantum_pipeline:latest \
    python quantum_pipeline.py --file /app/data/molecule.json --basis sto-3g
```
---

## Contributing

For now this project is not open for contribution, since its a university project, but feel free to fork it and make your own version.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## Contact

For questions or support, please reach out to:
- **Email:** piotrlis555@gmail.com
- **GitHub:** [straightchlorine](https://github.com/straightchlorine)
