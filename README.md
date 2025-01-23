# Quantum Pipeline

## Overview

The Quantum Pipeline project is an extensible framework designed for exploring Variational Quantum Eigensolver (VQE) algorithms. It combines quantum and classical computing to estimate the ground-state energy of molecular systems.

The framework provides modules to handle algorithm orchestration, prametrising it as well as monitoring and data visualization. Data is organised in extensible dataclasses, which
can also be streamed via Kafka for further processing.

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
- **Containerized Execution:** Deploy as a Docker container.

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
├── stream/               # Kafka streaming and messaging utilities
├── structures/           # Quantum and classical data structures
├── utils/                # Utility functions (logging, visualization, etc.)
├── visual/               # Visualization tools for molecules and operators
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

   or

   ```bash
   docker build .
   ```

---

## Usage

### 1. Prepare Input Data

Molecules should be defined like this:

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

Defaults for each option can be found in `configs/defaults.py` and the help message (`python quantum_pipeline.py -h`). Other available parameters include:

- `-f FILE, --file FILE`: Path to the molecule data file (required).
- `-b BASIS, --basis BASIS`: Specify the basis set for the simulation.
- `--local`: Use a local quantum simulator instead of IBM Quantum.
- `--min-qubits MIN_QUBITS`: Specify the minimum number of qubits required.
- `--max-iterations MAX_ITERATIONS`: Set the maximum number of VQE iterations.
- `--optimizer OPTIMIZER`: Choose from a variety of optimization algorithms.
- `--output-dir OUTPUT_DIR`: Specify the directory for storing output files.
- `--log-level {DEBUG,INFO,WARNING,ERROR}`: Set the logging level.
- `--shots SHOTS`: Number of shots for quantum circuit execution.
- `--optimization-level {0,1,2,3}`: Circuit optimization level.
- `--report`: Generate a PDF report after simulation.
- `--kafka`: Stream data to Apache Kafka for real-time processing.

### Example Configurations

Basic configuration (utilizes the `defaults.py` config) emphasizes performance over accuracy:
```bash
python quantum_pipeline.py -f data/molecules.json
```

Configuration with custom parameters:
```bash
python quantum_pipeline.py -f data/molecule.json -b cc-pvdz --max-iterations 200 --optimizer L-BFGS-B --shots 2048 --report
```

### 3. Kafka Integration

Enable streaming to Apache Kafka via `--kafka` parameter:
```bash
python quantum_pipeline.py -f data/molecule.json --kafka
```

---

## Examples

### Python API

The framework can be used programmatically:

```python
from quantum_pipeline.runners.vqe_runner import VQERunner

backend = VQERunner.default_backend()
runner = VQERunner(
    filepath='data/molecules.json',
    basis_set='sto3g',
    max_iterations=1,
    convergence_threshold=1e-6,
    optimizer='COBYLA',
    ansatz_reps=3
)
runner.run(backend)
```

### Docker Example

Run the pipeline in a Docker container:
```bash
docker run -v $(pwd)/data:/app/data quantum_pipeline:latest \
    python quantum_pipeline.py --file /app/data/molecule.json --basis sto-3g
```

### Example KafkaConsumer

You can test the Kafka integration with a simple consumer like this:

```python
from kafka import KafkaConsumer
from quantum_pipeline.stream.serialization.interfaces.vqe import VQEDecoratedResultInterface

class KafkaMessageConsumer:
    def __init__(self, topic='vqe_results', bootstrap_servers='localhost:9092'):
        self.deserializer = VQEDecoratedResultInterface()
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            value_deserializer=self.deserializer.from_avro_bytes,
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            group_id='vqe_consumer_group'
        )

    def consume_messages(self):
        try:
            for message in self.consumer:
                try:
                    # Process the message
                    decoded_message = message.value
                    yield decoded_message
                except Exception as e:
                    print(f"Error processing message: {str(e)}")
                    continue
        except Exception as e:
            print(f"Error in consumer: {str(e)}")
        finally:
            self.consumer.close()
```

Then you can use the consumer like this:
```python
consumer = KafkaMessageConsumer()
for msg in consumer.consume_messages():
    print(f"Received message: {msg}")
``````
---

## Contributing

For now, this project is not open for contributions since it is a university project, but feel free to fork it and make your own version.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## Contact

For questions or support, please reach out to:
- **Email:** piotrlis555@gmail.com
- **GitHub:** [straightchlorine](https://github.com/straightchlorine)
