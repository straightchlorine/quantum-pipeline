import time

import numpy as np
from qiskit_nature.second_q.drivers.pyscfd.pyscfdriver import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper

from quantum_pipeline.configs.parsing.backend_config import BackendConfig
from quantum_pipeline.drivers.basis_sets import validate_basis_set
from quantum_pipeline.drivers.molecule_loader import load_molecule
from quantum_pipeline.report.report_generator import ReportGenerator
from quantum_pipeline.runners.runner import Runner
from quantum_pipeline.solvers.vqe_solver import VQESolver
from quantum_pipeline.stream.kafka_interface import ProducerConfig, VQEKafkaProducer
from quantum_pipeline.structures.vqe_observation import VQEDecoratedResult
from quantum_pipeline.visual.ansatz import AnsatzViewer


class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()

    @property
    def elapsed(self):
        if self.start_time is None or self.end_time is None:
            raise ValueError('Timer has not finished yet.')
        return self.end_time - self.start_time


class VQERunner(Runner):
    """Class to handle the ground energy finding process."""

    def __init__(  # noqa: C901
        self,
        filepath,
        basis_set='sto3g',
        max_iterations=100,
        convergence_threshold=None,
        optimizer='COBYLA',
        ansatz_reps=3,
        default_shots=1024,
        report=False,
        kafka=False,
        kafka_bootstrap_servers='localhost:9092',
        kafka_topic='vqe_results',
        kafka_retries=3,
        kafka_internal_retries=5,
        kafka_acks='all',
        kafka_timeout=10,
        kafka_config: ProducerConfig | None = None,
        backend_type='local',
        backend_optimization_level=3,
        backend_min_num_qubits=None,
        filters=None,
        backend_config: BackendConfig | None = None,
    ):
        super().__init__()
        self.filepath = filepath
        self.basis_set = basis_set
        self.max_iterations = max_iterations
        self.ansatz_reps = ansatz_reps
        self.optimizer = optimizer
        self.default_shots = default_shots
        self.convergence_threshold = convergence_threshold
        self.optimization_level = backend_optimization_level

        self.report = report
        if self.report:
            self.report_gen = ReportGenerator()

        self.kafka = kafka
        if self.kafka and kafka_config is not None:
            self.kafka_config = kafka_config
        elif self.kafka and kafka_config is None:
            try:
                self.kafka_config = ProducerConfig(
                    servers=kafka_bootstrap_servers,
                    topic=kafka_topic,
                    retries=kafka_retries,
                    kafka_retries=kafka_internal_retries,
                    acks=kafka_acks,
                    timeout=kafka_timeout,
                )
            except Exception as e:
                self.logger.error(
                    f'Unable to create ProducerConfig, ensure required parameters are passed to the VQERunner instance: {e}'
                )

        def isAnyBackendOptionSet():
            return (
                backend_type is not None
                or backend_optimization_level is not None
                or backend_min_num_qubits is not None
                or filters is not None
            )

        if backend_config is not None:
            self.backend_config = backend_config
        elif backend_config is None and isAnyBackendOptionSet():
            try:
                self.backend_config = BackendConfig(
                    local=True if backend_type == 'local' else False,
                    optimization_level=backend_optimization_level,
                    min_num_qubits=backend_min_num_qubits,
                    filters=filters,
                )
            except Exception as e:
                self.logger.error(
                    f'Unable to create BackendConfig, ensure required parameters are passed to the VQERunner instance: {e}'
                )
        else:
            try:
                self.backend_config = BackendConfig.default_backend_config()
            except Exception as e:
                self.logger.error(
                    (
                        'Unable to create default backend_config. '
                        + f'ensure required parameters are passed to the VQERunner instance: {e}'
                    )
                )

        self.run_results = []

    @staticmethod
    def default_backend():
        return BackendConfig.default_backend_config()

    def load_molecules(self):
        """Load molecule data and validate the basis set."""
        self.logger.info(f'Loading molecule data from {self.filepath}')
        molecules = load_molecule(self.filepath)
        validate_basis_set(self.basis_set)
        return molecules

    def provide_hamiltonian(self, molecule):
        """Generate the second quantized operator."""
        driver = PySCFDriver.from_molecule(molecule, basis=self.basis_set)
        problem = driver.run()
        return problem.second_q_ops()[0]

    def runVQE(self, molecule, backend_config: BackendConfig):
        """Prepare and run the VQE algorithm."""

        self.logger.info('Generating hamiltonian based on the molecule...')
        with Timer() as t:
            second_q_op = self.provide_hamiltonian(molecule)

        self.hamiltonian_time = t.elapsed
        self.logger.info(f'Hamiltonian generated in {t.elapsed:.6f} seconds.')

        self.logger.info('Mapping fermionic operator to qubits')
        with Timer() as t:
            qubit_op = JordanWignerMapper().map(second_q_op)

        self.mapping_time = t.elapsed
        self.logger.info(f'Problem mapped to qubits in {t.elapsed:.6f} seconds.')

        self.logger.info('Running VQE procedure...')
        with Timer() as t:
            result = VQESolver(
                qubit_op=qubit_op,
                backend_config=backend_config,
                max_iterations=self.max_iterations,
                optimization_level=self.optimization_level,
                optimizer=self.optimizer,
                ansatz_reps=self.ansatz_reps,
                default_shots=self.default_shots,
                convergence_threshold=self.convergence_threshold,
            ).solve()

        self.vqe_time = t.elapsed
        self.logger.info(f'VQE procedure completed in {t.elapsed:.6f} seconds')

        return result

    def run(self):
        self.molecules = self.load_molecules()

        for id, molecule in enumerate(self.molecules):
            self.logger.info(f'Processing molecule {id + 1}:\n\n{molecule}\n')
            result = self.runVQE(molecule, self.backend_config)

            total_time = np.float64(self.hamiltonian_time + self.mapping_time + self.vqe_time)
            self.logger.info(f'Result provided in {total_time:.6f} seconds.')

            decorated_result = VQEDecoratedResult(
                vqe_result=result,
                molecule=molecule,
                basis_set=self.basis_set,
                id=id,
                hamiltonian_time=np.float64(self.hamiltonian_time),
                mapping_time=np.float64(self.mapping_time),
                vqe_time=np.float64(self.vqe_time),
                total_time=total_time,
            )
            self.run_results.append(decorated_result)
            self.logger.debug('Appended run information to the result.')

            if self.kafka:
                producer = VQEKafkaProducer(self.kafka_config)
                producer.send_result(decorated_result)

            if self.report:
                self.logger.info(f'Generating report for molecule {id + 1}...')
                self.generate_report()
                self.logger.info(f'Generated the report for molecule {id + 1}.')

        self.logger.info('All molecules processed.')

        if self.report:
            self.report_gen.generate_report()

    def generate_report(self):
        for result in self.run_results:
            self.report_gen.add_header('Structure of the molecule in 3D')
            self.report_gen.add_molecule_plot(result.molecule)

            self.report_gen.add_insight('Energy analysis', 'Results of VQE algorihtm:')
            self.report_gen.add_metrics(
                {
                    'Minimum Energy': str(round(result.vqe_result.minimum, 4)) + ' Heartree',
                    'Optimizer': result.vqe_result.initial_data.optimizer,
                    'Iterations': len(result.vqe_result.iteration_list),
                    'Ansatz Repetitions': result.vqe_result.initial_data.ansatz_reps,
                    'Basis set': result.basis_set,
                }
            )

            self.report_gen.new_page()
            self.report_gen.add_header('Real coefficients of the operators')

            self.report_gen.add_operator_coefficients_plot(
                result.vqe_result.initial_data.hamiltonian, result.molecule.symbols
            )

            self.report_gen.add_header('Complex coefficients of the operators')
            self.report_gen.add_complex_operator_coefficients_plot(
                result.vqe_result.initial_data.hamiltonian, result.molecule.symbols
            )

            self.report_gen.new_page()
            self.report_gen.add_insight(
                'Energy convergence of the algorithm',
                'Energy levels for each iteration:',
            )

            self.report_gen.add_convergence_plot(
                result.vqe_result.iteration_list, result.molecule.symbols
            )
            self.report_gen.add_insight(
                'Ansatz circuit for the molecule was generated in the project directory.',
                'They often take too much space to display. See graph/ansatz and graph/ansatz_decomposed.',
            )
            self.report_gen.new_page()

            AnsatzViewer(
                result.vqe_result.initial_data.ansatz, result.molecule.symbols
            ).save_circuit()
