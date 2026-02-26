import os

import numpy as np
from qiskit_nature.second_q.drivers.pyscfd.pyscfdriver import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper

from quantum_pipeline.configs.module.backend import BackendConfig
from quantum_pipeline.configs.module.producer import ProducerConfig
from quantum_pipeline.configs.module.security import SecurityConfig
from quantum_pipeline.drivers.basis_sets import validate_basis_set
from quantum_pipeline.drivers.molecule_loader import load_molecule
from quantum_pipeline.monitoring import get_performance_monitor
from quantum_pipeline.monitoring.scientific_references import get_reference_database
from quantum_pipeline.report.report_generator import ReportGenerator
from quantum_pipeline.runners.runner import Runner
from quantum_pipeline.solvers.vqe_solver import VQESolver
from quantum_pipeline.stream.kafka_interface import VQEKafkaProducer
from quantum_pipeline.structures.vqe_observation import VQEDecoratedResult
from quantum_pipeline.utils.timer import Timer
from quantum_pipeline.visual.ansatz import AnsatzViewer


class VQERunner(Runner):
    """Class to handle the ground energy finding process."""

    def __init__(
        self,
        filepath,
        basis_set='sto3g',
        max_iterations=100,
        convergence_threshold=None,
        optimizer='COBYLA',
        ansatz_reps=3,
        default_shots=1024,
        seed=None,
        report=False,
        kafka=False,
        kafka_bootstrap_servers='localhost:9092',
        kafka_topic='vqe_results',
        kafka_retries=3,
        kafka_internal_retries=5,
        kafka_acks='all',
        kafka_timeout=10,
        kafka_config: ProducerConfig | None = None,
        security_config: SecurityConfig | None = None,
        backend_type='local',
        backend_optimization_level=3,
        backend_min_num_qubits=None,
        backend_config: BackendConfig | None = None,
        backend_filters=None,
        backend_simulation_method=None,
        backend_gpu=None,
        backend_noise=None,
        backend_gpu_opts=None,
    ):
        super().__init__()
        self.filepath = filepath
        self.basis_set = basis_set
        self.max_iterations = max_iterations
        self.ansatz_reps = ansatz_reps
        self.optimizer = optimizer
        self.default_shots = default_shots
        self.convergence_threshold = convergence_threshold
        self.seed = seed
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
                    security=security_config
                    if security_config is not None
                    else SecurityConfig.get_default(),
                )
            except Exception as e:
                self.logger.error(
                    f'Unable to create ProducerConfig, ensure required parameters are passed to the VQERunner instance: {e}'
                )

        def is_any_backend_option_set():
            return (
                backend_type is not None
                or backend_optimization_level is not None
                or backend_min_num_qubits is not None
                or backend_filters is not None
                or backend_gpu is not None
                or backend_noise is not None
                or backend_gpu_opts is not None
            )

        if backend_config is not None:
            self.backend_config = backend_config
        elif backend_config is None and is_any_backend_option_set():
            try:
                self.backend_config = BackendConfig(
                    local=backend_type == 'local',
                    optimization_level=backend_optimization_level,
                    min_num_qubits=backend_min_num_qubits,
                    filters=backend_filters,
                    gpu=backend_gpu,
                    simulation_method=backend_simulation_method,
                    gpu_opts=backend_gpu_opts,
                    noise=backend_noise,
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
                    'Unable to create default backend_config. '
                    f'ensure required parameters are passed to the VQERunner instance: {e}'
                )

        self.run_results = []

        # Initialize performance monitoring and reference database
        self.performance_monitor = get_performance_monitor()
        self.reference_db = get_reference_database()

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

    def run_vqe(self, molecule, backend_config: BackendConfig):
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
            solver = VQESolver(
                qubit_op=qubit_op,
                backend_config=backend_config,
                max_iterations=self.max_iterations,
                optimization_level=self.optimization_level,
                optimizer=self.optimizer,
                ansatz_reps=self.ansatz_reps,
                default_shots=self.default_shots,
                convergence_threshold=self.convergence_threshold,
                seed=self.seed,
            )
            result = solver.solve()

        self.vqe_time = t.elapsed
        self.logger.info(f'VQE procedure completed in {t.elapsed:.6f} seconds')

        return result

    def run(self):
        self.molecules = self.load_molecules()

        # Start performance monitoring if enabled
        with self.performance_monitor:
            for molecule_id, molecule in enumerate(self.molecules):
                self.logger.info(f'Processing molecule {molecule_id + 1}:\n\n{molecule}\n')

                # Set experiment context for monitoring
                self.performance_monitor.set_experiment_context(
                    molecule_id=molecule_id,
                    molecule_symbols=''.join(molecule.symbols),
                    basis_set=self.basis_set,
                    optimizer=self.optimizer,
                    max_iterations=self.max_iterations,
                    backend_type='GPU' if getattr(self.backend_config, 'gpu', False) else 'CPU',
                )

                # Collect performance snapshot before VQE
                performance_start = self.performance_monitor.collect_metrics_snapshot()

                result = self.run_vqe(molecule, self.backend_config)

                # Collect performance snapshot after VQE
                performance_end = self.performance_monitor.collect_metrics_snapshot()

                total_time = np.float64(self.hamiltonian_time + self.mapping_time + self.vqe_time)
                self.logger.info(f'Result provided in {total_time:.6f} seconds.')

                # Update experiment context with VQE results for Prometheus export
                self.performance_monitor.set_experiment_context(
                    total_time=float(total_time),
                    minimum_energy=float(result.minimum),
                    hamiltonian_time=float(self.hamiltonian_time),
                    mapping_time=float(self.mapping_time),
                    vqe_time=float(self.vqe_time),
                    iterations_count=len(result.iteration_list),
                    optimal_parameters_count=len(result.optimal_parameters),
                )

                # Calculate accuracy metrics against scientific references
                molecule_name = ''.join(molecule.symbols)
                accuracy_metrics = self.reference_db.calculate_accuracy_metrics(
                    molecule_name=molecule_name,
                    vqe_energy=float(result.minimum),
                    basis_set=self.basis_set,
                )

                # Log accuracy results
                if accuracy_metrics['reference_available']:
                    self.logger.info(f'Accuracy assessment for {molecule_name}:')
                    self.logger.info(f'  VQE Energy: {result.minimum:.6f} Ha')
                    self.logger.info(
                        f'  Reference: {accuracy_metrics["reference_energy_hartree"]:.6f} Ha ({accuracy_metrics["reference_method"]})'
                    )
                    self.logger.info(
                        f'  Error: {accuracy_metrics["energy_error_millihartree"]:.3f} mHa'
                    )
                    self.logger.info(
                        f'  Accuracy Score: {accuracy_metrics["accuracy_score"]:.1f}/100'
                    )
                    self.logger.info(
                        f'  Chemical Accuracy: {"✓" if accuracy_metrics["within_chemical_accuracy"] else "✗"}'
                    )

                # Export VQE metrics immediately to Prometheus with full context and accuracy
                vqe_metrics_data = {
                    'container_type': os.getenv('CONTAINER_TYPE', 'unknown'),
                    'molecule_id': molecule_id,
                    'molecule_symbols': molecule_name,
                    'basis_set': self.basis_set,
                    'optimizer': self.optimizer,
                    'backend_type': 'GPU' if getattr(self.backend_config, 'gpu', False) else 'CPU',
                    'total_time': float(total_time),
                    'hamiltonian_time': float(self.hamiltonian_time),
                    'mapping_time': float(self.mapping_time),
                    'vqe_time': float(self.vqe_time),
                    'minimum_energy': float(result.minimum),
                    'iterations_count': len(result.iteration_list),
                    'optimal_parameters_count': len(result.optimal_parameters),
                    # Accuracy metrics
                    'reference_energy': accuracy_metrics.get('reference_energy_hartree', 0),
                    'energy_error_hartree': accuracy_metrics.get('energy_error_hartree', 0),
                    'energy_error_millihartree': accuracy_metrics.get(
                        'energy_error_millihartree', 0
                    ),
                    'accuracy_score': accuracy_metrics.get('accuracy_score', 0),
                    'within_chemical_accuracy': 1
                    if accuracy_metrics.get('within_chemical_accuracy', False)
                    else 0,
                }
                self.performance_monitor.export_vqe_metrics_immediate(vqe_metrics_data)

                decorated_result = VQEDecoratedResult(
                    vqe_result=result,
                    molecule=molecule,
                    basis_set=self.basis_set,
                    molecule_id=molecule_id,
                    hamiltonian_time=np.float64(self.hamiltonian_time),
                    mapping_time=np.float64(self.mapping_time),
                    vqe_time=np.float64(self.vqe_time),
                    total_time=total_time,
                    performance_start=performance_start
                    if self.performance_monitor.is_enabled()
                    else None,
                    performance_end=performance_end
                    if self.performance_monitor.is_enabled()
                    else None,
                )
                self.run_results.append(decorated_result)
                self.logger.debug('Appended run information to the result.')

                if self.kafka:
                    try:
                        self.producer = VQEKafkaProducer(self.kafka_config)

                        try:
                            self.producer.send_result(decorated_result)
                        except Exception as e:
                            self.logger.error('Unable to send the result to the Kafka broker.')
                            self.logger.debug(f'Error: {e}')

                    except Exception as e:
                        self.logger.error(
                            'Unable to create Kafka Producer, cannot send the results to the broker.'
                        )
                        self.logger.debug(f'Error: {e}')

                if self.report:
                    self.logger.info(f'Generating report for molecule {molecule_id + 1}...')
                    self.generate_report()
                    self.logger.info(f'Generated the report for molecule {molecule_id + 1}.')

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
