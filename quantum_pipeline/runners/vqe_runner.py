import math
import os

import numpy as np
from qiskit_nature.second_q.drivers.pyscfd.pyscfdriver import PySCFDriver

from quantum_pipeline.circuits import HFData
from quantum_pipeline.configs.module.backend import BackendConfig
from quantum_pipeline.configs.module.producer import ProducerConfig
from quantum_pipeline.configs.module.security import SecurityConfig
from quantum_pipeline.drivers.basis_sets import validate_basis_set
from quantum_pipeline.drivers.molecule_loader import load_molecule, load_molecule_names
from quantum_pipeline.mappers import JordanWignerMapper
from quantum_pipeline.monitoring import get_performance_monitor
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
        ansatz_type='EfficientSU2',
        default_shots=1024,
        seed=None,
        init_strategy='random',
        report=False,
        kafka=False,
        kafka_bootstrap_servers='localhost:9092',
        kafka_topic='experiment.vqe',
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
        molecule_index=None,
    ):
        super().__init__()
        self.filepath = filepath
        self.molecule_index = molecule_index
        self.basis_set = basis_set
        self.max_iterations = max_iterations
        self.ansatz_reps = ansatz_reps
        self.ansatz_type = ansatz_type
        self.optimizer = optimizer
        self.default_shots = default_shots
        self.convergence_threshold = convergence_threshold
        self.seed = seed
        self.init_strategy = init_strategy
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
                raise

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
                raise
        else:
            try:
                self.backend_config = BackendConfig.default_backend_config()
            except Exception as e:
                self.logger.error(
                    'Unable to create default backend_config. '
                    f'ensure required parameters are passed to the VQERunner instance: {e}'
                )
                raise

        self.run_results = []

        # Initialize performance monitoring
        self.performance_monitor = get_performance_monitor()

    @staticmethod
    def default_backend():
        return BackendConfig.default_backend_config()

    def load_molecules(self):
        """Load molecule data and validate the basis set."""
        self.logger.info(f'Loading molecule data from {self.filepath}')
        molecules = load_molecule(self.filepath)
        self.molecule_names = load_molecule_names(self.filepath)
        validate_basis_set(self.basis_set)
        return molecules

    def provide_hamiltonian(self, molecule):
        """Generate the second quantized operator and extract HF data."""
        driver = PySCFDriver.from_molecule(molecule, basis=self.basis_set)
        problem = driver.run()
        second_q_op = problem.second_q_ops()[0]

        hf_data = HFData(
            num_particles=problem.num_particles,
            num_spatial_orbitals=problem.num_spatial_orbitals,
            reference_energy=problem.reference_energy,
            nuclear_repulsion_energy=problem.nuclear_repulsion_energy,
        )

        # Log nuclear repulsion (stored in hamiltonian.constants, NOT in the FermionicOp)
        try:
            self.logger.debug(
                f'Nuclear repulsion energy: {problem.nuclear_repulsion_energy:.8f} Ha'
            )
            self.logger.debug(f'ElectronicEnergy constants: {problem.hamiltonian.constants}')
        except Exception as e:
            self.logger.debug(f'Could not log hamiltonian details: {e}')

        return second_q_op, hf_data

    def run_vqe(self, molecule, backend_config: BackendConfig):
        """Prepare and run the VQE algorithm."""

        self.logger.info('Generating hamiltonian based on the molecule...')
        with Timer() as t:
            second_q_op, hf_data = self.provide_hamiltonian(molecule)

        self.hamiltonian_time = t.elapsed
        self.hf_reference_energy = hf_data.reference_energy
        self.logger.info(f'Hamiltonian generated in {t.elapsed:.6f} seconds.')
        if self.hf_reference_energy is not None:
            self.logger.info(f'HF reference energy: {self.hf_reference_energy:.6f} Ha')

        mapper = JordanWignerMapper()

        self.logger.info('Mapping fermionic operator to qubits')
        with Timer() as t:
            qubit_op = mapper.map(second_q_op)

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
                ansatz_type=self.ansatz_type,
                default_shots=self.default_shots,
                convergence_threshold=self.convergence_threshold,
                seed=self.seed,
                init_strategy=self.init_strategy,
                hf_data=hf_data if self.init_strategy == 'hf' else None,
                mapper=mapper,
            )
            result = solver.solve()

        self.vqe_time = t.elapsed
        self.logger.info(f'VQE procedure completed in {t.elapsed:.6f} seconds')

        return result

    def _collect_accuracy_metrics(self, molecule_name, result) -> dict:
        """Calculate accuracy metrics for a VQE result against the HF baseline."""
        hf_energy = self.hf_reference_energy
        if hf_energy is None:
            return {
                'reference_available': False,
                'reference_energy_hartree': None,
                'energy_error_hartree': None,
                'energy_error_millihartree': None,
                'relative_error_percent': None,
                'accuracy_score': None,
            }

        vqe_energy = float(result.total_energy)
        energy_error = vqe_energy - hf_energy
        relative_error = abs(energy_error / hf_energy) * 100

        if abs(energy_error) < 1e-10:
            accuracy_score = 100.0
        else:
            log_error = math.log10(abs(energy_error) * 1000 + 1)
            accuracy_score = max(0.0, 100.0 * (1.0 - log_error / 5.0))

        self.logger.info(f'Accuracy assessment for {molecule_name}:')
        self.logger.info(f'  VQE Total Energy: {vqe_energy:.6f} Ha')
        self.logger.info(f'  HF Reference:     {hf_energy:.6f} Ha')
        self.logger.info(f'  Error:            {energy_error * 1000:.3f} mHa')
        self.logger.info(f'  Accuracy Score:   {accuracy_score:.1f}/100')

        return {
            'reference_available': True,
            'reference_energy_hartree': hf_energy,
            'energy_error_hartree': energy_error,
            'energy_error_millihartree': energy_error * 1000,
            'relative_error_percent': relative_error,
            'accuracy_score': min(100, accuracy_score),
        }

    def _build_metrics_data(self, molecule_id, molecule_name, result, total_time, accuracy_metrics) -> dict:
        """Build the metrics dict used for Prometheus export."""
        return {
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
            'minimum_energy': float(result.total_energy),
            'iterations_count': len(result.iteration_list),
            'optimal_parameters_count': len(result.optimal_parameters),
            # Accuracy metrics (use `or 0` to handle both missing keys and None values)
            'reference_energy': accuracy_metrics.get('reference_energy_hartree') or 0,
            'energy_error_hartree': accuracy_metrics.get('energy_error_hartree') or 0,
            'energy_error_millihartree': accuracy_metrics.get('energy_error_millihartree') or 0,
            'accuracy_score': accuracy_metrics.get('accuracy_score') or 0,
        }

    def _process_molecule(self, molecule_id, molecule) -> VQEDecoratedResult:
        """Run the full VQE pipeline for a single molecule and return a decorated result."""
        molecule_name = self.molecule_names[molecule_id]

        # Set experiment context for monitoring
        self.performance_monitor.set_experiment_context(
            molecule_id=molecule_id,
            molecule_symbols=molecule_name,
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
            minimum_energy=float(result.total_energy),
            hamiltonian_time=float(self.hamiltonian_time),
            mapping_time=float(self.mapping_time),
            vqe_time=float(self.vqe_time),
            iterations_count=len(result.iteration_list),
            optimal_parameters_count=len(result.optimal_parameters),
        )

        accuracy_metrics = self._collect_accuracy_metrics(molecule_name, result)

        # Export VQE metrics immediately to Prometheus with full context and accuracy
        try:
            vqe_metrics_data = self._build_metrics_data(
                molecule_id, molecule_name, result, total_time, accuracy_metrics
            )
            self.performance_monitor.export_vqe_metrics_immediate(vqe_metrics_data)
        except Exception as e:
            self.logger.warning(f'Metrics export failed (non-fatal): {e}')

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
        self.logger.debug('Appended run information to the result.')

        return decorated_result

    def _get_producer(self) -> VQEKafkaProducer:
        """Return the shared Kafka producer, creating it on first use."""
        if not hasattr(self, '_producer') or self._producer is None:
            self._producer = VQEKafkaProducer(self.kafka_config)
        return self._producer

    def _close_producer(self) -> None:
        """Close the Kafka producer if it was created."""
        if hasattr(self, '_producer') and self._producer is not None:
            try:
                self._producer.close()
            except Exception as e:
                self.logger.debug(f'Error closing Kafka producer: {e}')
            self._producer = None

    def _stream_result(self, decorated_result):
        """Send a decorated VQE result to the Kafka broker."""
        try:
            producer = self._get_producer()
            producer.send_result(decorated_result)
        except Exception as e:
            self.logger.error('Unable to send the result to the Kafka broker.')
            self.logger.debug(f'Error: {e}')

    def run(self):
        self.molecules = self.load_molecules()

        if self.molecule_index is not None:
            if self.molecule_index >= len(self.molecules):
                raise IndexError(
                    f'molecule-index {self.molecule_index} out of range '
                    f'(file has {len(self.molecules)} molecules)'
                )
            self.molecules = [self.molecules[self.molecule_index]]
            self.molecule_names = [self.molecule_names[self.molecule_index]]

        try:
            with self.performance_monitor:
                for molecule_id, molecule in enumerate(self.molecules):
                    self.logger.info(f'Processing molecule {molecule_id + 1}:\n\n{molecule}\n')
                    decorated_result = self._process_molecule(molecule_id, molecule)
                    self.run_results.append(decorated_result)

                    if self.kafka:
                        self._stream_result(decorated_result)

                    if self.report:
                        self.logger.info(f'Generating report for molecule {molecule_id + 1}...')
                        self.generate_report()
                        self.logger.info(f'Generated the report for molecule {molecule_id + 1}.')
        finally:
            if self.kafka:
                self._close_producer()

        self.logger.info('All molecules processed.')

        if self.report:
            self.report_gen.generate_report()

    def generate_report(self):
        for result in self.run_results:
            self.report_gen.add_header('Structure of the molecule in 3D')
            self.report_gen.add_molecule_plot(result.molecule)

            self.report_gen.add_insight('Energy analysis', 'Results of VQE algorithm:')
            self.report_gen.add_metrics(
                {
                    'Minimum Energy': str(round(result.vqe_result.minimum, 4)) + ' Hartree',
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
