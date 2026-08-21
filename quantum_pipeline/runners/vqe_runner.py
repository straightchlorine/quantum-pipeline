from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from qiskit_nature.second_q.drivers.pyscfd.pyscfdriver import PySCFDriver

from quantum_pipeline.circuits import HFData
from quantum_pipeline.configs.module.backend import BackendConfig
from quantum_pipeline.configs.module.producer import ProducerConfig
from quantum_pipeline.drivers.basis_sets import validate_basis_set
from quantum_pipeline.drivers.molecule_loader import load_molecule, load_molecule_names
from quantum_pipeline.mappers import JordanWignerMapper
from quantum_pipeline.monitoring import get_performance_monitor
from quantum_pipeline.report.report_generator import ReportGenerator
from quantum_pipeline.runners.runner import Runner
from quantum_pipeline.solvers.vqe_solver import VQESolver
from quantum_pipeline.stream.kafka_interface import KafkaProducerError, VQEKafkaProducer
from quantum_pipeline.structures.vqe_observation import VQEDecoratedResult
from quantum_pipeline.utils.timer import Timer
from quantum_pipeline.visual.ansatz import AnsatzViewer

if TYPE_CHECKING:
    from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
    from qiskit_nature.second_q.operators import FermionicOp

    from quantum_pipeline.structures.vqe_observation import VQEResult

# Local dead-letter directory for results that failed to reach Kafka.
UNDELIVERED_DIR = Path('gen/undelivered')

# hf_deviation_score scaling (see VQERunner._collect_hf_deviation_metrics).
_HA_TO_MILLIHARTREE = 1000.0
_LOG_DECADES_TO_ZERO = 5.0  # ~5 decades of mHa deviation spans the full 100..0 range


class VQERunOutput(NamedTuple):
    """Per-molecule outputs of run_vqe, returned instead of being stashed on self."""

    result: VQEResult
    hamiltonian_time: float
    mapping_time: float
    vqe_time: float
    hf_reference_energy: float | None


class VQERunner(Runner):
    """Class to handle the ground energy finding process."""

    def __init__(
        self,
        filepath: str,
        basis_set: str = 'sto3g',
        max_iterations: int | None = 100,
        convergence_threshold: float | None = None,
        optimizer: str = 'COBYLA',
        ansatz_reps: int = 3,
        ansatz_type: str = 'EfficientSU2',
        default_shots: int | None = 1024,
        seed: int | None = None,
        init_strategy: str = 'random',
        report: bool = False,
        kafka: bool = False,
        kafka_config: ProducerConfig | None = None,
        backend_config: BackendConfig | None = None,
        molecule_index: int | None = None,
    ) -> None:
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

        self.report = report
        if self.report:
            self.report_gen = ReportGenerator()

        self.kafka = kafka
        if self.kafka:
            # ProducerConfig.from_dict({}) fills every field from DEFAULTS.
            self.kafka_config = kafka_config or ProducerConfig.from_dict({})

        # backend_config is the single source of truth; fall back to the project
        # defaults (statevector method, gpu_opts, ...) when the caller passes none.
        self.backend_config = (
            backend_config
            if backend_config is not None
            else BackendConfig.default_backend_config()
        )

        # A config may leave optimization_level unset; fall back to VQESolver's default of 3.
        self.optimization_level = (
            self.backend_config.optimization_level
            if self.backend_config.optimization_level is not None
            else 3
        )

        self.run_results: list[VQEDecoratedResult] = []
        # Molecule indices whose results failed to reach Kafka (spooled to disk instead).
        self._undelivered: list[int] = []
        # Lazily-created shared Kafka producer (see _get_producer).
        self._producer: VQEKafkaProducer | None = None

        # Initialize performance monitoring
        self.performance_monitor = get_performance_monitor()

    @staticmethod
    def default_backend() -> BackendConfig:
        return BackendConfig.default_backend_config()

    def load_molecules(self) -> list[MoleculeInfo]:
        """Load molecule data and validate the basis set."""
        self.logger.info(f'Loading molecule data from {self.filepath}')
        molecules = load_molecule(self.filepath)
        self.molecule_names = load_molecule_names(self.filepath)
        validate_basis_set(self.basis_set)
        return molecules

    def provide_hamiltonian(self, molecule: MoleculeInfo) -> tuple[FermionicOp, HFData]:
        """Generate the second quantized operator and extract HF data."""
        driver = PySCFDriver.from_molecule(molecule, basis=self.basis_set)
        problem = driver.run()
        second_q_op = problem.second_q_ops()[0]

        num_particles = problem.num_particles
        num_spatial_orbitals = problem.num_spatial_orbitals
        if num_particles is None or num_spatial_orbitals is None:
            raise ValueError(
                f'PySCF driver returned an incomplete problem for {molecule.symbols}: '
                f'num_particles={num_particles}, num_spatial_orbitals={num_spatial_orbitals}'
            )

        hf_data = HFData(
            num_particles=num_particles,
            num_spatial_orbitals=num_spatial_orbitals,
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

    def run_vqe(self, molecule: MoleculeInfo, backend_config: BackendConfig) -> VQERunOutput:
        """Prepare and run the VQE algorithm."""

        self.logger.info('Generating hamiltonian based on the molecule...')
        with Timer() as t:
            second_q_op, hf_data = self.provide_hamiltonian(molecule)

        hamiltonian_time = t.elapsed
        hf_reference_energy = hf_data.reference_energy
        self.logger.info(f'Hamiltonian generated in {hamiltonian_time:.6f} seconds.')
        if hf_reference_energy is not None:
            self.logger.info(f'HF reference energy: {hf_reference_energy:.6f} Ha')

        mapper = JordanWignerMapper()

        self.logger.info('Mapping fermionic operator to qubits')
        with Timer() as t:
            qubit_op = mapper.map(second_q_op)

        mapping_time = t.elapsed
        self.logger.info(f'Problem mapped to qubits in {mapping_time:.6f} seconds.')

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
                # 'hf' init needs hf_data; ExcitationPreserving requires it structurally.
                # Other ansatze never read it.
                hf_data=hf_data,
                mapper=mapper,
            )
            result = solver.solve()

        vqe_time = t.elapsed
        self.logger.info(f'VQE procedure completed in {vqe_time:.6f} seconds')

        return VQERunOutput(
            result=result,
            hamiltonian_time=hamiltonian_time,
            mapping_time=mapping_time,
            vqe_time=vqe_time,
            hf_reference_energy=hf_reference_energy,
        )

    def _collect_hf_deviation_metrics(
        self, molecule_name: str, result: VQEResult, hf_energy: float | None
    ) -> dict:
        """Quantify how far the VQE energy sits from the Hartree-Fock reference.

        This measures deviation from HF, which is itself an approximation, not the
        exact ground state. It is a convergence/sanity indicator, NOT accuracy
        against the true ground-state energy. `hf_deviation_score` is a bounded
        [0, 100] heuristic: 100 at the HF energy, decaying with the log of the
        millihartree deviation.
        """
        if hf_energy is None:
            return {
                'reference_available': False,
                'reference_energy_hartree': None,
                'energy_error_hartree': None,
                'energy_error_millihartree': None,
                'relative_error_percent': None,
                'hf_deviation_score': None,
            }

        vqe_energy = float(result.total_energy)
        energy_error = vqe_energy - hf_energy
        relative_error = abs(energy_error / hf_energy) * 100

        if abs(energy_error) < 1e-10:
            hf_deviation_score = 100.0
        else:
            log_error = math.log10(abs(energy_error) * _HA_TO_MILLIHARTREE + 1)
            hf_deviation_score = 100.0 * (1.0 - log_error / _LOG_DECADES_TO_ZERO)
        hf_deviation_score = max(0.0, min(100.0, hf_deviation_score))

        self.logger.info(f'HF-deviation assessment for {molecule_name}:')
        self.logger.info(f'  VQE Total Energy: {vqe_energy:.6f} Ha')
        self.logger.info(f'  HF Reference:     {hf_energy:.6f} Ha')
        self.logger.info(f'  Deviation:        {energy_error * 1000:.3f} mHa')
        self.logger.info(f'  HF-deviation score: {hf_deviation_score:.1f}/100')

        return {
            'reference_available': True,
            'reference_energy_hartree': hf_energy,
            'energy_error_hartree': energy_error,
            'energy_error_millihartree': energy_error * 1000,
            'relative_error_percent': relative_error,
            'hf_deviation_score': hf_deviation_score,
        }

    def _build_metrics_data(
        self,
        molecule_id: int,
        molecule_name: str,
        run: VQERunOutput,
        total_time: float,
        deviation_metrics: dict,
    ) -> dict:
        """Build the metrics dict used for Prometheus export."""
        result = run.result
        return {
            'container_type': os.getenv('CONTAINER_TYPE', 'unknown'),
            'molecule_id': molecule_id,
            'molecule_symbols': molecule_name,
            'basis_set': self.basis_set,
            'optimizer': self.optimizer,
            'backend_type': 'GPU' if self.backend_config.gpu else 'CPU',
            'total_time': float(total_time),
            'hamiltonian_time': float(run.hamiltonian_time),
            'mapping_time': float(run.mapping_time),
            'vqe_time': float(run.vqe_time),
            'minimum_energy': float(result.total_energy),
            'iterations_count': len(result.iteration_list),
            'optimal_parameters_count': len(result.optimal_parameters),
            # HF-deviation metrics
            'reference_energy': deviation_metrics.get('reference_energy_hartree') or 0,
            'energy_error_hartree': deviation_metrics.get('energy_error_hartree') or 0,
            'energy_error_millihartree': deviation_metrics.get('energy_error_millihartree') or 0,
            'hf_deviation_score': deviation_metrics.get('hf_deviation_score') or 0,
        }

    def _process_molecule(self, molecule_id: int, molecule: MoleculeInfo) -> VQEDecoratedResult:
        """Run the full VQE pipeline for a single molecule and return a decorated result."""
        molecule_name = self.molecule_names[molecule_id]
        backend_type = 'GPU' if self.backend_config.gpu else 'CPU'

        # Set experiment context for monitoring
        self.performance_monitor.set_experiment_context(
            molecule_id=molecule_id,
            molecule_symbols=molecule_name,
            basis_set=self.basis_set,
            optimizer=self.optimizer,
            max_iterations=self.max_iterations,
            backend_type=backend_type,
        )

        # Collect performance snapshot before VQE
        performance_start = self.performance_monitor.collect_metrics_snapshot()

        run = self.run_vqe(molecule, self.backend_config)
        result = run.result

        # Collect performance snapshot after VQE
        performance_end = self.performance_monitor.collect_metrics_snapshot()

        total_time = run.hamiltonian_time + run.mapping_time + run.vqe_time
        self.logger.info(f'Result provided in {total_time:.6f} seconds.')

        # Update experiment context with VQE results for Prometheus export
        self.performance_monitor.set_experiment_context(
            total_time=total_time,
            minimum_energy=float(result.total_energy),
            hamiltonian_time=run.hamiltonian_time,
            mapping_time=run.mapping_time,
            vqe_time=run.vqe_time,
            iterations_count=len(result.iteration_list),
            optimal_parameters_count=len(result.optimal_parameters),
        )

        deviation_metrics = self._collect_hf_deviation_metrics(
            molecule_name, result, run.hf_reference_energy
        )

        # Export VQE metrics immediately to Prometheus with full context
        try:
            vqe_metrics_data = self._build_metrics_data(
                molecule_id, molecule_name, run, total_time, deviation_metrics
            )
            self.performance_monitor.export_vqe_metrics_immediate(vqe_metrics_data)
        except Exception as e:
            self.logger.warning(f'Metrics export failed (non-fatal): {e}')

        decorated_result = VQEDecoratedResult(
            vqe_result=result,
            molecule=molecule,
            basis_set=self.basis_set,
            molecule_id=molecule_id,
            hamiltonian_time=np.float64(run.hamiltonian_time),
            mapping_time=np.float64(run.mapping_time),
            vqe_time=np.float64(run.vqe_time),
            total_time=np.float64(total_time),
            performance_start=performance_start if self.performance_monitor.is_enabled() else None,
            performance_end=performance_end if self.performance_monitor.is_enabled() else None,
        )
        self.logger.debug('Appended run information to the result.')

        return decorated_result

    def _get_producer(self) -> VQEKafkaProducer:
        """Return the shared Kafka producer, creating it on first use."""
        if self._producer is None:
            self._producer = VQEKafkaProducer(self.kafka_config)
        return self._producer

    def _close_producer(self) -> None:
        """Close the Kafka producer if it was created."""
        if self._producer is not None:
            try:
                self._producer.close()
            except Exception as e:
                self.logger.debug(f'Error closing Kafka producer: {e}')
            self._producer = None

    def _stream_result(self, decorated_result: VQEDecoratedResult, molecule_id: int) -> bool:
        """Send a decorated VQE result to the Kafka broker.

        A failed send must never be silent: the computed result is spooled to disk
        for replay and the molecule is recorded so run() can exit non-zero. Otherwise
        a downed broker would drop paid GPU output while the job still reported success.

        Returns True on success, False on failure.
        """
        try:
            producer = self._get_producer()
            producer.send_result(decorated_result)
            return True
        except Exception as e:
            self.logger.error(
                f'Failed to stream result for molecule index {molecule_id} to Kafka: {e}'
            )
            self._undelivered.append(molecule_id)
            self._spool_undelivered(decorated_result, molecule_id)
            return False

    def _spool_undelivered(self, decorated_result: VQEDecoratedResult, molecule_id: int) -> None:
        """Persist an undelivered result to disk so it can be replayed (no recompute).

        Best-effort: uses the Avro interface's registry-free serialize() to write a
        JSON record.
        """
        producer = self._producer
        if producer is None:
            self.logger.error(
                f'Cannot spool molecule {molecule_id}: producer never initialized. '
                'Re-run once the broker is reachable (run will exit non-zero).'
            )
            return
        try:
            UNDELIVERED_DIR.mkdir(parents=True, exist_ok=True)
            payload = producer.serializer.serialize(decorated_result)
            path = UNDELIVERED_DIR / f'molecule_{molecule_id}.json'
            with open(path, 'w') as f:
                json.dump(payload, f, default=str)
            self.logger.warning(f'Spooled undelivered result to {path} for later replay.')
        except Exception as e:
            self.logger.error(
                f'Failed to spool undelivered result for molecule {molecule_id}: {e}'
            )

    def run(self) -> None:
        self.molecules = self.load_molecules()

        if self.molecule_index is not None:
            # Negative indices must be rejected too: Python would silently wrap them
            # and run a different molecule than the caller asked for.
            if not 0 <= self.molecule_index < len(self.molecules):
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
                        self._stream_result(decorated_result, molecule_id)

        finally:
            if self.kafka:
                self._close_producer()

        self.logger.info('All molecules processed.')

        if self.report:
            # Build content once over all results. Calling this per-molecule would
            # re-append every earlier molecule's content (O(N^2) duplicated pages).
            self.logger.info('Generating report...')
            self.generate_report()
            self.report_gen.generate_report()

        # Surface streaming failures loudly.
        # Results spooled (see _spool_undelivered) for replay.
        if self._undelivered:
            raise KafkaProducerError(
                f'{len(self._undelivered)} of {len(self.run_results)} results failed to '
                f'stream to Kafka (molecule indices {self._undelivered}). They were spooled '
                f'to {UNDELIVERED_DIR}/ for replay. Failing non-zero so the job is retried.'
            )

    def generate_report(self) -> None:
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
