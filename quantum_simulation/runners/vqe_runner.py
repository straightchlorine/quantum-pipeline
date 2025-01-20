import time
import numpy as np
from qiskit_nature.second_q.drivers.pyscfd.pyscfdriver import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper

from quantum_simulation.configs import settings
from quantum_simulation.drivers.basis_sets import validate_basis_set
from quantum_simulation.drivers.molecule_loader import load_molecule
from quantum_simulation.runners.runner import Runner
from quantum_simulation.solvers.vqe_solver import VQESolver
from quantum_simulation.utils.observation import BackendConfig, VQEDecoratedResult


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

    def __init__(self, filepath, basis_set=None):
        super().__init__()
        if basis_set:
            self.basis_set = basis_set
        elif settings.BASIS_SET:
            basis_set = settings.BASIS_SET
        else:
            self.basis_set = 'sto3g'

        if filepath:
            self.filepath = filepath

        self.run_results = []

    def load_molecules(self):
        """Load molecule data and validate the basis set."""
        self.logger.info(f'Loading molecule data from {self.filepath}')
        molecules = load_molecule(self.filepath)
        validate_basis_set(self.basis_set)
        return molecules

    def provide_hamiltonian(self, molecule, basis_set: str):
        """Generate the second quantized operator."""
        self.logger.info(f'Preparing simulation for molecule:\n {molecule}')
        driver = PySCFDriver.from_molecule(molecule, basis=basis_set)
        problem = driver.run()
        return problem.second_q_ops()[0]

    def runVQE(self, molecule, backend_config: BackendConfig):
        """Prepare and run the VQE algorithm."""

        self.logger.info('Generating hamiltonian based on the molecule...')
        with Timer() as t:
            second_q_op = self.provide_hamiltonian(molecule, self.basis_set)

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
                qubit_op,
                backend_config,
            ).solve()

        self.vqe_time = t.elapsed
        self.logger.info(f'VQE procedure completed in {t.elapsed:.6f} seconds')

        return result

    def run(self, backend_config: BackendConfig, process=True):
        self.molecules = self.load_molecules()

        for id, molecule in enumerate(self.molecules):
            self.logger.info(f'Beginning to process {id} molecule:\n\n{molecule}\n')
            result = self.runVQE(molecule, backend_config)

            total_time = np.float64(self.hamiltonian_time + self.mapping_time + self.vqe_time)
            self.logger.info(f'Result provided in {total_time:.6f} seconds.')

            decorated_result = VQEDecoratedResult(
                vqe_result=result,
                molecule=molecule,
                id=id,
                hamiltonian_time=np.float64(self.hamiltonian_time),
                mapping_time=np.float64(self.mapping_time),
                vqe_time=np.float64(self.vqe_time),
                total_time=total_time,
            )
            self.run_results.append(decorated_result)
            self.logger.debug('Appended run information to the result.')

        self.logger.info('All molecules processed.')

    def process_molecule(self, molecule, basis_set: str, backend_config):
        """Process a single molecule:
        - prepare simulation,
        - run VQE,
        - generate reports.
        """
        second_q_op = self.provide_hamiltonian(molecule, basis_set)

        self.report.add_header('Structure of the molecule in 3D')
        self.report.add_molecule_plot(molecule)

        energy = self.runVQE(qubit_op, backend_config)

        self.report.new_page()
        self.report.add_header('Real coefficients of the operators')
        self.report.add_operator_coefficients_plot(qubit_op, molecule.symbols)

        self.report.add_header('Complex coefficients of the operators')
        self.report.add_complex_operator_coefficients_plot(qubit_op, molecule.symbols)

        self.report.new_page()
        self.report.add_insight(
            'Energy convergence of the algorithm',
            'Energy levels for each iteration:',
        )
        self.report.add_convergence_plot(molecule)

        self.report.add_insight(
            'Ansatz circuit for the molecule was generated in the project directory.',
            'They often take too much space to display. See graph/ansatz and graph/ansatz_decomposed.',
        )

        return energy
