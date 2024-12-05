from qiskit_nature.second_q.drivers.pyscfd.pyscfdriver import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper

from quantum_simulation.drivers.basis_sets import validate_basis_set
from quantum_simulation.drivers.molecule_loader import load_molecule
from quantum_simulation.runners.runner import Runner
from quantum_simulation.solvers.vqe_solver import VQESolver
from quantum_simulation.utils.logger import get_logger

logger = get_logger('VQERunner')


class VQERunner(Runner):
    """Class to handle the ground energy finding process."""

    def load_and_validate(self, file_path: str, basis_set: str):
        """Load molecule data and validate the basis set."""
        logger.info(f'Loading molecule data from {file_path}')
        molecules = load_molecule(file_path)
        validate_basis_set(basis_set)
        return molecules

    def prepare_simulation(self, molecule, basis_set: str):
        """Generate the second quantized operator."""
        logger.info(f'Preparing simulation for molecule:\n {molecule}')
        driver = PySCFDriver.from_molecule(molecule, basis=basis_set)
        problem = driver.run()
        return problem.second_q_ops()[0]

    def run(self, qubit_op, symbols):
        """Run the VQE simulation and generate insights."""
        logger.info('Solving using VQE')
        return VQESolver(
            qubit_op, report_generator=self.report, symbols=symbols
        ).solve()

    def process_molecule(self, molecule, basis_set: str):
        """Process a single molecule:
        - prepare simulation,
        - run VQE,
        - generate reports.
        """
        second_q_op = self.prepare_simulation(molecule, basis_set)

        self.report.add_header('Structure of the molecule in 3D')
        self.report.add_molecule_plot(molecule)

        logger.info('Mapping fermionic operator to qubits')
        qubit_op = JordanWignerMapper().map(second_q_op)

        energy = self.run(qubit_op, molecule.symbols)

        self.report.new_page()
        self.report.add_header('Real coefficients of the operators')
        self.report.add_operator_coefficients_plot(qubit_op, molecule.symbols)

        self.report.add_header('Complex coefficients of the operators')
        self.report.add_complex_operator_coefficients_plot(
            qubit_op, molecule.symbols
        )

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
