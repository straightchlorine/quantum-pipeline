import argparse

from qiskit_nature.second_q.drivers.pyscfd.pyscfdriver import PySCFDriver
from qiskit_nature.second_q.problems.electronic_structure_problem import (
    ElectronicStructureProblem,
)

from quantum_simulation.configs import settings
from quantum_simulation.drivers.basis_sets import validate_basis_set
from quantum_simulation.drivers.molecule_loader import load_molecule
from quantum_simulation.mappers.jordan_winger_mapper import JordanWignerMapper
from quantum_simulation.report.report_generator import ReportGenerator
from quantum_simulation.solvers.vqe_solver import VQESolver
from quantum_simulation.utils.dir import ensureDirExists
from quantum_simulation.utils.logger import get_logger

logger = get_logger('QuantumAtomicSim')


def load_and_validate(file_path: str, basis_set: str):
    """
    Load molecule data and validate the basis set.
    """
    logger.info(f'Loading molecule data from {file_path}')
    molecules = load_molecule(file_path)
    validate_basis_set(basis_set)
    return molecules


def prepare_simulation(molecule, basis_set: str):
    """
    Prepare the simulation by generating the second quantized operator.
    """
    logger.info(f'Preparing simulation for molecule:\n {molecule}')
    driver = PySCFDriver.from_molecule(molecule, basis=basis_set)
    problem = driver.run()
    return problem.second_q_ops()[0]


def run_vqe_simulation(qubit_op, report_gen: ReportGenerator, symbols):
    """
    Run the VQE simulation and generate insights.
    """
    logger.info('Solving using VQE')
    return VQESolver(
        qubit_op, report_generator=report_gen, symbols=symbols
    ).solve()


def process_molecule(molecule, basis_set: str, report: ReportGenerator):
    """
    Process a single molecule:
        - prepare simulation,
        - run VQE
        - generate reports.
    """
    second_q_op = prepare_simulation(molecule, basis_set)

    report.add_header('Structure of the molecule in 3D')
    report.add_molecule_plot(molecule)

    logger.info('Mapping fermionic operator to qubits')
    qubit_op = JordanWignerMapper().map(second_q_op)

    energy = run_vqe_simulation(qubit_op, report, molecule.symbols)

    report.new_page()
    report.add_header('Real coefficients of the operators')
    report.add_operator_coefficients_plot(qubit_op, molecule.symbols)

    report.add_header('Complex coefficients of the operators')
    report.add_complex_operator_coefficients_plot(qubit_op, molecule.symbols)

    report.new_page()
    report.add_insight(
        'Energy convergence of the algorithm',
        'Energy levels for each iteration:',
    )
    report.add_convergence_plot(molecule)

    report.add_insight(
        'Ansatz circuit for the molecule was generated in the project directory.',
        'They often take too much space to display. See graph/ansatz and graph/ansatz_decomposed.',
    )

    return energy


def main(molecule_file: str, basis_set: str):
    """
    Main function to manage the simulation.
    """
    ensureDirExists(settings.GRAPH_DIR)
    report = ReportGenerator()

    try:
        molecules = load_and_validate(molecule_file, basis_set)

        for idx, molecule in enumerate(molecules):
            process_molecule(molecule, basis_set, report)

            if idx < len(molecules) - 1:
                report.new_page()

        report.generate_report()
    except Exception as e:
        logger.error(f'Error encountered: {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quantum Atomic Simulation')
    parser.add_argument(
        '-f', '--file', required=True, help='Path to molecule data file (JSON)'
    )
    parser.add_argument(
        '-b',
        '--basis',
        default='sto3g',
        help='Basis set for the simulation (default: sto3g)',
    )
    args = parser.parse_args()

    main(args.file, args.basis)
