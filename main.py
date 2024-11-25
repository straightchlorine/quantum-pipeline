import argparse
from quantum_simulation.drivers.molecule_loader import load_molecule
from quantum_simulation.drivers.basis_sets import validate_basis_set
from quantum_simulation.solvers.vqe_solver import solve_vqe
from quantum_simulation.mappers.jordan_winger_mapper import JordanWignerMapper
from quantum_simulation.utils.logger import get_logger
from qiskit_nature.second_q.drivers.pyscfd.pyscfdriver import PySCFDriver
from qiskit_nature.second_q.problems.electronic_structure_problem import (
    ElectronicStructureProblem,
)

logger = get_logger('QuantumAtomicSim')


def load_and_validate(file_path: str, basis_set: str):
    logger.info(f'Loading molecule data from {file_path}')
    molecules = load_molecule(file_path)
    validate_basis_set(basis_set)
    return molecules


def prepare_simulation(molecule, basis_set: str):
    logger.info(f'Preparing simulation for molecule:\n {molecule}')
    driver = PySCFDriver.from_molecule(molecule, basis=basis_set)
    problem = driver.run()
    second_q_op = problem.second_q_ops()[0]
    return second_q_op


def run_simulation(second_q_op):
    logger.info('Mapping fermionic operator to qubits')
    qubit_op = JordanWignerMapper().map(second_q_op)

    logger.info('Solving using VQE')
    energy = solve_vqe(qubit_op)
    return energy


def main(molecule_file, basis_set):
    try:
        molecules = load_and_validate(molecule_file, basis_set)
        for molecule in molecules:
            second_q_op = prepare_simulation(molecule, basis_set)
            energy = run_simulation(second_q_op)
            logger.info(
                f'Ground state energy for:\n {molecule}\n Energy: {energy:.6f} Hartree'
            )
    except Exception as e:
        logger.error(f'Error encountered: {str(e)}')


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
