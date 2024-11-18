import argparse
from quantum_simulation.drivers.molecule_loader import load_molecule
from quantum_simulation.drivers.basis_sets import validate_basis_set
from quantum_simulation.solvers.vqe_solver import solve_vqe
from quantum_simulation.mappers.jordan_winger_mapper import map_to_qubits
from quantum_simulation.utils.logger import get_logger
from qiskit_nature.second_q.drivers.pyscfd.pyscfdriver import PySCFDriver
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.problems.electronic_structure_problem import (
    ElectronicStructureProblem,
)

logger = get_logger("QuantumAtomicSim")


def main(molecule_file, basis_set):
    try:
        # Load molecules
        logger.info(f"Loading molecule data from {molecule_file}")
        molecules = load_molecule(molecule_file)

        for molecule in molecules:
            logger.info(f"Processing molecule: {molecule['name']}")
            validate_basis_set(basis_set)

            driver = PySCFDriver.from_molecule(molecule, basis=basis_set)

            # Build problem and fermionic operator
            problem = ElectronicStructureProblem(driver)
            second_q_op = problem.second_q_ops()[0]

            # Map fermionic operator to qubits
            logger.info("Mapping fermionic operator to qubits")
            qubit_op = map_to_qubits(second_q_op)

            # Solve with VQE
            logger.info("Solving using VQE")
            energy = solve_vqe(qubit_op)

            logger.info(
                f"Ground state energy for {molecule['name']}: {energy:.6f} Hartree"
            )
    except Exception as e:
        logger.error(f"Error encountered: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum Atomic Simulation")
    parser.add_argument(
        "-f", "--file", required=True, help="Path to molecule data file (JSON)"
    )
    parser.add_argument(
        "-b",
        "--basis",
        default="sto3g",
        help="Basis set for the simulation (default: sto3g)",
    )
    args = parser.parse_args()

    main(args.file, args.basis)
