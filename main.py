import argparse

from quantum_simulation.configs import settings
from quantum_simulation.report.report_generator import ReportGenerator
from quantum_simulation.runners.vqe_runner import VQERunner
from quantum_simulation.utils.dir import ensureDirExists
from quantum_simulation.utils.logger import get_logger

logger = get_logger('MainRunner')


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Quantum Circuit data generation')
    parser.add_argument('-f', '--file', required=True, help='Path to molecule data file (JSON)')
    parser.add_argument(
        '-b',
        '--basis',
        default='sto3g',
        help='Basis set for the simulation (default: sto3g)',
    )
    return parser.parse_args()


def initialize_simulation_environment():
    """Ensure required directories and configurations are in place."""
    ensureDirExists(settings.GRAPH_DIR)
    logger.info('Simulation environment initialized')


def execute_simulation(molecule_file: str, basis_set: str):
    """Coordinate the process, including raporting and running the VQE."""
    initialize_simulation_environment()
    report = ReportGenerator()

    try:
        runner = VQERunner(report, ibm=True)
        molecules = runner.load_and_validate(molecule_file, basis_set)

        for idx, molecule in enumerate(molecules):
            runner.process_molecule(molecule, basis_set)

            if idx < len(molecules) - 1:
                report.new_page()

        report.generate_report()
    except Exception as e:
        logger.error(f'Error encountered: {e}')


if __name__ == '__main__':
    # args = parse_arguments()
    # execute_simulation(args.file, args.basis)
    execute_simulation('data/molecules.json', 'sto3g')
