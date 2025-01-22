#!/usr/bin/env python

from quantum_pipeline.configs.argparser import BackendConfig, QuantumPipelineArgParser
from quantum_pipeline.runners.vqe_runner import VQERunner
from quantum_pipeline.utils.logger import get_logger

logger = get_logger('QuantumPipeline')


def execute_simulation(
    molecule_file: str,
    basis_set: str,
    **kwargs,
):
    threshold = None
    apply_threshold = kwargs.get('convergence_threshold')
    if apply_threshold:
        threshold = kwargs['threshold']
        logger.info(f'Applying convergence threshold {threshold} during minimization')

    runner = VQERunner(
        filepath=molecule_file,
        basis_set=basis_set,
        max_iterations=kwargs['max_iterations'],
        convergence_threshold=threshold,
        optimizer=kwargs['optimizer'],
        ansatz_reps=kwargs['ansatz_reps'],
        default_shots=kwargs['shots'],
        report=kwargs['report'],
        kafka=kwargs['kafka'],
        kafka_config=kwargs['kafka_config'] if kwargs['kafka'] else None,
    )
    runner.run(kwargs['backend_config'])


if __name__ == '__main__':
    parser = QuantumPipelineArgParser()
    args = parser.parse_args()
    kwargs = parser.get_simulation_kwargs(args)
    execute_simulation(args.file, args.basis, **kwargs)
