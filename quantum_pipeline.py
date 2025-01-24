#!/usr/bin/env python

import os

from quantum_pipeline.configs.parsing.argparser import QuantumPipelineArgParser
from quantum_pipeline.runners.vqe_runner import VQERunner
from quantum_pipeline.utils.logger import get_logger

logger = get_logger('QuantumPipeline')


def execute_simulation(**kwargs):
    threshold = None
    apply_threshold = kwargs.get('convergence_threshold')
    if apply_threshold:
        threshold = kwargs['threshold']
        logger.info(f'Applying convergence threshold {threshold} during minimization')

    if os.getenv('KAFKA_SERVERS', None):
        kwargs['kafka_config'].servers = os.getenv('KAFKA_SERVERS')

    runner = VQERunner(
        filepath=kwargs['file'],
        basis_set=kwargs['basis'],
        max_iterations=kwargs['max_iterations'],
        convergence_threshold=threshold,
        optimizer=kwargs['optimizer'],
        ansatz_reps=kwargs['ansatz_reps'],
        default_shots=kwargs['shots'],
        report=kwargs['report'],
        kafka=kwargs['kafka'],
        kafka_config=kwargs['kafka_config'] if kwargs['kafka'] else None,
        backend_config=kwargs['backend_config'],
    )
    runner.run()


if __name__ == '__main__':
    parser = QuantumPipelineArgParser()
    kwargs = parser.get_config()
    execute_simulation(**kwargs)
