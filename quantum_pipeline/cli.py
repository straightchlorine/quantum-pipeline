"""CLI entry point for the quantum pipeline."""

import os

from quantum_pipeline.configs.defaults import DEFAULTS
from quantum_pipeline.configs.parsing.argparser import QuantumPipelineArgParser
from quantum_pipeline.runners.vqe_runner import VQERunner
from quantum_pipeline.utils.logger import get_logger

logger = get_logger('QuantumPipeline')


def execute_simulation(**kwargs):
    threshold = None
    max_iterations = kwargs['max_iterations']
    apply_threshold = kwargs.get('convergence')
    if apply_threshold:
        threshold = kwargs['threshold']
        logger.info(f'Applying convergence threshold {threshold} during minimization')
        if max_iterations != DEFAULTS['max_iterations']:
            logger.warning(
                f'--max-iterations {max_iterations} ignored: --convergence takes precedence.'
            )
        # convergence is prioritized over the max iter
        # raising mid run removed
        max_iterations = None

    # kafka_config is None unless --kafka is set, so the env override needs the guard.
    if kwargs['kafka'] and kwargs['kafka_config'] is not None and os.getenv('KAFKA_SERVERS'):
        kwargs['kafka_config'].servers = os.getenv('KAFKA_SERVERS')

    runner = VQERunner(
        filepath=kwargs['file'],
        basis_set=kwargs['basis'],
        max_iterations=max_iterations,
        convergence_threshold=threshold,
        optimizer=kwargs['optimizer'],
        ansatz_reps=kwargs['ansatz_reps'],
        ansatz_type=kwargs.get('ansatz_type', 'EfficientSU2'),
        default_shots=None if kwargs.get('exact') else kwargs['shots'],
        seed=kwargs.get('seed'),
        init_strategy=kwargs.get('init_strategy', 'random'),
        report=kwargs['report'],
        kafka=kwargs['kafka'],
        kafka_config=kwargs['kafka_config'] if kwargs['kafka'] else None,
        backend_config=kwargs['backend_config'],
        molecule_index=kwargs.get('molecule_index'),
    )
    runner.run()


def main():
    parser = QuantumPipelineArgParser()
    kwargs = parser.get_config()
    execute_simulation(**kwargs)
