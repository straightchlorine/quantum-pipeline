DEFAULTS = {
    'basis_set': 'sto3g',
    'ansatz_reps': 2,
    'local': True,
    'ibm_quantum': False,
    'max_iterations': 100,
    'convergence_threshold_enable': False,
    'convergence_threshold': 1e-6,
    'optimizer': 'COBYLA',
    'shots': 1024,
    'backend': {
        'local': True,
        'min_qubits': None,
        'optimization_level': 3,
        'filters': None,
        'method': 'automatic',
        'gpu': False,
        'noise_backend': None,
        'gpu_opts': {
            'device': 'GPU',
            'cuStateVec_enable': True,
            'blocking_enable': True,
            'batched_shots_gpu': True,
            'shot_branching_enable': True,
        },
    },
    'kafka': {
        'servers': 'localhost:9092',
        'topic': 'vqe_results',
        'retries': 3,
        'internal_retries': 0,
        'acks': 'all',
        'timeout': 10,
        'retry_delay': 2,
    },
}
