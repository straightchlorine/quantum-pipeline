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
        'method': 'tensor_network',  # Optimized for GPU acceleration and larger circuits
        'gpu': False,
        'noise_backend': None,
        'gpu_opts': {
            'device': 'GPU',
            # set to true if you have Volta or Ampere architecture GPUs (and cuda >=11.2)
            # and your qiskit-aer is built with cuQuantum support (Dockerfile.gpu should
            # account for that)
            # https://github.com/Qiskit/qiskit-aer/blob/main/CONTRIBUTING.md
            'cuStateVec_enable': False,  # Correct for Pascal architecture (GTX 1060/1050 Ti)
            'blocking_enable': False,    # Reduce synchronization overhead
            'batched_shots_gpu': True,   # Enable shot parallelization for better GPU utilization
            'shot_branching_enable': True,  # Enable circuit branching optimization
            'max_memory_mb': 5500,       # GTX 1060: 6GB - 500MB buffer for optimal memory usage
        },
    },
    'kafka': {
        'servers': 'localhost:9092',
        'topic': 'vqe_decorated_result',
        'retries': 3,
        'internal_retries': 0,
        'acks': 'all',
        'timeout': 10,
        'retry_delay': 2,
        'security': {
            'ssl': False,
            'sasl_ssl': False,
            'ssl_check_hostname': True,
            'certs': {
                'dir': './secrets/',
                'cafile': 'ca.crt',
                'certfile': 'client.crt',
                'keyfile': 'client.key',
                'pass': '1234',
            },
            'sasl_ssl_opts': {
                'sasl_mechanism': 'PLAIN',
                'sasl_plain_username': 'user',
                'sasl_plain_password': 'password',
            },
        },
    },
}
