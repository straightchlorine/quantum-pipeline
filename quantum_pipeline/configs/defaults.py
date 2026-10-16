from typing import Any

DEFAULTS: dict[str, Any] = {
    'basis_set': 'sto3g',
    'ansatz_reps': 2,
    'local': True,
    'ibm_quantum': False,
    'max_iterations': 100,
    # Hard ceiling on objective (energy) evaluations.
    # --
    # Makes the run terminate and bounds the result size (one recorded row per evaluation).
    # It lets the broker limit in compose/docker-compose.ml.yaml be a fixed number.
    # None disables it, at the cost of both guarantees.
    #
    # When max_iterations is also set the smaller of the two applies.
    # --
    # Introduced in order to (just in case) cound the --convergence runs. scipy's
    # maxiter (15000) counts optimiser *steps* and the Aear lanes spend n_params + 1 evals
    # per step on finite differences.
    #
    # For example NH3 has 96 parameters, so unbounded --convergence can reach ~1.5M
    # of evaluations. That introduces it's own problems (besides the long time it takes)
    # but also exceeds Kafka message limit. It would be around 3GB at that point.
    'max_evaluations': 30000,
    'convergence_threshold_enable': False,
    'convergence_threshold': 1e-6,
    'optimizer': 'L-BFGS-B',
    'ansatz_type': 'EfficientSU2',
    'seed': None,
    'init_strategy': 'random',
    # None means exact expectation values (analytic gradients for gradient optimizers).
    # --shots N switches to sampling.
    'shots': None,
    # IBM hardware cannot be exact, so --ibm without --shots samples this many shots.
    'ibm_shots': 1024,
    'backend': {
        'local': True,
        'min_qubits': None,
        'optimization_level': 3,
        'filters': None,
        'method': 'statevector',  # tensor_network requires cuQuantum
        'gpu': False,
        'noise_backend': None,
        'gpu_opts': {
            'device': 'GPU',
            # set to true if you have Volta or Ampere architecture GPUs (and cuda >=11.2)
            # and your qiskit-aer is built with cuQuantum support (Dockerfile.gpu should
            # account for that)
            # https://github.com/Qiskit/qiskit-aer/blob/main/CONTRIBUTING.md
            'cuStateVec_enable': False,  # Correct for Pascal architecture (GTX 1060/1050 Ti)
            'blocking_enable': False,  # Reduce synchronization overhead
            'batched_shots_gpu': True,  # Enable shot parallelization for better GPU utilization
            'shot_branching_enable': True,  # Enable circuit branching optimization
            'max_memory_mb': 5500,  # GTX 1060: 6GB - 500MB buffer for optimal memory usage
        },
    },
    'kafka': {
        'servers': 'localhost:9092',
        'topic': 'experiment.vqe',
        'retries': 3,
        'internal_retries': 0,
        'acks': 'all',
        'timeout': 10,
        'retry_delay': 2,
        # kafka-python defaults this to 1 MiB, which is easily exceeded by larger runs
        # (for example, a single row is ~2 KB for NH3 and there is one row per evaluation).
        # NOTE: must stay below the broker's own socket.request.max.bytes.
        'max_request_size': 134217728,
        # kafka-python rejects a record larger than buffer_memory and one larger than max_request_size
        # (producer raises MessageSizeTooLargeError for both),
        # so raising only max_request_size leaves the send failing at 32 MiB.
        'buffer_memory': 134217728,
        'security': {
            'ssl': False,
            'sasl_ssl': False,
            'ssl_check_hostname': True,
            'certs': {
                'dir': './secrets/',
                'cafile': 'ca.crt',
                'certfile': 'client.crt',
                'keyfile': 'client.key',
                'pass': None,
            },
            'sasl_ssl_opts': {
                'sasl_mechanism': 'PLAIN',
                'sasl_plain_username': None,
                'sasl_plain_password': None,
            },
        },
    },
}
