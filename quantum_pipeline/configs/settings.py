import logging
from pathlib import Path

SUPPORTED_OPTIMIZERS = {
    'Nelder-Mead': 'A simplex algorithm for unconstrained optimization.',
    'Powell': 'A directional set method for unconstrained optimization.',
    'CG': 'Non-linear conjugate gradient method for unconstrained optimization.',
    'BFGS': 'Quasi-Newton method using the Broyden–Fletcher–Goldfarb–Shanno algorithm.',
    'Newton-CG': "Newton's method with conjugate gradient for unconstrained optimization.",
    'L-BFGS-B': 'Limited-memory BFGS with box constraints.',
    'TNC': 'Truncated Newton method for bound-constrained optimization.',
    'COBYLA': 'Constrained optimization by linear approximations.',
    'COBYQA': 'Constrained optimization by quadratic approximations.',
    'SLSQP': 'Sequential Least Squares Programming for constrained optimization.',
    'trust-constr': 'Trust-region method for constrained optimization.',
    'dogleg': 'Dog-leg trust-region algorithm for unconstrained optimization.',
    'trust-ncg': 'Trust-region Newton conjugate gradient method.',
    'trust-exact': 'Exact trust-region optimization.',
    'trust-krylov': 'Trust-region method with Krylov subspace solver.',
    'custom': 'A user-provided callable object implementing the optimization method.',
}

SIMULATION_METHODS = {
    'automatic': 'Default simulation method. Selects the best method based on the circuit and noise model.',
    'statevector': 'Dense statevector simulation. Supports ideal circuits and noisy circuit sampling.',
    'density_matrix': 'Dense density matrix simulation for noisy circuits with end-of-circuit measurements.',
    'stabilizer': 'Efficient Clifford stabilizer state simulator for Clifford circuits and errors.',
    'extended_stabilizer': 'Approximate simulator for Clifford + T circuits using ranked-stabilizer decomposition.',
    'matrix_product_state': 'Tensor-network statevector simulator using MPS representation with optional truncation.',
    'unitary': 'Dense unitary matrix simulation. Computes the unitary matrix of the circuit (no measurement/noise).',
    'superop': 'Dense superoperator matrix simulation for ideal/noisy circuits (supports reset but no measurement).',
    'tensor_network': 'GPU-accelerated tensor-network simulation using cuTensorNet APIs.',
}

SCHEMA_REGISTRY_URL = 'http://schema-registry:8081'

LOG_LEVEL = logging.DEBUG
GEN_DIR = 'gen'

GRAPH_DIR = Path(GEN_DIR, 'graphs')
REPORT_DIR = GEN_DIR

SCHEMA_DIR = Path('quantum_pipeline', 'stream', 'serialization', 'schemas')
RUN_CONFIGS = Path('run_configs')

# molecule graph settings
MOLECULE_PLOT_DIR = Path(GRAPH_DIR, 'molecule_plots')
MOLECULE_PLOT = 'molecule'

OPERATOR_COEFFS_PLOT_DIR = Path(GRAPH_DIR, 'operator_plots')
OPERATOR_COEFFS_PLOT = 'operator'

OPERATOR_CPLX_COEFFS_PLOT_DIR = Path(GRAPH_DIR, 'complex_operator_plots')
OPERATOR_CPLX_COEFFS_PLOT = 'operator'

ENERGY_CONVERGENCE_PLOT_DIR = Path(GRAPH_DIR, 'energy_plots')
ENERGY_CONVERGENCE_PLOT = 'convergence'

ANSATZ_PLOT_DIR = Path(GRAPH_DIR, 'ansatz')
ANSATZ = 'ansatz'

ANSATZ_DECOMPOSED_PLOT_DIR = Path(GRAPH_DIR, 'ansatz_decomposed')
ANSATZ_DECOMPOSED = 'ansatz_decomposed'
