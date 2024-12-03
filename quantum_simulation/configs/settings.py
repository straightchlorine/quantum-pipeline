from pathlib import Path

BASIS_SET = 'sto3g'
OPTIMIZER = 'COBYLA'
ANSATZ_REPS = 2

GRAPH_DIR = 'graphs'

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
