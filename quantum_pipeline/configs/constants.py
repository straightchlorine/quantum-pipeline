"""
Named constants for quantum simulation algorithm parameters.

These are physics/algorithm constants used internally by the simulation engine.
They are separate from DEFAULTS (in defaults.py) which drive user-facing CLI defaults.
"""

# Hartree-Fock initialization pre-optimization
HF_FIDELITY_THRESHOLD = 0.9999
HF_PRE_OPT_ATTEMPTS = 10
HF_PRE_OPT_MAXITER = 1000

# L-BFGS-B optimizer defaults
LBFGSB_TIGHT_TOL = 1e-15
LBFGSB_DEFAULT_MAXITER = 15000

# COBYLA optimizer defaults
COBYLA_DEFAULT_MAXITER = 1000

# SLSQP optimizer defaults
SLSQP_DEFAULT_MAXITER = 100
