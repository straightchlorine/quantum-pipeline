"""
Named constants for quantum simulation algorithm parameters.

These are physics/algorithm constants used internally by the simulation engine.
They are separate from DEFAULTS (in defaults.py) which drive user-facing CLI defaults.
"""

# Hartree-Fock initialization pre-optimization
HF_FIDELITY_THRESHOLD = 0.9999
HF_PRE_OPT_ATTEMPTS = 10
HF_PRE_OPT_MAXITER = 1000

# ExcitationPreserving initial-parameter jitter.
# Zero params sit exactly on HF, where the gradient vanishes and the optimizer
# stalls. A small jitter nudges it off HF so it can descend into correlation.
# 0.1 too timid to escape reliably; 0.5 reaches chemical accuracy across seeds.
EP_INIT_JITTER = 0.5
# The `hf` arm: starts within a few mHa of HF. Measured 2026-10-08 on H2/sto3g with COBYLA it
# stays within ~8 mHa of HF (the Brillouin plateau); the 0.5 jitter above is the `random` arm.
EP_HF_INIT_JITTER = 0.01

# L-BFGS-B optimizer defaults
LBFGSB_DEFAULT_MAXITER = 15000

# COBYLA optimizer defaults
COBYLA_DEFAULT_MAXITER = 1000

# SLSQP optimizer defaults
SLSQP_DEFAULT_MAXITER = 100

# Newton-CG optimizer defaults
NEWTON_CG_DEFAULT_MAXITER = 1000

# trust-constr optimizer defaults
TRUSTCONSTR_DEFAULT_MAXITER = 1000

# Optimizers capable of using an analytic gradient and need no Hessian.
# (matrix of second derivatives of the energy with respect to ansatz params)
#
# The fast path is the exact statevector run that computes energies and gradients
# analytically instead of sampling shots.
#
# Dogleg, trust-ncg, trust-exact and trust-krylov need one, so they are excluded.
# Newton-CG needs a gradient, so it only runs on the fast path.
#
# Every name here must also be configurable through OptimizerConfigFactory.
GRADIENT_OPTIMIZERS = frozenset(
    {'L-BFGS-B', 'BFGS', 'CG', 'TNC', 'SLSQP', 'Newton-CG', 'trust-constr'}
)

# Largest problem the fast path takes. ReverseEstimatorGradient builds a sparse matrix of the
# Hamiltonian on every call; at 20 qubits (H2 cc-pVDZ) that passed 11 GiB RSS and was OOM-killed.
# Larger problems fall back to the Aer exact path, whose statevector is 16 MiB at 20 qubits.
FAST_PATH_MAX_QUBITS = 16

# The backend name stored in VQEInitialData.backend for fast-path runs.
# A fast-path run has no Aer backend object, so it needs a name of its own.
# This lets the dataset tell a fast-path run from an Aer run without adding a field.
FAST_PATH_BACKEND = 'statevector_analytic'
