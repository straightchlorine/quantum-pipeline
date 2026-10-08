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
