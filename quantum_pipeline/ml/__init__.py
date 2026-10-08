"""ML utilities for VQE outcome prediction.

Requires `pdm install -G ml`.

Import from the submodules:
- `schema` for the input contract shared by both models,
- `energy_estimator` and `convergence_predictor` for the models themselves,
- `preprocessing` for the shared feature transformer,
- `tracking` for MLflow.

Both predictors ship a `generate_synthetic_trajectories`; they emit different feature
columns for their own models over the same `schema` contract.
"""
