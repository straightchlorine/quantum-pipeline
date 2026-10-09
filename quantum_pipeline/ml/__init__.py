"""ML utilities for VQE outcome prediction.

Requires `pdm install -G ml`.

Import from the submodules:
- `schema` for the input contract shared by both models,
- `energy` and `convergence` for the two models (features, models, synthetic data, results),
- `preprocessing`, `fitting` and `trajectory` for shared feature and data handling,
- `registry` for model lookup and `tracking` for MLflow.

Both predictors ship a `generate_synthetic_trajectories`; they emit different feature
columns for their own models over the same `schema` contract.
"""
