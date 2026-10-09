"""Energy estimator: predict a run's final ground-state energy from a partial trajectory.

Regression on trajectories truncated at 25%, 50% and 75% completion, evaluated
leave-one-molecule-out so a molecule is never in both train and test.

- `features`   trajectory rows -> one row per run at a completion fraction
- `models`     XGBoost and Ridge pipelines
- `estimator`  `EnergyEstimator`, the training and evaluation entry point
- `results`    `EvaluationResult`, `LomoPredictions`, `EnergyEstimatorResults`
- `synthetic`  fake trajectories for development

The `converged` label is the optimizer's `success` flag, not agreement with a
reference energy. See `quantum_pipeline.ml.schema`.
"""
