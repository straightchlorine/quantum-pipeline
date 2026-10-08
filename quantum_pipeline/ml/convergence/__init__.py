"""Convergence predictor: will a VQE run converge, given only its first K iterations?

Binary classification at horizons K in {10, 20, 50}, evaluated leave-one-molecule-out
so a run is never trained alongside another run of the same molecule.

- `features`   iteration rows -> one row per run, using only steps 1..K
- `models`     XGBoost, RandomForest and LogisticRegression pipelines
- `predictor`  `ConvergencePredictor`, the training and evaluation entry point
- `results`    `FoldResult`, `ConvergencePredictorResults`
- `synthetic`  fake trajectories for development

The `converged` label is the optimizer's `success` flag, not agreement with a
reference energy. See `quantum_pipeline.ml.schema`.
"""
