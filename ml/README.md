# PAIMANA predictive analytics

This module predicts July 2026 cost and schedule-overrun risk from April–June 2026 project snapshots in `Data/processed_paimana_projects.csv`. July creates ground truth only: cost overrun is July `revised_cost > original_cost`; time overrun is July `revised_completion_date > target_completion_date`.

Features are original cost, expenditure, expenditure/progress mismatch, project timeline, prior-snapshot velocity, state, agency and snapshot month. `revised_cost` and `revised_completion_date` are deliberately excluded; July values never enter feature rows. Projects, not rows, are split via deterministic `GroupShuffleSplit`, so train/test project IDs cannot overlap.

Two pipelines per target are evaluated: class-balanced Logistic Regression and Random Forest. Numeric values use median imputation and scaling; categories use most-frequent imputation and unknown-safe one-hot encoding. Selection uses F1, then PR-AUC and ROC-AUC. `models/metrics.json` and `models/evaluation_report.md` record accuracy, precision, recall, F1, ROC-AUC, PR-AUC and confusion matrices. Pipeline-aware coefficient/feature-importance utilities describe model signals, never causal claims.

Run from `ml/`:

`python scripts/build_ml_dataset.py`

`python scripts/train_models.py`

`python scripts/predict.py`

`python -m pytest tests`

Backend callers can use `src.prediction.predictor.predict_project_risk(snapshot)`. It returns probabilities, a 50/50 0–100 score, LOW (0–24), MEDIUM (25–49), HIGH (50–74), or CRITICAL (75–100), condition-based warnings, and carefully worded associated risk factors.

The model is trained only on available April–July 2026 observations. More historical periods would improve robustness. Outputs are risk estimates, not certainties, and should support—not replace—human monitoring decisions.
