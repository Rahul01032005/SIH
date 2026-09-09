from dataclasses import dataclass
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ML_ROOT.parent
DATASET_PATH = PROJECT_ROOT / "Data" / "processed_paimana_projects.csv"
MODELS_PATH = ML_ROOT / "models"
RANDOM_STATE = 42
PREDICTION_MONTHS = ("2026-04", "2026-05", "2026-06")
OUTCOME_MONTH = "2026-07"
REQUIRED_COLUMNS = {"project_id", "project_name", "agency", "state", "snapshot_month", "start_date", "target_completion_date", "revised_completion_date", "original_cost", "revised_cost", "cumulative_expenditure", "physical_progress"}
COST_FEATURES = ["original_cost", "cumulative_expenditure", "expenditure_ratio", "physical_progress", "progress_gap", "absolute_progress_gap", "project_age_months", "months_to_target_completion", "progress_velocity", "expenditure_velocity", "expenditure_ratio_velocity", "state", "agency", "snapshot_month"]
TIME_FEATURES = COST_FEATURES.copy()
RISK_LEVELS = ((25, "LOW"), (50, "MEDIUM"), (75, "HIGH"), (101, "CRITICAL"))
HIGH_PROBABILITY, SPEND_PROGRESS_GAP = .70, .20
APPROACHING_TARGET_MONTHS, LOW_PROGRESS, SLOW_PROGRESS_VELOCITY = 6, 25.0, 1.0

@dataclass(frozen=True)
class MLConfig:
    random_state: int = RANDOM_STATE
    test_size: float = .25
