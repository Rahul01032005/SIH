"""Validation and supervised-data construction; July is targets only, never features."""
from pathlib import Path
import pandas as pd
import numpy as np
from .config import DATASET_PATH, OUTCOME_MONTH, PREDICTION_MONTHS, REQUIRED_COLUMNS
from .features.engineering import build_features

def load_and_validate(path: str | Path = DATASET_PATH) -> pd.DataFrame:
    data = pd.read_csv(path); missing = REQUIRED_COLUMNS - set(data.columns)
    if missing: raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if len(data) != 7590: raise ValueError(f"Expected 7,590 rows; found {len(data)}")
    data["snapshot_month"] = data["snapshot_month"].astype(str).str[:7]
    expected = {"2026-04":1981,"2026-05":1987,"2026-06":1847,"2026-07":1775}
    if data.snapshot_month.value_counts().to_dict() != expected: raise ValueError("Unexpected monthly counts")
    if data.project_id.isna().any(): raise ValueError("project_id contains null values")
    if not data.physical_progress.dropna().between(0,100).all(): raise ValueError("physical_progress must be 0..100")
    for column in ("original_cost", "cumulative_expenditure"):
        if not pd.to_numeric(data[column],errors="coerce").dropna().ge(0).all(): raise ValueError(f"{column} must be non-negative")
    return data

def build_supervised_dataset(path: str | Path = DATASET_PATH) -> pd.DataFrame:
    raw = load_and_validate(path)
    july = raw[raw.snapshot_month.eq(OUTCOME_MONTH)].copy()
    july["cost_overrun_by_july"] = (july.revised_cost > july.original_cost).astype(float)
    july.loc[july.revised_cost.isna() | july.original_cost.isna(), "cost_overrun_by_july"] = np.nan
    target, revised = pd.to_datetime(july.target_completion_date,errors="coerce"), pd.to_datetime(july.revised_completion_date,errors="coerce")
    july["time_overrun_by_july"] = (revised > target).astype(float)
    july.loc[target.isna() | revised.isna(), "time_overrun_by_july"] = np.nan
    features = build_features(raw[raw.snapshot_month.isin(PREDICTION_MONTHS)])
    return features.merge(july[["project_id","cost_overrun_by_july","time_overrun_by_july"]],on="project_id",how="inner",validate="many_to_one")
