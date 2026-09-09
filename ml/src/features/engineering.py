"""Leakage-safe snapshot features for PAIMANA longitudinal project data."""
import numpy as np
import pandas as pd

def build_features(frame: pd.DataFrame, reference_date: str | None = None) -> pd.DataFrame:
    data = frame.copy()
    if not {"project_id", "snapshot_month"}.issubset(data): return data
    data["snapshot_month"] = data["snapshot_month"].astype(str).str[:7]
    data["_snapshot_date"] = pd.to_datetime(data["snapshot_month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    for column in ("original_cost", "cumulative_expenditure", "physical_progress"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["expenditure_ratio"] = data["cumulative_expenditure"] / data["original_cost"].replace(0, np.nan)
    data["progress_gap"] = data["expenditure_ratio"] - data["physical_progress"] / 100.0
    data["absolute_progress_gap"] = data["progress_gap"].abs()
    start = pd.to_datetime(data["start_date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    target = pd.to_datetime(data["target_completion_date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    data["project_age_months"] = (data._snapshot_date.dt.year-start.dt.year)*12 + data._snapshot_date.dt.month-start.dt.month
    data["months_to_target_completion"] = (target.dt.year-data._snapshot_date.dt.year)*12 + target.dt.month-data._snapshot_date.dt.month
    ordered = data.sort_values(["project_id", "_snapshot_date"], kind="stable")
    grouped = ordered.groupby("project_id", sort=False)
    for feature, source in (("progress_velocity", "physical_progress"), ("expenditure_velocity", "cumulative_expenditure"), ("expenditure_ratio_velocity", "expenditure_ratio")):
        ordered[feature] = grouped[source].diff()
    return ordered.sort_index().drop(columns=["_snapshot_date"])

def build_targets(frame: pd.DataFrame, cost_target: str = "cost_overrun", time_target: str = "time_overrun") -> pd.DataFrame:
    """Return explicit targets only; target definitions await PAIMANA data governance."""
    return frame[[c for c in (cost_target, time_target) if c in frame.columns]].copy()
