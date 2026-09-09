import pandas as pd

def validation_report(frame: pd.DataFrame, required_columns: list[str] | None = None) -> dict:
    required_columns = required_columns or []
    return {"rows": len(frame), "columns": list(frame.columns), "duplicate_rows": int(frame.duplicated().sum()),
            "missing_by_column": frame.isna().sum().to_dict(), "missing_required": [c for c in required_columns if c not in frame.columns]}
