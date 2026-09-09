import re
import pandas as pd

def normalize_column_name(name: object) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower())).strip("_")

def clean_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize headers, remove exact duplicates, and parse common values."""

    data = frame.copy()

    # Normalize column names
    data.columns = [normalize_column_name(c) for c in data.columns]

    # Remove exact duplicate rows
    data = data.drop_duplicates().reset_index(drop=True)

    for column in data.columns:
        if not pd.api.types.is_string_dtype(data[column]):
            continue

        values = data[column].astype(str).str.strip()

        # Percentage / progress / ratio columns
        if any(token in column for token in ("percent", "percentage", "progress", "ratio")):
            numeric_values = pd.to_numeric(
                values.str.replace("%", "", regex=False),
                errors="coerce"
            )

            if values.str.contains("%", regex=False).any():
                numeric_values = numeric_values / 100
            elif numeric_values.max() > 1:
                numeric_values = numeric_values / 100

            data[column] = numeric_values

        # Date columns
        elif (
    column.endswith("_date")
    or column in ("date", "start", "end", "deadline")
    or column.endswith("_deadline")
):

            data[column] = pd.to_datetime(
                values,
                errors="coerce"
            )

        # Monetary/numeric columns
        elif any(token in column for token in (
            "cost", "amount", "budget", "expenditure", "value"
        )):

            data[column] = pd.to_numeric(
                values.str.replace(r"[^0-9.\-]", "", regex=True),
                errors="coerce"
            )

    return data