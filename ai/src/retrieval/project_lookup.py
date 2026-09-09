from pathlib import Path
import pandas as pd


class ProjectLookup:
    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Project CSV not found: {self.csv_path}"
            )

        self.data = pd.read_csv(self.csv_path)

        self.data["project_id"] = self.data["project_id"].astype(str)

    def get_project(self, project_id: str) -> list[dict]:
        project_id = str(project_id)

        rows = self.data[
            self.data["project_id"] == project_id
        ].copy()

        if rows.empty:
            return []

        rows = rows.sort_values("snapshot_month")

        return rows.to_dict(orient="records")