"""Create a clearly labelled synthetic dataset; it is not PAIMANA data."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
rng = np.random.default_rng(42); n = 180
physical = rng.uniform(.05, .98, n); financial = np.clip(physical + rng.normal(.08, .18, n), 0, 1.4)
data = pd.DataFrame({"project_id": [f"SYN-{i:03d}" for i in range(n)], "approved_cost": rng.integers(50, 500, n) * 1e6, "expenditure": lambda x: x.approved_cost * financial, "physical_progress": physical, "financial_progress": financial, "start_date": "2024-01-01", "planned_end_date": "2026-01-01"})
data["cost_overrun"] = (financial - physical > .14).astype(int); data["time_overrun"] = ((physical < .55) | (financial - physical > .22)).astype(int)
output = ROOT / "data" / "processed" / "synthetic_project_monitoring.csv"; output.parent.mkdir(parents=True, exist_ok=True); data.to_csv(output, index=False); print(f"Created synthetic (not PAIMANA) data: {output}")
