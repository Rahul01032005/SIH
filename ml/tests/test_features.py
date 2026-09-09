import pandas as pd
import pytest
from src.features.engineering import build_features
def test_financial_and_trend_features():
    result = build_features(pd.DataFrame({"project_id":["P","P"],"snapshot_month":["2026-04","2026-05"],"start_date":["2025-01","2025-01"],"target_completion_date":["2026-12","2026-12"],"original_cost":[100,100],"cumulative_expenditure":[50,70],"physical_progress":[40,50]}))
    assert result.expenditure_ratio.tolist() == [.5,.7] and result.progress_gap.tolist() == pytest.approx([.1,.2])
    assert result.progress_velocity.iloc[1] == 10 and result.expenditure_velocity.iloc[1] == 20
