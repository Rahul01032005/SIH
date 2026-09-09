from src.config import COST_FEATURES, TIME_FEATURES
def test_leakage_fields_are_excluded():
    assert "revised_cost" not in COST_FEATURES
    assert "revised_completion_date" not in TIME_FEATURES
