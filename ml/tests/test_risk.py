from src.risk.risk_scorer import score_risk
from src.risk.early_warning import generate_warnings
def test_high_risk_has_warning():
    assert score_risk(.9, .8)["risk_level"] == "CRITICAL"
    assert "High probability of cost overrun." in generate_warnings({"physical_progress": 10}, .8, None)
