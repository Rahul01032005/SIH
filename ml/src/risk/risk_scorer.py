from ..config import RISK_LEVELS
def score_risk(cost_probability: float | None, time_probability: float | None, **_) -> dict:
    score = round(max(0, min(100, 50 * float(cost_probability or 0) + 50 * float(time_probability or 0))), 1)
    return {"risk_score":score, "risk_level":next(label for upper,label in RISK_LEVELS if score < upper)}
