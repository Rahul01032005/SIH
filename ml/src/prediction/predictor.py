import joblib
import pandas as pd
from ..config import COST_FEATURES, MODELS_PATH, TIME_FEATURES
from ..features.engineering import build_features
from ..risk.early_warning import generate_warnings
from ..risk.risk_scorer import score_risk
from ..explainability.explainer import explain_top_factors

class ProjectRiskPredictor:
    def __init__(self, cost_model=None, time_model=None):
        self.cost_model = cost_model or joblib.load(MODELS_PATH / "cost_overrun_production.joblib")
        self.time_model = time_model or joblib.load(MODELS_PATH / "time_overrun_production.joblib")
    def predict(self, project_data: dict) -> dict:
        row = build_features(pd.DataFrame([project_data]))
        cost = float(self.cost_model.predict_proba(row[COST_FEATURES])[:,1][0])
        time = float(self.time_model.predict_proba(row[TIME_FEATURES])[:,1][0])
        enriched = row.iloc[0].to_dict(); factors = []
        if pd.notna(enriched.get("progress_gap")) and enriched["progress_gap"] >= .2: factors.append("High expenditure relative to physical progress is associated with higher predicted risk.")
        if pd.notna(enriched.get("months_to_target_completion")):
            months_to_target = float(enriched["months_to_target_completion"])

            if months_to_target < 0:
                factors.append(
                    "Target completion date has already passed, which is associated with higher predicted schedule risk."
                )
            elif months_to_target <= 6:
                factors.append(
                    "Target completion date is approaching, which is associated with higher predicted risk."
                )
        if pd.notna(enriched.get("progress_velocity")) and enriched["progress_velocity"] <= 1: factors.append("Recent physical progress is slow, which is associated with higher predicted risk.")
        return {"project_id":project_data.get("project_id"),"cost_risk":round(cost,4),"time_risk":round(time,4),**score_risk(cost,time),"warnings":generate_warnings(enriched,cost,time),"risk_factors":factors}

def predict_project_risk(project_snapshot: dict) -> dict:
    return ProjectRiskPredictor().predict(project_snapshot)
