from pathlib import Path
import sys, json
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.dataset import build_supervised_dataset
from src.prediction.predictor import predict_project_risk
if __name__ == "__main__":
    row = build_supervised_dataset().iloc[0].drop(labels=["cost_overrun_by_july","time_overrun_by_july"]).to_dict()
    print(json.dumps(predict_project_risk(row),indent=2))
