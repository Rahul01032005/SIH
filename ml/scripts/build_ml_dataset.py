from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.dataset import build_supervised_dataset

if __name__ == "__main__":
    data = build_supervised_dataset(); path = ROOT / "data" / "processed" / "july_prediction_dataset.csv"
    path.parent.mkdir(parents=True, exist_ok=True); data.to_csv(path,index=False)
    print(f"Built {len(data)} feature rows at {path}")
