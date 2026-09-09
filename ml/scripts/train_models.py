from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.dataset import build_supervised_dataset
from src.training import train_all

if __name__ == "__main__":
    metrics = train_all(build_supervised_dataset())
    print(__import__("json").dumps(metrics, indent=2))
