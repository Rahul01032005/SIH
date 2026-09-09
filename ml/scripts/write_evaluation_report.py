from pathlib import Path
import json, sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.training import write_evaluation_report

if __name__ == "__main__":
    write_evaluation_report(json.loads((ROOT / "models" / "metrics.json").read_text(encoding="utf-8")))
