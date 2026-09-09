"""CSV loading with clear errors for data that has not arrived yet."""
from pathlib import Path
import pandas as pd

def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Add the real PAIMANA export when it is available.")
    return pd.read_csv(path)
