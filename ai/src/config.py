from dataclasses import dataclass
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
@dataclass
class RAGConfig:
    documents_path: Path = ROOT / "data" / "documents"
    vector_store_path: Path = ROOT / "data" / "vector_store"
    embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    chunk_size: int = 1200
    chunk_overlap: int = 150
    top_k: int = 5
    raw_pdf_path: Path = PROJECT_ROOT / "Data" / "Raw"
    projects_csv_path: Path = PROJECT_ROOT / "Data" / "processed_paimana_projects.csv"