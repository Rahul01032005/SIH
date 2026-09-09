from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAGConfig
from src.embeddings.embedder import Embedder
from src.vectorstore.faiss_store import FAISSStore
from src.retrieval.retriever import Retriever
from ml.src.prediction.predictor import predict_project_risk
from src.retrieval.project_lookup import ProjectLookup
from src.rag.pipeline import RAGPipeline
from src.llm.llm_client import configured_llm


question = " ".join(sys.argv[1:]) or "What does the project report say?"

project_match = re.search(r"\bproject\s+(\d{4,})\b", question, re.IGNORECASE)
project_id = project_match.group(1) if project_match else None

config = RAGConfig()

project_lookup = ProjectLookup(config.projects_csv_path)

project_data = (
    project_lookup.get_project(project_id)
    if project_id
    else []
)

risk_context = None

if project_data:
    project_data = sorted(
        project_data,
        key=lambda row: row.get("snapshot_month", "")
    )

    if len(project_data) > 1:
        prediction_snapshot = project_data[-2]
    else:
        prediction_snapshot = project_data[-1]

    risk_context = predict_project_risk(prediction_snapshot)


pipeline = RAGPipeline(
    Retriever(
        FAISSStore.load(config.vector_store_path),
        Embedder(config.embedding_model)
    ),
    configured_llm()
)

print(
    pipeline.ask(
    question,
    top_k=config.top_k,
    project_id=project_id,
    project_data=project_data,
    risk_context=risk_context,
)
)