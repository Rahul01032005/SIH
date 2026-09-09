from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAGConfig
from src.embeddings.embedder import Embedder
from src.vectorstore.faiss_store import FAISSStore

def build_vector_store():
    config = RAGConfig()
    if not config.projects_csv_path.exists():
        print(f"Projects CSV not found at {config.projects_csv_path}")
        return

    df = pd.read_csv(config.projects_csv_path)

    important_ids = ['705728', '702668', '612786']
    subset = df[df['project_id'].astype(str).isin(important_ids) | df['original_cost'].gt(10000)].head(150)

    chunks = []
    for _, row in subset.iterrows():
        p_id = str(row['project_id'])
        name = row.get('project_name', 'Unknown')
        agency = row.get('agency', 'Unknown')
        state = row.get('state', 'Unknown')
        month = row.get('snapshot_month', '2026-06')
        orig_cost = row.get('original_cost', '')
        rev_cost = row.get('revised_cost', '')
        cum_exp = row.get('cumulative_expenditure', '')
        prog = row.get('physical_progress', '')

        text = (
            f"Project {p_id}: {name}. Implementing agency: {agency}. Location: {state}. "
            f"Monitoring snapshot month: {month}. Approved cost: {orig_cost} crore. "
            f"Revised cost: {rev_cost} crore. Cumulative expenditure: {cum_exp} crore. "
            f"Physical progress: {prog}%."
        )

        chunks.append({
            "text": text,
            "metadata": {
                "project_id": p_id,
                "project_name": str(name),
                "snapshot_month": str(month),
                "source_file": f"paimana_{month}.pdf",
                "chunk_id": f"{p_id}_{month}",
            }
        })

    store = FAISSStore()
    embeddings = Embedder().embed([c["text"] for c in chunks])
    store.add(chunks, embeddings)
    store.save(config.vector_store_path)
    print(f"Vector store initialized with {len(chunks)} chunks at {config.vector_store_path}")

if __name__ == "__main__":
    build_vector_store()
