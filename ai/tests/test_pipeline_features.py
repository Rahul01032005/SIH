import numpy as np
from src.vectorstore.faiss_store import FAISSStore
from src.retrieval.retriever import Retriever
from src.rag.pipeline import RAGPipeline

class Embedder:
    def embed(self, texts): return np.array([[1, 0] for _ in texts], dtype="float32")

class FakeLLM:
    def __init__(self): self.prompt = ""
    def generate(self, prompt): self.prompt = prompt; return "Grounded answer"

def test_project_filter_sources_and_risk_context_are_grounded():
    store = FAISSStore(); store.add([
        {"text":"612786 July status", "metadata":{"project_id":"612786","source_file":"July.pdf","snapshot_month":"2026-07","chunk_id":"July#1"}},
        {"text":"Other project", "metadata":{"project_id":"X","source_file":"April.pdf"}},
    ], Embedder().embed(["a", "b"]))
    llm = FakeLLM(); result = RAGPipeline(Retriever(store, Embedder()), llm).ask("latest status", project_id="612786", risk_context={"risk_level":"HIGH"})
    assert result["answer"] == "Grounded answer" and result["sources"][0]["snapshot_month"] == "2026-07"
    assert "PAIMANA context" in llm.prompt and "ML risk context" in llm.prompt
