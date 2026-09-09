import numpy as np
from src.vectorstore.faiss_store import FAISSStore
from src.retrieval.retriever import Retriever
from src.rag.pipeline import RAGPipeline
class E:
    def embed(self, x): return np.array([[1,0] for _ in x], dtype="float32")
def test_rag_is_retrieval_only_without_llm():
    store=FAISSStore(); store.add([{"text":"Status: delayed", "metadata":{"source":"a.txt"}}], E().embed(["x"]))
    result=RAGPipeline(Retriever(store,E())).ask("status")
    assert "No LLM configured" in result["answer"] and result["sources"][0]["source"] == "a.txt"
