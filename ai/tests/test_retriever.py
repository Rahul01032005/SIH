import numpy as np
from src.vectorstore.faiss_store import FAISSStore
from src.retrieval.retriever import Retriever
class TestEmbedder:
    def embed(self, texts): return np.array([[1,0] if "delay" in t else [0,1] for t in texts], dtype="float32")
def test_retrieval_keeps_text_score_and_metadata():
    store = FAISSStore(); store.add([{"text":"Delay due to land issue", "metadata":{"project_id":"P1"}}, {"text":"Work completed", "metadata":{"project_id":"P2"}}], TestEmbedder().embed(["delay", "work"]))
    found = Retriever(store, TestEmbedder()).search("delay", 1)[0]
    assert found["metadata"]["project_id"] == "P1" and "similarity_score" in found
