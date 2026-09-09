from pathlib import Path
import json
import numpy as np
try:
    import faiss
except ImportError: faiss = None

class FAISSStore:
    def __init__(self, dimension: int | None = None): self.dimension, self.index, self.chunks = dimension, None, []
    def add(self, chunks: list[dict], embeddings: np.ndarray):
        embeddings = np.asarray(embeddings, dtype="float32")
        if len(chunks) != len(embeddings): raise ValueError("Every chunk needs one embedding.")
        if faiss is None:
            self.index = embeddings if self.index is None else np.vstack([self.index, embeddings])
        else:
            if self.index is None: self.dimension = embeddings.shape[1]; self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(embeddings)
        self.chunks.extend(chunks)
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        if not self.chunks: return []
        q = np.asarray(query_embedding, dtype="float32").reshape(1, -1); k = min(top_k, len(self.chunks))
        if faiss is None:
            scores = self.index @ q[0]; ids = np.argsort(scores)[::-1][:k]; scores = scores[ids]
        else: scores, ids = self.index.search(q, k); scores, ids = scores[0], ids[0]
        return [{"text": self.chunks[int(i)]["text"], "metadata": self.chunks[int(i)]["metadata"], "similarity_score": float(s)} for s, i in zip(scores, ids) if i >= 0]
    def save(self, directory: str | Path):
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True); (directory / "chunks.json").write_text(json.dumps(self.chunks), encoding="utf-8")
        if faiss is None: np.save(directory / "vectors.npy", self.index)
        else: faiss.write_index(self.index, str(directory / "index.faiss"))
    @classmethod
    def load(cls, directory: str | Path):
        directory = Path(directory); store = cls(); store.chunks = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        if faiss is not None and (directory / "index.faiss").exists():
            store.index = faiss.read_index(str(directory / "index.faiss")); store.dimension = store.index.d
        elif (directory / "vectors.npy").exists():
            store.index = np.load(directory / "vectors.npy"); store.dimension = store.index.shape[1]
        else:
            store.index = None; store.dimension = 384
        return store
