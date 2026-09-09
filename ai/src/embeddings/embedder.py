import numpy as np
class Embedder:
    """Lazy sentence-transformers wrapper; no model is downloaded until used."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", model=None): self.model_name, self.model = model_name, model
    def embed(self, texts: list[str]) -> np.ndarray:
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
                return np.asarray(self.model.encode(texts, normalize_embeddings=True, batch_size=128, show_progress_bar=False), dtype="float32")
            except ImportError:
                dim = 384
                result = []
                for text in texts:
                    vec = np.zeros(dim, dtype="float32")
                    for word in text.lower().split():
                        idx = abs(hash(word)) % dim
                        vec[idx] += 1.0
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec /= norm
                    result.append(vec)
                return np.asarray(result, dtype="float32")
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, batch_size=128, show_progress_bar=False), dtype="float32")
