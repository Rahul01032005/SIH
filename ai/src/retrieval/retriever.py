class Retriever:
    def __init__(self, store, embedder): self.store, self.embedder = store, embedder
    def search(self, query: str, top_k: int = 5, project_id: str | None = None) -> list[dict]:
        # Retrieve extra candidates before applying an optional project-ID text/metadata filter.
        results = self.store.search(self.embedder.embed([query])[0], len(self.store.chunks) if project_id else top_k)
        if project_id:
            project_id = str(project_id)
            filtered = [item for item in results if project_id in item["text"] or str(item["metadata"].get("project_id", "")) == project_id]
            if filtered: results = filtered
        if "latest" in query.lower() and project_id:
            results.sort(key=lambda item: (item["metadata"].get("snapshot_month") or "", item["similarity_score"]), reverse=True)
        return results[:top_k]
