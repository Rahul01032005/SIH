def chunk_text(text: str, metadata: dict | None = None, chunk_size: int = 500, overlap: int = 75) -> list[dict]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size: raise ValueError("chunk_size must be positive and overlap must be smaller than it.")
    chunks, start, index = [], 0, 0
    while start < len(text):
        end = min(len(text), start + chunk_size); piece = text[start:end].strip()
        if piece:
            chunk_metadata = {**(metadata or {}), "chunk_index": index}
            if chunk_metadata.get("source_file"): chunk_metadata["chunk_id"] = f"{chunk_metadata['source_file']}#{index}"
            chunks.append({"text": piece, "metadata": chunk_metadata}); index += 1
        if end == len(text): break
        start = end - overlap
    return chunks
