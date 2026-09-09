from src.ingestion.chunker import chunk_text
def test_chunker_preserves_metadata():
    chunks = chunk_text("abcdefghij", {"project_id":"P1", "document_name":"note.txt"}, chunk_size=6, overlap=2)
    assert len(chunks) == 2 and chunks[1]["metadata"] == {"project_id":"P1", "document_name":"note.txt", "chunk_index":1}
