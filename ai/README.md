# PAIMANA Project Intelligence Assistant

This RAG module ingests the four authoritative PAIMANA PDFs in `Data/Raw`, extracts text, creates overlapping metadata-preserving chunks, embeds them, and persists a FAISS similarity index plus `chunks.json`. FAISS is a vector-similarity index, not the primary relational database.

From `ai/`, run `python scripts/ingest_documents.py`, then `python scripts/ask.py "What is the latest available information about project 612786?"`. Retrieval returns the top five relevant chunks and source file, month, chunk ID, and similarity score. Project IDs act as an optional retrieval filter.

Set `GROQ_API_KEY` (optionally `GROQ_MODEL`) to enable grounded Groq generation. No key is stored in code. Without it, the assistant operates safely in retrieval-only mode and returns context/sources rather than inventing an answer.

`RAGPipeline.ask(question, project_id=None, risk_context=None)` accepts ML output as structured context. Answers must distinguish ML estimates from PAIMANA facts and never claim a prediction proves causation. The available April–July 2026 reports limit coverage; missing context results in an explicit insufficiency response.
