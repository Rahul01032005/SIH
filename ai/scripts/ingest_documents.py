from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import RAGConfig
from src.ingestion.document_loader import load_directory
from src.ingestion.chunker import chunk_text
from src.embeddings.embedder import Embedder
from src.vectorstore.faiss_store import FAISSStore
config = RAGConfig(); documents = load_directory(config.raw_pdf_path); chunks = [chunk for doc in documents for chunk in chunk_text(doc["text"], doc["metadata"], config.chunk_size, config.chunk_overlap)]
if not chunks: raise SystemExit("No readable PAIMANA PDFs found in Data/Raw.")
store = FAISSStore(); store.add(chunks, Embedder(config.embedding_model).embed([c["text"] for c in chunks])); store.save(config.vector_store_path); print(f"Ingested {len(chunks)} chunks.")
