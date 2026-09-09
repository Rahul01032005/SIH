from pathlib import Path
import re

def _month_from_filename(name: str) -> str | None:
    match = re.search(r"(April|May|June|July).*?(2026)", name, re.I)
    if not match: return None
    return f"2026-{['January','February','March','April','May','June','July'].index(match.group(1).title())+1:02d}"

def load_pdf_document(path: str | Path) -> dict:
    """Extract local PAIMANA PDF text with per-document monitoring-month metadata."""
    path = Path(path)
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF ingestion requires pypdf; install ai/requirements.txt.") from exc
    text = "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return {"text": text, "metadata": {"source_file": path.name, "document_name": path.name, "snapshot_month": _month_from_filename(path.name), "source": str(path)}}
def load_text_document(path: str | Path, project_id: str | None = None) -> dict:
    path = Path(path)
    if not path.exists(): raise FileNotFoundError(f"Document not found: {path}")
    if path.suffix.lower() == ".pdf": return load_pdf_document(path)
    if path.suffix.lower() not in {".txt", ".md"}: raise ValueError("Supported formats are PDF, text and Markdown.")
    return {"text": path.read_text(encoding="utf-8"), "metadata": {"source_file":path.name,"document_name": path.name, "project_id": project_id, "source": str(path)}}
def load_directory(directory: str | Path) -> list[dict]:
    return [load_text_document(p) for p in Path(directory).glob("*") if p.suffix.lower() in {".txt", ".md", ".pdf"}]
