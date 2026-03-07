"""
Document ingestion pipeline for Eureka Legal Agent.

Processes PDF and DOCX files from a folder, chunks them with legal-aware
separators, generates dense + sparse embeddings with BGE-M3, and upserts
into Qdrant for hybrid search.

Usage:
    cd backend/
    python -m src.ingestion.ingest --docs-path src/ingestion/docs/
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from uuid import uuid4

import PyPDF2
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.agent.embedder import BGEEmbedder
from src.config import get_settings
from src.db.qdrant import QdrantStore
from src.domain.errors import IngestionError

logger = logging.getLogger(__name__)

LEGAL_SEPARATORS = [
    "\n\nARTÍCULO",
    "\n\nArtículo",
    "\n\nCONSIDERANDO",
    "\n\nHECHOS",
    "\n\nPRETENSIONES",
    "\n\nFUNDAMENTOS",
    "\n\nDECIDE",
    "\n\nRESOLVE",
    "\n\n",
    "\n",
    " ",
    "",
]


def _read_pdf(path: Path) -> str:
    """Extract text from a PDF file."""
    text_parts: list[str] = []
    try:
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")
        return "\n\n".join(text_parts)
    except Exception as exc:
        raise IngestionError(f"Failed to read PDF: {exc}", filename=path.name) from exc


def _read_docx(path: Path) -> str:
    """Extract text from a DOCX file."""
    try:
        doc = Document(str(path))
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as exc:
        raise IngestionError(f"Failed to read DOCX: {exc}", filename=path.name) from exc


def _detect_doc_type(filename: str) -> str:
    """Infer document type from filename."""
    name = filename.lower()
    if "tutela" in name:
        return "tutela"
    if "demanda" in name:
        return "demanda"
    if "sentencia" in name:
        return "sentencia"
    if "contrato" in name:
        return "contrato"
    if "peticion" in name or "petición" in name:
        return "derecho_de_peticion"
    return "general"


async def ingest_file(
    path: Path,
    embedder: BGEEmbedder,
    store: QdrantStore,
    chunk_size: int,
    chunk_overlap: int,
) -> int:
    """Ingest a single file. Returns the number of chunks indexed."""
    logger.info("Processing: %s", path.name)

    if path.suffix.lower() == ".pdf":
        text = _read_pdf(path)
    elif path.suffix.lower() == ".docx":
        text = _read_docx(path)
    else:
        logger.warning("Skipping unsupported format: %s", path.suffix)
        return 0

    if not text.strip():
        logger.warning("Empty content in %s, skipping", path.name)
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=LEGAL_SEPARATORS,
    )
    chunks = splitter.split_text(text)
    doc_type = _detect_doc_type(path.name)

    for i, chunk in enumerate(chunks):
        dense, sparse_i, sparse_v = await embedder.embed(chunk)
        payload = {
            "text": chunk,
            "source_file": path.name,
            "doc_type": doc_type,
            "chunk_index": i,
        }
        await store.upsert_chunk(
            point_id=str(uuid4()),
            dense_vector=dense,
            sparse_indices=sparse_i,
            sparse_values=sparse_v,
            payload=payload,
        )
        if (i + 1) % 10 == 0:
            logger.info("  → %d/%d chunks indexed", i + 1, len(chunks))

    logger.info("✓ %s — %d chunks indexed (type: %s)", path.name, len(chunks), doc_type)
    return len(chunks)


async def run_ingestion(docs_path: Path) -> None:
    """Run the full ingestion pipeline over all supported files in docs_path."""
    settings = get_settings()
    embedder = BGEEmbedder(model=settings.embed_model, ollama_host=settings.ollama_base_url)
    store = QdrantStore.from_url(settings.qdrant_url, settings.qdrant_collection)

    await store.ensure_collection()

    supported = [".pdf", ".docx"]
    files = [f for f in docs_path.iterdir() if f.suffix.lower() in supported]

    if not files:
        logger.info("No PDF or DOCX files found in %s", docs_path)
        return

    logger.info("Found %d files to ingest", len(files))
    total_chunks = 0
    for file_path in sorted(files):
        chunks_added = await ingest_file(
            path=file_path,
            embedder=embedder,
            store=store,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        total_chunks += chunks_added

    logger.info("Ingestion complete. Total chunks indexed: %d", total_chunks)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Ingest legal documents into Qdrant")
    parser.add_argument(
        "--docs-path",
        type=Path,
        default=Path("src/ingestion/docs"),
        help="Path to folder containing PDF/DOCX files",
    )
    args = parser.parse_args()

    if not args.docs_path.exists():
        logger.error("Docs path does not exist: %s", args.docs_path)
        sys.exit(1)

    asyncio.run(run_ingestion(args.docs_path))


if __name__ == "__main__":
    main()
