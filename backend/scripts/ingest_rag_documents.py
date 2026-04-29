"""Ingestion script: embed and store RAG documents into Postgres.

Usage (from backend/ directory):
    uv run python scripts/ingest_rag_documents.py

Source files live in backend/rag_data/raw/*.txt.
Each file must begin with key: value header lines, then a --- separator, then body text.
Required headers: destination, source_title, source_url
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import structlog

# Make the `app` package importable when the script is run from backend/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.db.session import create_engine, create_session_factory  # noqa: E402
from app.services.rag_service import RagService  # noqa: E402

logger = structlog.get_logger(__name__)

RAW_DIR = Path(__file__).resolve().parent.parent / "rag_data" / "raw"


def parse_document_file(path: Path) -> tuple[dict[str, str], str]:
    """Return (headers, body) from a document file with key: value front-matter."""
    text = path.read_text(encoding="utf-8")
    if "---" not in text:
        raise ValueError(f"No '---' separator found in {path.name}")
    header_block, _, body = text.partition("---")
    headers: dict[str, str] = {}
    for line in header_block.strip().splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            headers[key.strip().lower()] = value.strip()
    required = {"destination", "source_title", "source_url"}
    missing = required - set(headers)
    if missing:
        raise ValueError(f"Missing required headers {missing} in {path.name}")
    return headers, body.strip()


async def main() -> None:
    from sentence_transformers import (
        SentenceTransformer,
    )

    settings = get_settings()
    txt_files = sorted(RAW_DIR.glob("*.txt"))

    if not txt_files:
        logger.warning("rag.ingest.no_files", directory=str(RAW_DIR))
        return

    logger.info("rag.ingest.start", file_count=len(txt_files))

    embedder = await asyncio.to_thread(SentenceTransformer, settings.embedding_model)
    service = RagService(embedder=embedder)

    engine = create_engine()
    factory = create_session_factory(engine)

    async with factory() as session:
        for path in txt_files:
            try:
                headers, body = parse_document_file(path)
                document, chunks = await service.store_document_with_chunks(
                    session=session,
                    destination_name=headers["destination"],
                    source_title=headers["source_title"],
                    source_url=headers["source_url"],
                    raw_text=body,
                )
                await session.commit()
                logger.info(
                    "rag.ingest.file_done",
                    file=path.name,
                    document_id=str(document.id),
                    chunks=len(chunks),
                )
            except Exception:
                await session.rollback()
                logger.exception("rag.ingest.file_error", file=path.name)

    await engine.dispose()
    logger.info("rag.ingest.complete", total_files=len(txt_files))


if __name__ == "__main__":
    asyncio.run(main())
