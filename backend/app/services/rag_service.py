from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db import DestinationChunk, DestinationDocument
from app.schemas.rag import ChunkResult

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 75

logger = structlog.get_logger(__name__)


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping fixed-size character chunks."""
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


class RagService:
    def __init__(self, embedder: SentenceTransformer) -> None:
        self._embedder = embedder

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        result = self._embedder.encode(texts, convert_to_numpy=True)
        return cast(list[list[float]], result.tolist())

    def _embed_single(self, text: str) -> list[float]:
        result = self._embedder.encode([text], convert_to_numpy=True)
        return cast(list[float], result[0].tolist())

    async def store_document_with_chunks(
        self,
        session: AsyncSession,
        destination_name: str,
        source_title: str,
        source_url: str,
        raw_text: str,
    ) -> tuple[DestinationDocument, list[DestinationChunk]]:
        """Store a document and all its embedded chunks in one transaction boundary.

        Caller owns session.commit(). Uses flush() so document.id is available
        for the FK on chunks without committing early.
        """
        document = DestinationDocument(
            destination_name=destination_name,
            source_title=source_title,
            source_url=source_url,
            raw_text=raw_text,
        )
        session.add(document)
        await session.flush()

        chunks = await self._create_chunks(session, document)

        logger.info(
            "rag.document_stored",
            document_id=str(document.id),
            destination=destination_name,
            chunk_count=len(chunks),
        )
        return document, chunks

    async def _create_chunks(
        self,
        session: AsyncSession,
        document: DestinationDocument,
    ) -> list[DestinationChunk]:
        texts = chunk_text(document.raw_text)
        embeddings: list[list[float]] = await asyncio.to_thread(self._embed_batch, texts)
        db_chunks: list[DestinationChunk] = []
        for idx, (text, vector) in enumerate(zip(texts, embeddings, strict=True)):
            chunk = DestinationChunk(
                document_id=document.id,
                destination_name=document.destination_name,
                chunk_text=text,
                chunk_index=idx,
                embedding=vector,
            )
            session.add(chunk)
            db_chunks.append(chunk)
        await session.flush()
        return db_chunks

    async def retrieve_top_k(
        self,
        session: AsyncSession,
        query: str,
        top_k: int = 5,
        destination_filter: str | None = None,
    ) -> list[ChunkResult]:
        """Return top-k chunks closest to the query embedding by cosine distance."""
        query_vector: list[float] = await asyncio.to_thread(self._embed_single, query)

        stmt = (
            select(
                DestinationChunk,
                DestinationDocument.source_title,
                DestinationChunk.embedding.cosine_distance(query_vector).label(
                    "cosine_distance"
                ),
            )
            .join(DestinationDocument, DestinationChunk.document_id == DestinationDocument.id)
            .order_by("cosine_distance")
            .limit(top_k)
        )
        if destination_filter is not None:
            stmt = stmt.where(DestinationChunk.destination_name == destination_filter)

        rows = (await session.execute(stmt)).all()

        return [
            ChunkResult(
                chunk_id=row.DestinationChunk.id,
                document_id=row.DestinationChunk.document_id,
                destination_name=row.DestinationChunk.destination_name,
                source_title=row.source_title,
                chunk_text=row.DestinationChunk.chunk_text,
                chunk_index=row.DestinationChunk.chunk_index,
                cosine_distance=row.cosine_distance,
            )
            for row in rows
        ]


def get_rag_service_for_request(embedder: SentenceTransformer) -> RagService:
    return RagService(embedder=embedder)
