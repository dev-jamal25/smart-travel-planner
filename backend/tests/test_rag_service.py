from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from app.schemas.rag import ChunkResult
from app.services.rag_service import CHUNK_OVERLAP, CHUNK_SIZE, RagService, chunk_text

# ── Pure function tests (no DB, no embedder) ──────────────────────────────────


def test_chunk_text_basic() -> None:
    text = "x" * (CHUNK_SIZE + 50)
    result = chunk_text(text)
    assert len(result) > 1
    for chunk in result:
        assert len(chunk) <= CHUNK_SIZE


def test_chunk_text_short_text() -> None:
    text = "hello world"
    result = chunk_text(text)
    assert result == ["hello world"]


def test_chunk_text_exact_size() -> None:
    text = "a" * CHUNK_SIZE
    result = chunk_text(text)
    assert result == [text]


def test_chunk_text_overlap() -> None:
    text = "a" * CHUNK_SIZE + "b" * CHUNK_SIZE
    result = chunk_text(text)
    assert len(result) >= 2
    tail_of_first = result[0][-CHUNK_OVERLAP:]
    head_of_second = result[1][:CHUNK_OVERLAP]
    assert tail_of_first == head_of_second


def test_chunk_text_custom_params() -> None:
    result = chunk_text("abcdefghij", chunk_size=4, overlap=1)
    assert result[0] == "abcd"
    assert result[1] == "defg"
    assert result[2] == "ghij"


# ── RagService.retrieve_top_k with mocked session ─────────────────────────────


@pytest.fixture
def fake_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.encode.return_value = np.zeros((1, 384), dtype="float32")
    return embedder


async def test_retrieve_top_k_calls_db(fake_embedder: MagicMock) -> None:
    service = RagService(embedder=fake_embedder)
    session = AsyncMock()

    chunk_id = uuid.uuid4()
    doc_id = uuid.uuid4()

    fake_row = MagicMock()
    fake_row.DestinationChunk.id = chunk_id
    fake_row.DestinationChunk.document_id = doc_id
    fake_row.DestinationChunk.destination_name = "Paris"
    fake_row.DestinationChunk.chunk_text = "Paris is a magnificent city."
    fake_row.DestinationChunk.chunk_index = 0
    fake_row.source_title = "Paris Travel Guide"
    fake_row.cosine_distance = 0.05

    fake_result = MagicMock()
    fake_result.all.return_value = [fake_row]
    session.execute.return_value = fake_result

    with patch(
        "app.services.rag_service.asyncio.to_thread",
        new_callable=AsyncMock,
    ) as mock_thread:
        mock_thread.return_value = [0.0] * 384
        results = await service.retrieve_top_k(session, query="best city in France")

    session.execute.assert_called_once()
    assert len(results) == 1
    assert isinstance(results[0], ChunkResult)
    assert results[0].destination_name == "Paris"
    assert results[0].source_title == "Paris Travel Guide"
    assert results[0].cosine_distance == 0.05
    assert results[0].chunk_id == chunk_id
    assert results[0].document_id == doc_id


async def test_retrieve_top_k_empty_db(fake_embedder: MagicMock) -> None:
    service = RagService(embedder=fake_embedder)
    session = AsyncMock()

    fake_result = MagicMock()
    fake_result.all.return_value = []
    session.execute.return_value = fake_result

    with patch(
        "app.services.rag_service.asyncio.to_thread",
        new_callable=AsyncMock,
    ) as mock_thread:
        mock_thread.return_value = [0.0] * 384
        results = await service.retrieve_top_k(session, query="any query")

    assert results == []
