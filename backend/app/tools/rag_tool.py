"""RAG retrieval tool for agent."""
from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from app.schemas.agent_tools import DestinationKnowledgeInput
from app.tools.base import ToolResult

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.services.rag_service import RagService

logger = structlog.get_logger(__name__)


async def destination_knowledge_retrieval(
    tool_input: DestinationKnowledgeInput,
    rag_service: RagService,
    session: AsyncSession,
) -> ToolResult:
    """Retrieve destination knowledge from RAG.

    Args:
        tool_input: Validated destination knowledge query.
        rag_service: RAG service dependency.
        session: Database session for retrieval.

    Returns:
        ToolResult with chunks if successful, error otherwise.
    """
    try:
        # Validate input
        tool_input = DestinationKnowledgeInput(**tool_input.model_dump())

        logger.info(
            "rag_tool.retrieve_start",
            query=tool_input.query,
            top_k=tool_input.top_k,
            destination_filter=tool_input.destination_filter,
        )

        # Call RagService directly (not via HTTP endpoint)
        chunks = await rag_service.retrieve_top_k(
            query=tool_input.query,
            top_k=tool_input.top_k,
            destination_filter=tool_input.destination_filter,
            session=session,
        )

        logger.info(
            "rag_tool.retrieve_success",
            num_chunks=len(chunks),
        )

        # Format chunks for agent
        output = {
            "query": tool_input.query,
            "top_k": tool_input.top_k,
            "destination_filter": tool_input.destination_filter,
            "chunks": [
                {
                    "destination": chunk.destination_name,
                    "source_title": chunk.source_title,
                    "chunk_text": chunk.chunk_text[:200],  # First 200 chars
                    "distance": chunk.cosine_distance,
                    "chunk_index": chunk.chunk_index,
                }
                for chunk in chunks
            ],
        }

        return ToolResult.ok("destination_knowledge_retrieval", output)

    except Exception as exc:
        logger.error(
            "rag_tool.retrieve_failure",
            error=str(exc),
            exc_info=True,
        )
        return ToolResult.fail(
            "destination_knowledge_retrieval",
            f"RAG retrieval failed: {exc}",
            retryable=True,  # Network errors are retryable
        )
