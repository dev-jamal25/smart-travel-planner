"""RAG retrieval endpoints for testing and evaluation."""

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_rag_service, get_session
from app.evaluation.rag_eval_cases import load_rag_eval_cases
from app.schemas.rag import (
    RagChunkResult,
    RagEvalCaseResponse,
    RagEvalResponse,
    RagSearchRequest,
    RagSearchResponse,
)
from app.services.rag_service import RagService

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/rag", tags=["rag"])


async def _retrieve_and_format(
    rag_service: RagService,
    session: AsyncSession,
    query: str,
    top_k: int,
    destination_filter: str | None = None,
) -> RagSearchResponse:
    """Internal helper to retrieve chunks and format as RagSearchResponse."""
    chunk_results = await rag_service.retrieve_top_k(
        session=session,
        query=query,
        top_k=top_k,
        destination_filter=destination_filter,
    )

    # Convert ChunkResult to RagChunkResult
    formatted_results = [
        RagChunkResult(
            destination_name=chunk.destination_name,
            source_title=chunk.source_title,
            source_url="https://en.wikivoyage.org/wiki/" + chunk.destination_name,
            distance=chunk.cosine_distance,
            chunk_preview=chunk.chunk_text[:200],
            chunk_index=chunk.chunk_index,
        )
        for chunk in chunk_results
    ]

    return RagSearchResponse(
        query=query,
        top_k=top_k,
        results=formatted_results,
    )


@router.post("/search")
async def rag_search(
    request: RagSearchRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
    rag_service: Annotated[RagService, Depends(get_rag_service)],
) -> RagSearchResponse:
    """Ad-hoc RAG search for manual retrieval testing.

    Returns the top-k chunks most similar to the query embedding.
    Supports optional destination filtering.
    """
    logger.info(
        "rag.search.start",
        query=request.query[:100],
        top_k=request.top_k,
        destination_filter=request.destination_filter,
    )

    response = await _retrieve_and_format(
        rag_service=rag_service,
        session=session,
        query=request.query,
        top_k=request.top_k,
        destination_filter=request.destination_filter,
    )

    logger.info(
        "rag.search.complete",
        query=request.query[:100],
        result_count=len(response.results),
    )

    return response


@router.get("/eval")
async def rag_eval(
    session: Annotated[AsyncSession, Depends(get_session)],
    rag_service: Annotated[RagService, Depends(get_rag_service)],
) -> RagEvalResponse:
    """Evaluate RAG retrieval quality against 7 fixed test cases.

    For each case, retrieves top-5 chunks and checks if any expected
    destination appears in top-3 and top-5 results.
    """
    eval_cases = load_rag_eval_cases()

    logger.info("rag.eval.start", case_count=len(eval_cases))

    cases: list[RagEvalCaseResponse] = []
    top_3_passes = 0
    top_5_passes = 0

    for eval_case in eval_cases:
        # Retrieve top-5 for this case
        search_response = await _retrieve_and_format(
            rag_service=rag_service,
            session=session,
            query=eval_case.query,
            top_k=5,
            destination_filter=None,
        )

        # Check if any expected destination appears in top-3 and top-5
        result_destinations = [r.destination_name for r in search_response.results]
        expected_dests = eval_case.expected_destinations

        top_3_results = result_destinations[:3]
        top_5_results = result_destinations[:5]

        top_3_pass = any(dest in top_3_results for dest in expected_dests)
        top_5_pass = any(dest in top_5_results for dest in expected_dests)

        if top_3_pass:
            top_3_passes += 1
        if top_5_pass:
            top_5_passes += 1

        case_response = RagEvalCaseResponse(
            name=eval_case.name,
            query=eval_case.query,
            expected_destinations=expected_dests,
            top_3_pass=top_3_pass,
            top_5_pass=top_5_pass,
            search=search_response,
        )
        cases.append(case_response)

        logger.info(
            "rag.eval.case_complete",
            name=eval_case.name,
            top_3_pass=top_3_pass,
            top_5_pass=top_5_pass,
            result_destinations=result_destinations,
        )

    response = RagEvalResponse(
        total_cases=len(eval_cases),
        top_3_passes=top_3_passes,
        top_5_passes=top_5_passes,
        cases=cases,
    )

    logger.info(
        "rag.eval.complete",
        total_cases=len(eval_cases),
        top_3_passes=top_3_passes,
        top_5_passes=top_5_passes,
    )

    return response
