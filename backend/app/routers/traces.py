"""Trace inspection routes — authenticated read-only access to agent run history."""
from __future__ import annotations

import uuid
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories.agent_runs import (
    get_agent_run_for_user,
    list_agent_runs_for_user,
    list_llm_usage_for_run,
    list_tool_calls_for_run,
    list_trace_events_for_run,
)
from app.dependencies import get_current_user, get_session
from app.schemas.auth import CurrentUser
from app.schemas.traces import (
    AgentRunSummary,
    AgentRunTraceDetail,
    AgentTraceEventResponse,
    LLMUsageLogResponse,
    ToolCallLogResponse,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/traces", tags=["traces"])


@router.get("", response_model=list[AgentRunSummary])
async def list_traces(
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
) -> list[AgentRunSummary]:
    """Return recent agent runs for the authenticated user.

    Results are scoped to the caller — other users' runs are never returned.
    Internal fields (error messages, webhook URLs, stack traces) are excluded.
    """
    runs = await list_agent_runs_for_user(session, current_user.user_id, limit=limit)
    logger.info(
        "traces.list",
        user_id=str(current_user.user_id),
        count=len(runs),
    )
    return [AgentRunSummary.model_validate(r) for r in runs]


@router.get("/{run_id}", response_model=AgentRunTraceDetail)
async def get_trace(
    run_id: uuid.UUID,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> AgentRunTraceDetail:
    """Return full trace detail for a single agent run.

    Returns 404 if the run does not exist or belongs to a different user.
    Internal fields (error_message, webhook_status_code) are never included.
    """
    run = await get_agent_run_for_user(session, run_id, current_user.user_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Agent run not found")

    tool_calls = await list_tool_calls_for_run(session, run.id)
    llm_usage = await list_llm_usage_for_run(session, run.id)
    trace_events = await list_trace_events_for_run(session, run.id)

    logger.info(
        "traces.get",
        user_id=str(current_user.user_id),
        run_id=str(run.id),
        tool_call_count=len(tool_calls),
        llm_usage_count=len(llm_usage),
        trace_event_count=len(trace_events),
    )

    return AgentRunTraceDetail(
        id=run.id,
        status=run.status,
        user_query=run.user_query,
        final_answer=run.final_answer,
        recommended_destination=run.recommended_destination,
        total_cost_usd=run.total_cost_usd,
        webhook_delivered=run.webhook_delivered,
        created_at=run.created_at,
        completed_at=run.completed_at,
        tool_calls=[ToolCallLogResponse.model_validate(tc) for tc in tool_calls],
        llm_usage=[LLMUsageLogResponse.model_validate(lu) for lu in llm_usage],
        trace_events=[AgentTraceEventResponse.model_validate(te) for te in trace_events],
    )
