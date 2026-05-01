"""Repository for agent run persistence."""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy import select

from app.models.db import AgentRun, AgentTraceEvent, LLMUsageLog, ToolCallLog

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)


async def create_agent_run(
    session: AsyncSession,
    user_id: uuid.UUID,
    user_query: str,
) -> AgentRun:
    """Create a new agent run for a user.

    Args:
        session: Database session.
        user_id: User ID.
        user_query: Original user query.

    Returns:
        Created AgentRun with status "running".
    """
    run = AgentRun(
        user_id=user_id,
        user_query=user_query,
        status="running",
    )
    session.add(run)
    await session.flush()
    return run


async def mark_agent_run_completed(
    session: AsyncSession,
    run_id: uuid.UUID,
    final_answer: str,
    recommended_destination: str | None = None,
    total_cost_usd: float | None = None,
    webhook_delivered: bool | None = None,
    webhook_status_code: int | None = None,
) -> AgentRun:
    """Mark an agent run as completed.

    Args:
        session: Database session.
        run_id: Agent run ID.
        final_answer: Final synthesized answer to the user.
        recommended_destination: Recommended destination (optional).
        total_cost_usd: Total cost of the run (optional).
        webhook_delivered: Whether webhook was delivered (optional).
        webhook_status_code: Webhook HTTP status code (optional).

    Returns:
        Updated AgentRun with status "completed".
    """
    stmt = select(AgentRun).where(AgentRun.id == run_id)
    result = await session.execute(stmt)
    run = result.scalar_one_or_none()
    if run is None:
        raise ValueError(f"AgentRun {run_id} not found")

    run.final_answer = final_answer
    run.recommended_destination = recommended_destination
    run.status = "completed"
    run.total_cost_usd = total_cost_usd
    run.webhook_delivered = webhook_delivered
    run.webhook_status_code = webhook_status_code
    run.completed_at = datetime.now(UTC)

    await session.flush()
    return run


async def mark_agent_run_failed(
    session: AsyncSession,
    run_id: uuid.UUID,
    error_message: str,
) -> AgentRun:
    """Mark an agent run as failed.

    Args:
        session: Database session.
        run_id: Agent run ID.
        error_message: Error message.

    Returns:
        Updated AgentRun with status "failed".
    """
    stmt = select(AgentRun).where(AgentRun.id == run_id)
    result = await session.execute(stmt)
    run = result.scalar_one_or_none()
    if run is None:
        raise ValueError(f"AgentRun {run_id} not found")

    run.status = "failed"
    run.error_message = error_message
    run.completed_at = datetime.now(UTC)

    await session.flush()
    return run


async def log_tool_call(
    session: AsyncSession,
    run_id: uuid.UUID,
    tool_name: str,
    input_json: dict[str, Any] | None = None,
    output_summary: str | None = None,
    status: str = "ok",
    latency_ms: int | None = None,
) -> ToolCallLog:
    """Log a tool invocation.

    Args:
        session: Database session.
        run_id: Agent run ID.
        tool_name: Name of the tool.
        input_json: Tool input (JSON-serializable).
        output_summary: Summary of tool output.
        status: "ok" or "error".
        latency_ms: Latency in milliseconds.

    Returns:
        Created ToolCallLog.
    """
    log = ToolCallLog(
        run_id=run_id,
        tool_name=tool_name,
        input_json=input_json,
        output_summary=output_summary,
        status=status,
        latency_ms=latency_ms,
    )
    session.add(log)
    await session.flush()
    return log


async def log_llm_usage(
    session: AsyncSession,
    run_id: uuid.UUID,
    step_name: str,
    model: str,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cost_usd: float | None = None,
    latency_ms: int | None = None,
) -> LLMUsageLog:
    """Log LLM API usage.

    Args:
        session: Database session.
        run_id: Agent run ID.
        step_name: Name of the agent step (e.g., "intent_extraction", "synthesis").
        model: Model name (e.g., "claude-3-haiku-20240307").
        input_tokens: Input token count.
        output_tokens: Output token count.
        cost_usd: Cost in USD.
        latency_ms: Latency in milliseconds.

    Returns:
        Created LLMUsageLog.
    """
    log = LLMUsageLog(
        run_id=run_id,
        step_name=step_name,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
    )
    session.add(log)
    await session.flush()
    return log


async def log_trace_event(
    session: AsyncSession,
    run_id: uuid.UUID,
    event_type: str,
    event_name: str,
    detail_json: dict[str, Any] | None = None,
    latency_ms: int | None = None,
) -> AgentTraceEvent:
    """Log a trace event for debugging and observability.

    Args:
        session: Database session.
        run_id: Agent run ID.
        event_type: Event type (e.g., "tool_call", "llm_call", "decision").
        event_name: Human-readable event name.
        detail_json: Event details (JSON-serializable).
        latency_ms: Latency in milliseconds.

    Returns:
        Created AgentTraceEvent.
    """
    event = AgentTraceEvent(
        run_id=run_id,
        event_type=event_type,
        event_name=event_name,
        detail_json=detail_json,
        latency_ms=latency_ms,
    )
    session.add(event)
    await session.flush()
    return event


async def list_agent_runs_for_user(
    session: AsyncSession,
    user_id: uuid.UUID,
    limit: int = 20,
) -> list[AgentRun]:
    """List agent runs for a user.

    Args:
        session: Database session.
        user_id: User ID.
        limit: Maximum number of runs to return (default 20).

    Returns:
        List of AgentRun ordered by created_at descending.
    """
    stmt = (
        select(AgentRun)
        .where(AgentRun.user_id == user_id)
        .order_by(AgentRun.created_at.desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_agent_run_for_user(
    session: AsyncSession,
    run_id: uuid.UUID,
    user_id: uuid.UUID,
) -> AgentRun | None:
    """Get an agent run for a specific user.

    Ensures runs are never exposed across users.

    Args:
        session: Database session.
        run_id: Agent run ID.
        user_id: User ID (for access control).

    Returns:
        AgentRun if found and belongs to user, None otherwise.
    """
    stmt = select(AgentRun).where(
        (AgentRun.id == run_id) & (AgentRun.user_id == user_id)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def list_tool_calls_for_run(
    session: AsyncSession,
    run_id: uuid.UUID,
) -> list[ToolCallLog]:
    """Return all tool calls for a run in chronological order.

    Args:
        session: Database session.
        run_id: Agent run ID.

    Returns:
        List of ToolCallLog ordered by created_at ascending.
    """
    stmt = (
        select(ToolCallLog)
        .where(ToolCallLog.run_id == run_id)
        .order_by(ToolCallLog.created_at.asc())
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def list_llm_usage_for_run(
    session: AsyncSession,
    run_id: uuid.UUID,
) -> list[LLMUsageLog]:
    """Return all LLM usage records for a run in chronological order.

    Args:
        session: Database session.
        run_id: Agent run ID.

    Returns:
        List of LLMUsageLog ordered by created_at ascending.
    """
    stmt = (
        select(LLMUsageLog)
        .where(LLMUsageLog.run_id == run_id)
        .order_by(LLMUsageLog.created_at.asc())
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def list_trace_events_for_run(
    session: AsyncSession,
    run_id: uuid.UUID,
) -> list[AgentTraceEvent]:
    """Return all trace events for a run in chronological order.

    Args:
        session: Database session.
        run_id: Agent run ID.

    Returns:
        List of AgentTraceEvent ordered by created_at ascending.
    """
    stmt = (
        select(AgentTraceEvent)
        .where(AgentTraceEvent.run_id == run_id)
        .order_by(AgentTraceEvent.created_at.asc())
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())
