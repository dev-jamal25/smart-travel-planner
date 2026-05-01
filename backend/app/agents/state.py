"""LangGraph state for the controlled travel-planning graph."""
from typing import Any, TypedDict
from uuid import UUID

from app.schemas.agent import TripIntent
from app.tools.base import ToolResult


class AgentGraphState(TypedDict):
    """State flowing through the controlled LangGraph travel-planning graph.

    Fields are populated sequentially as each node executes.
    ``session`` is a runtime-only object (AsyncSession) injected at graph
    invocation; it is never serialised or checkpointed.
    """

    # ── Request context ───────────────────────────────────────────────────────
    run_id: UUID
    user_id: UUID
    user_query: str

    # ── Haiku LLM outputs ────────────────────────────────────────────────────
    intent: TripIntent | None
    rewritten_query: str | None

    # ── RAG tool ─────────────────────────────────────────────────────────────
    rag_result: ToolResult | None
    retrieved_destinations: list[str]

    # ── Haiku destination selection ───────────────────────────────────────────
    selected_destination: str | None

    # ── Classifier tool ───────────────────────────────────────────────────────
    classifier_result: ToolResult | None

    # ── Weather tool ─────────────────────────────────────────────────────────
    weather_result: ToolResult | None

    # ── Sonnet synthesis ─────────────────────────────────────────────────────
    final_answer: str | None

    # ── Webhook delivery outcome ─────────────────────────────────────────────
    webhook_delivered: bool | None
    webhook_status_code: int | None

    # ── Non-fatal error accumulation (tool errors synthesis must explain) ─────
    errors: list[str]

    # ── Accumulated cost placeholder (individual costs logged in LLMUsageLog) ─
    total_cost_usd: float | None

    # ── Runtime-only — injected at invocation, never checkpointed ────────────
    session: Any  # sqlalchemy.ext.asyncio.AsyncSession
