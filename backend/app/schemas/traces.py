"""Response schemas for agent trace inspection endpoints."""
from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class AgentRunSummary(BaseModel):
    """Summary view of one agent run — used in the list endpoint."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    status: str
    user_query: str
    recommended_destination: str | None = None
    total_cost_usd: float | None = None
    webhook_delivered: bool | None = None
    created_at: datetime
    completed_at: datetime | None = None


class ToolCallLogResponse(BaseModel):
    """Tool invocation record within an agent run."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    tool_name: str
    input_json: dict[str, Any] | None = None
    output_summary: str | None = None
    status: str
    latency_ms: int | None = None
    created_at: datetime


class LLMUsageLogResponse(BaseModel):
    """LLM API usage record within an agent run."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    step_name: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    latency_ms: int | None = None
    created_at: datetime


class AgentTraceEventResponse(BaseModel):
    """Trace event record within an agent run."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    event_type: str
    event_name: str
    detail_json: dict[str, Any] | None = None
    latency_ms: int | None = None
    created_at: datetime


class AgentRunTraceDetail(BaseModel):
    """Full trace detail for a single agent run — used in the detail endpoint."""

    id: UUID
    status: str
    user_query: str
    final_answer: str | None = None
    recommended_destination: str | None = None
    total_cost_usd: float | None = None
    webhook_delivered: bool | None = None
    created_at: datetime
    completed_at: datetime | None = None
    tool_calls: list[ToolCallLogResponse]
    llm_usage: list[LLMUsageLogResponse]
    trace_events: list[AgentTraceEventResponse]
