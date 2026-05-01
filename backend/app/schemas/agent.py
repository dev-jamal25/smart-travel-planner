"""Agent request/response and supporting types."""
from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator


class TripIntent(BaseModel):
    """Extracted structured intent from user message."""

    model_config = ConfigDict(extra="forbid")

    budget_usd: float | None = None
    duration_days: int | None = None
    travel_month: str | None = None
    preferred_style: str | None = None
    preferred_activities: list[str] = []
    climate_preference: str | None = None
    candidate_destination: str | None = None

    @field_validator("budget_usd")
    @classmethod
    def budget_non_negative(cls, v: float | None) -> float | None:
        if v is not None and v < 0:
            raise ValueError("budget_usd must be >= 0")
        return v

    @field_validator("duration_days")
    @classmethod
    def duration_positive(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError("duration_days must be >= 1")
        return v


class AgentToolError(BaseModel):
    """Structured error from a tool execution."""

    tool_name: str
    error: str
    retryable: bool = False


class PlanTripRequest(BaseModel):
    """Request to plan a trip."""

    model_config = ConfigDict(extra="forbid")

    message: str

    @field_validator("message")
    @classmethod
    def message_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be empty")
        return v.strip()


class PlanTripResponse(BaseModel):
    """Response from trip planning agent."""

    run_id: UUID
    answer: str
    recommended_destination: str | None = None
    webhook_delivered: bool | None = None
