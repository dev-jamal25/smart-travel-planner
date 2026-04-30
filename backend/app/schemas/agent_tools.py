"""Input/output schemas for agent tools."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator


class DestinationKnowledgeInput(BaseModel):
    """Input for the destination knowledge retrieval tool."""

    model_config = ConfigDict(extra="forbid")

    query: str
    top_k: int = 5
    destination_filter: str | None = None

    @field_validator("query")
    @classmethod
    def query_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be empty")
        return v.strip()

    @field_validator("top_k")
    @classmethod
    def top_k_in_range(cls, v: int) -> int:
        if not (1 <= v <= 10):
            raise ValueError("top_k must be between 1 and 10")
        return v

    @field_validator("destination_filter")
    @classmethod
    def normalize_destination_filter(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        # Normalize empty/"none"/"null"/"all" to None
        if not v or v.lower() in ("none", "null", "all"):
            return None
        return v


class ClassifyDestinationInput(BaseModel):
    """Input for the destination style classification tool."""

    model_config = ConfigDict(extra="forbid")

    destination: str

    @field_validator("destination")
    @classmethod
    def destination_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("destination must not be empty")
        return v.strip()


class LiveWeatherInput(BaseModel):
    """Input for the live weather tool."""

    model_config = ConfigDict(extra="forbid")

    destination: str
    forecast_days: int = 3

    @field_validator("destination")
    @classmethod
    def destination_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("destination must not be empty")
        return v.strip()

    @field_validator("forecast_days")
    @classmethod
    def forecast_days_in_range(cls, v: int) -> int:
        if not (1 <= v <= 7):
            raise ValueError("forecast_days must be between 1 and 7")
        return v
