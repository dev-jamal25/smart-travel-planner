from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator


class WebhookTripPlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_query: str
    destination: str | None = None
    trip_plan: str

    @field_validator("user_query")
    @classmethod
    def user_query_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("user_query must not be empty")
        return v.strip()

    @field_validator("trip_plan")
    @classmethod
    def trip_plan_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("trip_plan must not be empty")
        return v.strip()


class WebhookDeliveryResponse(BaseModel):
    delivered: bool
    provider: str = "discord"
    status_code: int | None = None
    error: str | None = None
