"""Tests for POST /chat/plan-trip.

No calls to real Anthropic, DB, RAG, weather, classifier, or Discord.
AgentService is replaced with a mock via dependency override.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_agent_service, get_current_user
from app.main import app
from app.schemas.agent import PlanTripResponse
from app.services.agent_service import AgentService

_TEST_RUN_ID = UUID("00000000-0000-0000-0000-000000000042")
_VALID_REQUEST = {"message": "I want 10 days in Bali, relaxation, around $2000"}


def _make_mock_service(
    *,
    answer: str = "Here is your Bali trip plan.",
    destination: str | None = "Bali",
    webhook_delivered: bool | None = True,
) -> AgentService:
    mock_svc = MagicMock(spec=AgentService)
    mock_svc.plan_trip = AsyncMock(
        return_value=PlanTripResponse(
            run_id=_TEST_RUN_ID,
            answer=answer,
            recommended_destination=destination,
            webhook_delivered=webhook_delivered,
        )
    )
    return mock_svc  # type: ignore[return-value]


@pytest.fixture()
def chat_client(client: TestClient):  # type: ignore[no-untyped-def]
    mock_svc = _make_mock_service()
    app.dependency_overrides[get_agent_service] = lambda: mock_svc
    yield client, mock_svc
    app.dependency_overrides.pop(get_agent_service, None)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def test_endpoint_requires_auth(client: TestClient) -> None:
    """Missing auth header must return 401."""
    app.dependency_overrides.pop(get_current_user, None)
    response = client.post("/chat/plan-trip", json=_VALID_REQUEST)
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_endpoint_returns_200_with_plan_trip_response(chat_client) -> None:  # type: ignore[no-untyped-def]
    client, _ = chat_client
    response = client.post("/chat/plan-trip", json=_VALID_REQUEST)
    assert response.status_code == 200
    data = response.json()
    assert data["run_id"] == str(_TEST_RUN_ID)
    assert data["answer"] == "Here is your Bali trip plan."
    assert data["recommended_destination"] == "Bali"
    assert data["webhook_delivered"] is True


def test_endpoint_calls_agent_service_with_correct_args(chat_client) -> None:  # type: ignore[no-untyped-def]
    client, mock_svc = chat_client
    client.post("/chat/plan-trip", json=_VALID_REQUEST)
    mock_svc.plan_trip.assert_awaited_once()
    call_kwargs = mock_svc.plan_trip.call_args.kwargs
    assert call_kwargs["request"].message == _VALID_REQUEST["message"]


def test_endpoint_returns_null_destination_when_none(chat_client) -> None:  # type: ignore[no-untyped-def]
    client, mock_svc = chat_client
    mock_svc.plan_trip = AsyncMock(
        return_value=PlanTripResponse(
            run_id=_TEST_RUN_ID,
            answer="Could not determine destination.",
            recommended_destination=None,
            webhook_delivered=False,
        )
    )
    response = client.post("/chat/plan-trip", json=_VALID_REQUEST)
    assert response.status_code == 200
    data = response.json()
    assert data["recommended_destination"] is None
    assert data["webhook_delivered"] is False


# ---------------------------------------------------------------------------
# Response shape — internal fields must be absent
# ---------------------------------------------------------------------------


def test_response_excludes_internal_cost_and_token_fields(chat_client) -> None:  # type: ignore[no-untyped-def]
    """Internal accounting fields must never appear in the user-facing response."""
    client, _ = chat_client
    response = client.post("/chat/plan-trip", json=_VALID_REQUEST)
    assert response.status_code == 200
    data = response.json()
    for forbidden in (
        "cost_usd",
        "total_cost_usd",
        "input_tokens",
        "output_tokens",
        "tools_used",
        "errors",
        "trace",
        "tool_calls",
        "llm_usage",
    ):
        assert forbidden not in data, f"response must not contain '{forbidden}'"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_endpoint_returns_500_on_agent_failure(client: TestClient) -> None:
    failing_svc = MagicMock(spec=AgentService)
    failing_svc.plan_trip = AsyncMock(side_effect=RuntimeError("Graph exploded"))
    app.dependency_overrides[get_agent_service] = lambda: failing_svc
    try:
        response = client.post("/chat/plan-trip", json=_VALID_REQUEST)
        assert response.status_code == 500
        assert response.json()["detail"] == "Travel planning failed. Please try again."
    finally:
        app.dependency_overrides.pop(get_agent_service, None)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_endpoint_rejects_empty_message(chat_client) -> None:  # type: ignore[no-untyped-def]
    client, _ = chat_client
    response = client.post("/chat/plan-trip", json={"message": ""})
    assert response.status_code == 422


def test_endpoint_rejects_whitespace_message(chat_client) -> None:  # type: ignore[no-untyped-def]
    client, _ = chat_client
    response = client.post("/chat/plan-trip", json={"message": "   "})
    assert response.status_code == 422


def test_endpoint_rejects_missing_message_field(chat_client) -> None:  # type: ignore[no-untyped-def]
    client, _ = chat_client
    response = client.post("/chat/plan-trip", json={})
    assert response.status_code == 422
