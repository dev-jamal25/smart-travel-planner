"""Tests for GET /traces and GET /traces/{run_id}.

All repository calls are mocked — no real DB required.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_current_user
from app.main import app

_USER_A_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
_RUN_ID = uuid.UUID("aaaaaaaa-0000-0000-0000-000000000001")
_NOW = datetime.now(UTC)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_run(**overrides: Any) -> MagicMock:
    defaults: dict[str, Any] = {
        "id": _RUN_ID,
        "user_id": _USER_A_ID,
        "status": "completed",
        "user_query": "I want beaches in Bali",
        "final_answer": "Here is your Bali plan.",
        "recommended_destination": "Bali",
        "total_cost_usd": 0.01,
        "webhook_delivered": True,
        # Internal fields that must NOT appear in any response
        "error_message": "internal error detail — must be hidden",
        "webhook_status_code": 200,
        "created_at": _NOW,
        "completed_at": _NOW,
    }
    defaults.update(overrides)
    m = MagicMock()
    for k, v in defaults.items():
        setattr(m, k, v)
    return m


def _mock_tool_call(**overrides: Any) -> MagicMock:
    defaults: dict[str, Any] = {
        "id": uuid.uuid4(),
        "run_id": _RUN_ID,
        "tool_name": "destination_knowledge_retrieval",
        "input_json": {"query": "beaches"},
        "output_summary": "Retrieved 3 chunks",
        "status": "ok",
        "latency_ms": 120,
        "created_at": _NOW,
    }
    defaults.update(overrides)
    m = MagicMock()
    for k, v in defaults.items():
        setattr(m, k, v)
    return m


def _mock_llm_usage(**overrides: Any) -> MagicMock:
    defaults: dict[str, Any] = {
        "id": uuid.uuid4(),
        "run_id": _RUN_ID,
        "step_name": "intent_extraction",
        "model": "claude-haiku-4-5-20251001",
        "input_tokens": 100,
        "output_tokens": 50,
        "cost_usd": 0.001,
        "latency_ms": 400,
        "created_at": _NOW,
    }
    defaults.update(overrides)
    m = MagicMock()
    for k, v in defaults.items():
        setattr(m, k, v)
    return m


def _mock_trace_event(**overrides: Any) -> MagicMock:
    defaults: dict[str, Any] = {
        "id": uuid.uuid4(),
        "run_id": _RUN_ID,
        "event_type": "tool_call",
        "event_name": "RAG retrieval started",
        "detail_json": {"query": "beaches in summer"},
        "latency_ms": None,
        "created_at": _NOW,
    }
    defaults.update(overrides)
    m = MagicMock()
    for k, v in defaults.items():
        setattr(m, k, v)
    return m


# ---------------------------------------------------------------------------
# Auth — both endpoints must reject unauthenticated requests
# ---------------------------------------------------------------------------


def test_list_traces_requires_auth(client: TestClient) -> None:
    """GET /traces must return 401 without a valid auth token."""
    app.dependency_overrides.pop(get_current_user, None)
    response = client.get("/traces")
    assert response.status_code == 401


def test_get_trace_requires_auth(client: TestClient) -> None:
    """GET /traces/{run_id} must return 401 without a valid auth token."""
    app.dependency_overrides.pop(get_current_user, None)
    response = client.get(f"/traces/{_RUN_ID}")
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# GET /traces — list endpoint
# ---------------------------------------------------------------------------


def test_list_traces_returns_current_user_runs(client: TestClient) -> None:
    """GET /traces returns runs belonging to the authenticated user."""
    import unittest.mock as um

    mock_run = _mock_run()
    with um.patch(
        "app.routers.traces.list_agent_runs_for_user",
        new=AsyncMock(return_value=[mock_run]),
    ):
        response = client.get("/traces")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["id"] == str(_RUN_ID)
    assert data[0]["status"] == "completed"
    assert data[0]["user_query"] == "I want beaches in Bali"
    assert data[0]["recommended_destination"] == "Bali"
    assert data[0]["webhook_delivered"] is True


def test_list_traces_empty_when_no_runs(client: TestClient) -> None:
    import unittest.mock as um

    with um.patch(
        "app.routers.traces.list_agent_runs_for_user",
        new=AsyncMock(return_value=[]),
    ):
        response = client.get("/traces")

    assert response.status_code == 200
    assert response.json() == []


def test_list_traces_excludes_internal_fields(client: TestClient) -> None:
    """List response must not expose error_message, webhook_status_code, or secrets."""
    import unittest.mock as um

    with um.patch(
        "app.routers.traces.list_agent_runs_for_user",
        new=AsyncMock(return_value=[_mock_run()]),
    ):
        response = client.get("/traces")

    assert response.status_code == 200
    item = response.json()[0]
    for forbidden in ("error_message", "webhook_status_code", "api_key", "webhook_url"):
        assert forbidden not in item, f"list response must not contain '{forbidden}'"


def test_list_traces_default_limit_accepted(client: TestClient) -> None:
    import unittest.mock as um

    with um.patch(
        "app.routers.traces.list_agent_runs_for_user",
        new=AsyncMock(return_value=[]),
    ):
        response = client.get("/traces")
    assert response.status_code == 200


def test_list_traces_custom_limit_accepted(client: TestClient) -> None:
    import unittest.mock as um

    with um.patch(
        "app.routers.traces.list_agent_runs_for_user",
        new=AsyncMock(return_value=[]),
    ):
        response = client.get("/traces?limit=5")
    assert response.status_code == 200


def test_list_traces_rejects_limit_zero(client: TestClient) -> None:
    response = client.get("/traces?limit=0")
    assert response.status_code == 422


def test_list_traces_rejects_limit_over_50(client: TestClient) -> None:
    response = client.get("/traces?limit=51")
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /traces/{run_id} — detail endpoint
# ---------------------------------------------------------------------------


def test_get_trace_returns_run_with_sublists(client: TestClient) -> None:
    """GET /traces/{run_id} must return run detail with tool calls, LLM usage, trace events."""
    import unittest.mock as um

    mock_run = _mock_run()
    mock_tc = _mock_tool_call()
    mock_lu = _mock_llm_usage()
    mock_te = _mock_trace_event()

    with um.patch("app.routers.traces.get_agent_run_for_user", new=AsyncMock(return_value=mock_run)), \
         um.patch("app.routers.traces.list_tool_calls_for_run", new=AsyncMock(return_value=[mock_tc])), \
         um.patch("app.routers.traces.list_llm_usage_for_run", new=AsyncMock(return_value=[mock_lu])), \
         um.patch("app.routers.traces.list_trace_events_for_run", new=AsyncMock(return_value=[mock_te])):
        response = client.get(f"/traces/{_RUN_ID}")

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == str(_RUN_ID)
    assert data["status"] == "completed"
    assert data["final_answer"] == "Here is your Bali plan."
    assert len(data["tool_calls"]) == 1
    assert len(data["llm_usage"]) == 1
    assert len(data["trace_events"]) == 1


def test_get_trace_tool_call_fields(client: TestClient) -> None:
    """Tool call entries must include tool_name, status, latency_ms, input_json."""
    import unittest.mock as um

    mock_tc = _mock_tool_call(
        tool_name="fetch_live_weather",
        input_json={"destination": "Bali", "forecast_days": 3},
        output_summary="Clear skies, 28°C",
        status="ok",
        latency_ms=200,
    )

    with um.patch("app.routers.traces.get_agent_run_for_user", new=AsyncMock(return_value=_mock_run())), \
         um.patch("app.routers.traces.list_tool_calls_for_run", new=AsyncMock(return_value=[mock_tc])), \
         um.patch("app.routers.traces.list_llm_usage_for_run", new=AsyncMock(return_value=[])), \
         um.patch("app.routers.traces.list_trace_events_for_run", new=AsyncMock(return_value=[])):
        response = client.get(f"/traces/{_RUN_ID}")

    tc = response.json()["tool_calls"][0]
    assert tc["tool_name"] == "fetch_live_weather"
    assert tc["status"] == "ok"
    assert tc["latency_ms"] == 200
    assert tc["input_json"] == {"destination": "Bali", "forecast_days": 3}


def test_get_trace_llm_usage_fields(client: TestClient) -> None:
    """LLM usage entries must include step_name, model, token counts, and cost."""
    import unittest.mock as um

    mock_lu = _mock_llm_usage(
        step_name="synthesis",
        model="claude-sonnet-4-6",
        input_tokens=500,
        output_tokens=300,
        cost_usd=0.012,
    )

    with um.patch("app.routers.traces.get_agent_run_for_user", new=AsyncMock(return_value=_mock_run())), \
         um.patch("app.routers.traces.list_tool_calls_for_run", new=AsyncMock(return_value=[])), \
         um.patch("app.routers.traces.list_llm_usage_for_run", new=AsyncMock(return_value=[mock_lu])), \
         um.patch("app.routers.traces.list_trace_events_for_run", new=AsyncMock(return_value=[])):
        response = client.get(f"/traces/{_RUN_ID}")

    lu = response.json()["llm_usage"][0]
    assert lu["step_name"] == "synthesis"
    assert lu["model"] == "claude-sonnet-4-6"
    assert lu["input_tokens"] == 500
    assert lu["output_tokens"] == 300
    assert lu["cost_usd"] == pytest.approx(0.012)


def test_get_trace_trace_event_fields(client: TestClient) -> None:
    """Trace event entries must include event_type, event_name, and detail_json."""
    import unittest.mock as um

    mock_te = _mock_trace_event(
        event_type="decision",
        event_name="destination selected",
        detail_json={"destination": "Bali"},
    )

    with um.patch("app.routers.traces.get_agent_run_for_user", new=AsyncMock(return_value=_mock_run())), \
         um.patch("app.routers.traces.list_tool_calls_for_run", new=AsyncMock(return_value=[])), \
         um.patch("app.routers.traces.list_llm_usage_for_run", new=AsyncMock(return_value=[])), \
         um.patch("app.routers.traces.list_trace_events_for_run", new=AsyncMock(return_value=[mock_te])):
        response = client.get(f"/traces/{_RUN_ID}")

    te = response.json()["trace_events"][0]
    assert te["event_type"] == "decision"
    assert te["event_name"] == "destination selected"
    assert te["detail_json"] == {"destination": "Bali"}


def test_get_trace_wrong_user_returns_404(client: TestClient) -> None:
    """User scoping: wrong user (repo returns None) must yield 404, not run data."""
    import unittest.mock as um

    with um.patch(
        "app.routers.traces.get_agent_run_for_user",
        new=AsyncMock(return_value=None),
    ):
        response = client.get(f"/traces/{_RUN_ID}")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_trace_missing_run_returns_404(client: TestClient) -> None:
    """Non-existent run_id must return 404."""
    import unittest.mock as um

    with um.patch(
        "app.routers.traces.get_agent_run_for_user",
        new=AsyncMock(return_value=None),
    ):
        response = client.get(f"/traces/{uuid.uuid4()}")

    assert response.status_code == 404


def test_get_trace_excludes_internal_fields(client: TestClient) -> None:
    """Detail response must not expose error_message, webhook_status_code, or secrets."""
    import unittest.mock as um

    with um.patch("app.routers.traces.get_agent_run_for_user", new=AsyncMock(return_value=_mock_run())), \
         um.patch("app.routers.traces.list_tool_calls_for_run", new=AsyncMock(return_value=[])), \
         um.patch("app.routers.traces.list_llm_usage_for_run", new=AsyncMock(return_value=[])), \
         um.patch("app.routers.traces.list_trace_events_for_run", new=AsyncMock(return_value=[])):
        response = client.get(f"/traces/{_RUN_ID}")

    assert response.status_code == 200
    data = response.json()
    for forbidden in ("error_message", "webhook_status_code", "api_key", "webhook_url"):
        assert forbidden not in data, f"detail response must not contain '{forbidden}'"
