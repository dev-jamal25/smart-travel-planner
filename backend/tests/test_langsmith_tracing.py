"""Tests for LangSmith tracing integration.

Verifies that:
1. LANGCHAIN_TRACING_V2 is off in the test environment (no real network calls).
2. The tracing module exports the three expected decorators.
3. Each decorator is transparent — it does not change the return value of the
   decorated function.
4. Decorated async functions are still awaitable and behave normally.
"""
from __future__ import annotations

import os
import uuid

import pytest

from app.schemas.agent import PlanTripRequest, PlanTripResponse
from app.schemas.agent_tools import ClassifyDestinationInput, LiveWeatherInput
from app.schemas.auth import CurrentUser
from app.tools.base import ToolResult

# ---------------------------------------------------------------------------
# Guard: tracing must be disabled in the test suite
# ---------------------------------------------------------------------------


def test_langsmith_tracing_disabled_in_test_env() -> None:
    """LANGCHAIN_TRACING_V2 must not be 'true' — tests must never hit LangSmith."""
    value = os.environ.get("LANGCHAIN_TRACING_V2", "false").strip().lower()
    assert value != "true", (
        "LANGCHAIN_TRACING_V2=true is set in the test environment. "
        "LangSmith network calls would be made during tests. "
        "Set LANGCHAIN_TRACING_V2=false before running the test suite."
    )


# ---------------------------------------------------------------------------
# Module-level imports
# ---------------------------------------------------------------------------


def test_tracing_module_exports_expected_decorators() -> None:
    """app.tracing must export plan_trip_trace, model_call_trace, tool_call_trace."""
    from app import tracing

    assert callable(tracing.plan_trip_trace), "plan_trip_trace must be callable"
    assert callable(tracing.model_call_trace), "model_call_trace must be callable"
    assert callable(tracing.tool_call_trace), "tool_call_trace must be callable"


# ---------------------------------------------------------------------------
# Transparency — decorators must not change return values
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_model_call_trace_transparent() -> None:
    """@model_call_trace must not alter the return value of the wrapped function."""
    from app.tracing import model_call_trace

    @model_call_trace
    async def _echo(message: str, session: object = None, run_id: object = None) -> str:
        return f"echo:{message}"

    assert await _echo(message="hello") == "echo:hello"
    assert await _echo(message="world", session=None, run_id=None) == "echo:world"


@pytest.mark.asyncio
async def test_tool_call_trace_transparent_with_tool_result() -> None:
    """@tool_call_trace must not alter a ToolResult return value."""
    from app.tracing import tool_call_trace

    expected = ToolResult.ok("test_tool", {"answer": 42})

    @tool_call_trace
    async def _fake_tool(tool_input: ClassifyDestinationInput) -> ToolResult:
        return expected

    result = await _fake_tool(tool_input=ClassifyDestinationInput(destination="Bali"))
    assert result is expected
    assert result.status == "ok"
    assert result.output == {"answer": 42}


@pytest.mark.asyncio
async def test_tool_call_trace_transparent_with_error_result() -> None:
    """@tool_call_trace must preserve ToolResult.fail return values."""
    from app.tracing import tool_call_trace

    expected = ToolResult.fail("test_tool", "service unavailable", retryable=True)

    @tool_call_trace
    async def _failing_tool(tool_input: LiveWeatherInput) -> ToolResult:
        return expected

    result = await _failing_tool(
        tool_input=LiveWeatherInput(destination="Kyoto", forecast_days=3)
    )
    assert result is expected
    assert result.status == "error"
    assert result.retryable is True


@pytest.mark.asyncio
async def test_plan_trip_trace_transparent() -> None:
    """@plan_trip_trace must not alter the PlanTripResponse return value."""
    from app.tracing import plan_trip_trace

    run_id = uuid.uuid4()
    expected = PlanTripResponse(
        run_id=run_id,
        answer="Here is your plan.",
        recommended_destination="Bali",
        webhook_delivered=True,
    )

    @plan_trip_trace
    async def _fake_plan_trip(
        self: object,
        session: object,
        current_user: CurrentUser,
        request: PlanTripRequest,
    ) -> PlanTripResponse:
        return expected

    result = await _fake_plan_trip(
        self=object(),
        session=None,
        current_user=CurrentUser(user_id=uuid.uuid4(), email="t@test.com"),
        request=PlanTripRequest(message="I want beaches"),
    )
    assert result is expected
    assert result.run_id == run_id
    assert result.answer == "Here is your plan."


# ---------------------------------------------------------------------------
# No LangSmith client is created when tracing is disabled
# ---------------------------------------------------------------------------


def test_no_langsmith_client_during_import() -> None:
    """Importing app.tracing must not create a LangSmith Client."""
    from unittest.mock import patch

    with patch("langsmith.client.Client.__init__", return_value=None) as mock_init:
        import importlib

        import app.tracing

        importlib.reload(app.tracing)
        mock_init.assert_not_called()
