"""LangSmith tracing utilities for the travel-planning agent.

Three pre-configured decorators are exported:

    plan_trip_trace    — top-level chain trace for AgentService.plan_trip
    model_call_trace   — chain trace for ModelRouter LLM methods
    tool_call_trace    — tool trace for the three agent tool functions

All decorators use ``process_inputs`` to strip non-serialisable or sensitive
parameters (``self``, ``session``, ``run_id``, service objects) before any data
is sent to LangSmith.

When LangSmith tracing is disabled (``LANGCHAIN_TRACING_V2 != 'true'`` or
``LANGCHAIN_API_KEY`` is absent), every decorated function behaves exactly as if
the decorator were not present — no network calls, no overhead beyond a thin
function wrapper.
"""
from __future__ import annotations

from typing import Any

from langsmith import traceable as _traceable

# ---------------------------------------------------------------------------
# Input filters — called by langsmith BEFORE data is sent to the server
# ---------------------------------------------------------------------------


def _strip_internals(inputs: dict[str, Any]) -> dict[str, Any]:
    """Remove non-serialisable / sensitive parameters.

    Drops: self (contains Anthropic client + API key), session (SQLAlchemy
    AsyncSession), run_id (internal DB FK, not useful in LangSmith).
    """
    return {k: v for k, v in inputs.items() if k not in ("self", "session", "run_id")}


def _plan_trip_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Keep only the user message and user_id for the top-level plan_trip run."""
    request = inputs.get("request")
    current_user = inputs.get("current_user")
    uid = getattr(current_user, "user_id", None)
    return {
        "message": getattr(request, "message", None),
        "user_id": str(uid) if uid is not None else None,
    }


def _tool_input_only(inputs: dict[str, Any]) -> dict[str, Any]:
    """Keep only the validated tool_input Pydantic object, serialised to dict."""
    ti = inputs.get("tool_input")
    dump = getattr(ti, "model_dump", None)
    return {"tool_input": dump() if callable(dump) else ti}


# ---------------------------------------------------------------------------
# Public decorators — apply directly to functions / methods
# ---------------------------------------------------------------------------

#: Top-level LangSmith run for AgentService.plan_trip.
plan_trip_trace = _traceable(
    run_type="chain",
    name="plan_trip",
    process_inputs=_plan_trip_inputs,
)

#: Chain trace for ModelRouter LLM methods (extract_intent, rewrite_query, etc.).
model_call_trace = _traceable(
    run_type="chain",
    process_inputs=_strip_internals,
)

#: Tool trace for the three agent tool functions.
tool_call_trace = _traceable(
    run_type="tool",
    process_inputs=_tool_input_only,
)
