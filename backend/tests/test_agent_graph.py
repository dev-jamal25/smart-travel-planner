"""Tests for the LangGraph travel-planning graph and AgentService.

All external services (Anthropic, Open-Meteo, Discord, DB, embeddings) are
mocked.  No real network calls are made.
"""
from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.schemas.agent import PlanTripRequest, TripIntent
from app.schemas.auth import CurrentUser
from app.schemas.webhook import WebhookDeliveryResponse

# ── Constants ────────────────────────────────────────────────────────────────

_USER_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
_RUN_ID = uuid.UUID("00000000-0000-0000-0000-000000000002")
_FAKE_USER = CurrentUser(user_id=_USER_ID, email="test@test.com")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_mock_run() -> MagicMock:
    run = MagicMock()
    run.id = _RUN_ID
    return run


def _make_model_router() -> MagicMock:
    router = MagicMock()
    router.extract_trip_intent = AsyncMock(
        return_value=TripIntent(
            candidate_destination="Bali",
            preferred_style="relaxation",
        )
    )
    router.rewrite_rag_query = AsyncMock(return_value="Bali beaches relaxation")
    router.select_candidate_destination = AsyncMock(return_value="bali")
    router.synthesize_final_answer = AsyncMock(
        return_value="Here is your personalised Bali trip plan."
    )
    return router


def _make_rag_service() -> MagicMock:
    svc = MagicMock()
    svc.retrieve_top_k = AsyncMock(
        return_value=[
            MagicMock(
                destination_name="Bali",
                source_title="Bali Travel Guide",
                chunk_text="Bali is a tropical paradise with rice terraces and temples.",
                cosine_distance=0.12,
                chunk_index=0,
            )
        ]
    )
    return svc


def _make_classifier_service() -> MagicMock:
    svc = MagicMock()
    svc.predict = MagicMock(return_value=("Relaxation", 0.87))
    return svc


def _make_weather_service() -> MagicMock:
    from app.schemas.weather import DailyWeather, WeatherForecastResponse

    svc = MagicMock()
    svc.get_forecast = AsyncMock(
        return_value=WeatherForecastResponse(
            destination="Bali",
            latitude=-8.34,
            longitude=115.09,
            current_temperature_c=28.5,
            current_weather_code=0,
            cached=False,
            daily=[
                DailyWeather(
                    date="2026-05-01",
                    temperature_max_c=31.0,
                    temperature_min_c=25.0,
                    precipitation_probability_max=10,
                )
            ],
        )
    )
    return svc


def _make_webhook_service() -> MagicMock:
    svc = MagicMock()
    svc.send_trip_plan = AsyncMock(
        return_value=WebhookDeliveryResponse(delivered=True, status_code=204)
    )
    return svc


def _make_session() -> MagicMock:
    session = MagicMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    return session


# ── Patch helper ──────────────────────────────────────────────────────────────
# Repository functions are imported directly in agent_service.py and graph.py,
# so patches must target those import locations, not the source module.

@contextmanager
def _patch_repos(
    log_tool_call_mock: AsyncMock | None = None,
) -> Iterator[dict[str, AsyncMock]]:
    """Patch all repository helpers at the locations they are imported."""
    create_mock = AsyncMock(return_value=_make_mock_run())
    complete_mock = AsyncMock()
    fail_mock = AsyncMock()
    log_tool_mock = log_tool_call_mock or AsyncMock()
    trace_svc_mock = AsyncMock()
    trace_graph_mock = AsyncMock()

    with (
        patch("app.services.agent_service.create_agent_run", create_mock),
        patch("app.services.agent_service.mark_agent_run_completed", complete_mock),
        patch("app.services.agent_service.mark_agent_run_failed", fail_mock),
        patch("app.services.agent_service.log_trace_event", trace_svc_mock),
        patch("app.agents.graph.log_tool_call", log_tool_mock),
        patch("app.agents.graph.log_trace_event", trace_graph_mock),
    ):
        yield {
            "create": create_mock,
            "complete": complete_mock,
            "fail": fail_mock,
            "log_tool": log_tool_mock,
            "trace_svc": trace_svc_mock,
            "trace_graph": trace_graph_mock,
        }


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def model_router() -> MagicMock:
    return _make_model_router()


@pytest.fixture
def rag_service() -> MagicMock:
    return _make_rag_service()


@pytest.fixture
def classifier_service() -> MagicMock:
    return _make_classifier_service()


@pytest.fixture
def weather_service() -> MagicMock:
    return _make_weather_service()


@pytest.fixture
def webhook_service() -> MagicMock:
    return _make_webhook_service()


@pytest.fixture
def session() -> MagicMock:
    return _make_session()


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_plan_trip_happy_path(
    model_router: MagicMock,
    rag_service: MagicMock,
    classifier_service: MagicMock,
    weather_service: MagicMock,
    webhook_service: MagicMock,
    session: MagicMock,
) -> None:
    """Graph happy path: all three tools fire and final answer is returned."""
    from app.services.agent_service import AgentService

    with _patch_repos():
        svc = AgentService(
            model_router=model_router,
            rag_service=rag_service,
            classifier_service=classifier_service,
            weather_service=weather_service,
            webhook_service=webhook_service,
        )
        response = await svc.plan_trip(
            session=session,
            current_user=_FAKE_USER,
            request=PlanTripRequest(message="I want to relax on a tropical beach."),
        )

    assert response.run_id == _RUN_ID
    assert "Bali" in response.answer
    assert response.webhook_delivered is True


@pytest.mark.asyncio
async def test_haiku_used_before_sonnet(
    model_router: MagicMock,
    rag_service: MagicMock,
    classifier_service: MagicMock,
    weather_service: MagicMock,
    webhook_service: MagicMock,
    session: MagicMock,
) -> None:
    """Haiku methods are called before Sonnet synthesis."""
    call_order: list[str] = []

    async def track_intent(*args: Any, **kwargs: Any) -> TripIntent:
        call_order.append("extract_trip_intent")
        return TripIntent(candidate_destination="Bali")

    async def track_rewrite(*args: Any, **kwargs: Any) -> str:
        call_order.append("rewrite_rag_query")
        return "Bali beaches"

    async def track_select(*args: Any, **kwargs: Any) -> str:
        call_order.append("select_candidate_destination")
        return "bali"

    async def track_synthesize(*args: Any, **kwargs: Any) -> str:
        call_order.append("synthesize_final_answer")
        return "Your Bali trip plan."

    model_router.extract_trip_intent = track_intent
    model_router.rewrite_rag_query = track_rewrite
    model_router.select_candidate_destination = track_select
    model_router.synthesize_final_answer = track_synthesize

    from app.services.agent_service import AgentService

    with _patch_repos():
        svc = AgentService(
            model_router=model_router,
            rag_service=rag_service,
            classifier_service=classifier_service,
            weather_service=weather_service,
            webhook_service=webhook_service,
        )
        await svc.plan_trip(
            session=session,
            current_user=_FAKE_USER,
            request=PlanTripRequest(message="Relax on a beach."),
        )

    haiku_calls = ["extract_trip_intent", "rewrite_rag_query", "select_candidate_destination"]
    for haiku_call in haiku_calls:
        assert haiku_call in call_order, f"{haiku_call} not called"

    assert "synthesize_final_answer" in call_order
    for haiku_call in haiku_calls:
        assert call_order.index(haiku_call) < call_order.index("synthesize_final_answer")


@pytest.mark.asyncio
async def test_rag_tool_failure_does_not_crash(
    model_router: MagicMock,
    rag_service: MagicMock,
    classifier_service: MagicMock,
    weather_service: MagicMock,
    webhook_service: MagicMock,
    session: MagicMock,
) -> None:
    """RAG tool failure continues the graph and synthesis explains missing data."""
    rag_service.retrieve_top_k = AsyncMock(side_effect=Exception("pgvector unavailable"))

    from app.services.agent_service import AgentService

    with _patch_repos():
        svc = AgentService(
            model_router=model_router,
            rag_service=rag_service,
            classifier_service=classifier_service,
            weather_service=weather_service,
            webhook_service=webhook_service,
        )
        response = await svc.plan_trip(
            session=session,
            current_user=_FAKE_USER,
            request=PlanTripRequest(message="I want a beach holiday."),
        )

    assert response.answer


@pytest.mark.asyncio
async def test_classifier_failure_does_not_crash(
    model_router: MagicMock,
    rag_service: MagicMock,
    classifier_service: MagicMock,
    weather_service: MagicMock,
    webhook_service: MagicMock,
    session: MagicMock,
) -> None:
    """Classifier tool failure does not crash the graph."""
    classifier_service.predict = MagicMock(side_effect=RuntimeError("model not loaded"))

    from app.services.agent_service import AgentService

    with _patch_repos():
        svc = AgentService(
            model_router=model_router,
            rag_service=rag_service,
            classifier_service=classifier_service,
            weather_service=weather_service,
            webhook_service=webhook_service,
        )
        response = await svc.plan_trip(
            session=session,
            current_user=_FAKE_USER,
            request=PlanTripRequest(message="Adventure trip."),
        )

    assert response.answer


@pytest.mark.asyncio
async def test_weather_failure_does_not_crash(
    model_router: MagicMock,
    rag_service: MagicMock,
    classifier_service: MagicMock,
    weather_service: MagicMock,
    webhook_service: MagicMock,
    session: MagicMock,
) -> None:
    """Weather tool failure does not crash the graph."""
    from app.services.weather_service import WeatherServiceError

    weather_service.get_forecast = AsyncMock(
        side_effect=WeatherServiceError("Open-Meteo timeout")
    )

    from app.services.agent_service import AgentService

    with _patch_repos():
        svc = AgentService(
            model_router=model_router,
            rag_service=rag_service,
            classifier_service=classifier_service,
            weather_service=weather_service,
            webhook_service=webhook_service,
        )
        response = await svc.plan_trip(
            session=session,
            current_user=_FAKE_USER,
            request=PlanTripRequest(message="Beach trip."),
        )

    assert response.answer


@pytest.mark.asyncio
async def test_webhook_failure_does_not_fail_response(
    model_router: MagicMock,
    rag_service: MagicMock,
    classifier_service: MagicMock,
    weather_service: MagicMock,
    webhook_service: MagicMock,
    session: MagicMock,
) -> None:
    """Webhook failure returns delivered=False but does not raise."""
    webhook_service.send_trip_plan = AsyncMock(
        return_value=WebhookDeliveryResponse(
            delivered=False, status_code=None, error="Network error"
        )
    )

    from app.services.agent_service import AgentService

    with _patch_repos():
        svc = AgentService(
            model_router=model_router,
            rag_service=rag_service,
            classifier_service=classifier_service,
            weather_service=weather_service,
            webhook_service=webhook_service,
        )
        response = await svc.plan_trip(
            session=session,
            current_user=_FAKE_USER,
            request=PlanTripRequest(message="Relaxing holiday."),
        )

    assert response.answer
    assert response.webhook_delivered is False


@pytest.mark.asyncio
async def test_tool_calls_are_logged(
    model_router: MagicMock,
    rag_service: MagicMock,
    classifier_service: MagicMock,
    weather_service: MagicMock,
    webhook_service: MagicMock,
    session: MagicMock,
) -> None:
    """After each tool call, log_tool_call is invoked with the tool name."""
    from app.services.agent_service import AgentService

    log_tool_mock = AsyncMock()
    with _patch_repos(log_tool_call_mock=log_tool_mock) as mocks:
        svc = AgentService(
            model_router=model_router,
            rag_service=rag_service,
            classifier_service=classifier_service,
            weather_service=weather_service,
            webhook_service=webhook_service,
        )
        await svc.plan_trip(
            session=session,
            current_user=_FAKE_USER,
            request=PlanTripRequest(message="I want to visit Bali."),
        )
        log_tool_mock = mocks["log_tool"]

    logged_tool_names = {call.kwargs["tool_name"] for call in log_tool_mock.call_args_list}
    assert "destination_knowledge_retrieval" in logged_tool_names
    assert "classify_destination_style" in logged_tool_names
    assert "fetch_live_weather" in logged_tool_names


@pytest.mark.asyncio
async def test_agent_run_marked_completed_on_success(
    model_router: MagicMock,
    rag_service: MagicMock,
    classifier_service: MagicMock,
    weather_service: MagicMock,
    webhook_service: MagicMock,
    session: MagicMock,
) -> None:
    """AgentRun is marked completed after successful graph execution."""
    from app.services.agent_service import AgentService

    with _patch_repos() as mocks:
        svc = AgentService(
            model_router=model_router,
            rag_service=rag_service,
            classifier_service=classifier_service,
            weather_service=weather_service,
            webhook_service=webhook_service,
        )
        await svc.plan_trip(
            session=session,
            current_user=_FAKE_USER,
            request=PlanTripRequest(message="Trip to Bali."),
        )

    mocks["complete"].assert_called_once()
    mocks["fail"].assert_not_called()
    session.commit.assert_called()


@pytest.mark.asyncio
async def test_agent_run_marked_failed_on_unhandled_error(
    model_router: MagicMock,
    rag_service: MagicMock,
    classifier_service: MagicMock,
    weather_service: MagicMock,
    webhook_service: MagicMock,
    session: MagicMock,
) -> None:
    """AgentRun is marked failed when the graph raises an unhandled exception."""
    model_router.extract_trip_intent = AsyncMock(
        side_effect=RuntimeError("Anthropic API key invalid")
    )

    from app.services.agent_service import AgentService

    with _patch_repos() as mocks:
        svc = AgentService(
            model_router=model_router,
            rag_service=rag_service,
            classifier_service=classifier_service,
            weather_service=weather_service,
            webhook_service=webhook_service,
        )
        with pytest.raises(RuntimeError, match="Anthropic API key invalid"):
            await svc.plan_trip(
                session=session,
                current_user=_FAKE_USER,
                request=PlanTripRequest(message="Trip."),
            )

    mocks["fail"].assert_called_once()
    mocks["complete"].assert_not_called()


@pytest.mark.asyncio
async def test_final_response_excludes_internal_fields(
    model_router: MagicMock,
    rag_service: MagicMock,
    classifier_service: MagicMock,
    weather_service: MagicMock,
    webhook_service: MagicMock,
    session: MagicMock,
) -> None:
    """PlanTripResponse does not include tools_used, token counts, or cost."""
    from app.services.agent_service import AgentService

    with _patch_repos():
        svc = AgentService(
            model_router=model_router,
            rag_service=rag_service,
            classifier_service=classifier_service,
            weather_service=weather_service,
            webhook_service=webhook_service,
        )
        response = await svc.plan_trip(
            session=session,
            current_user=_FAKE_USER,
            request=PlanTripRequest(message="Trip to Bali."),
        )

    response_dict = response.model_dump()
    assert "tools_used" not in response_dict
    assert "input_tokens" not in response_dict
    assert "output_tokens" not in response_dict
    assert "cost_usd" not in response_dict
    assert "total_cost_usd" not in response_dict
    allowed_keys = {"run_id", "answer", "recommended_destination", "webhook_delivered"}
    assert set(response_dict.keys()) == allowed_keys
