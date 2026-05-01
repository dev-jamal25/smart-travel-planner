"""Tests for model router (Haiku and Sonnet LLM calls)."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from anthropic import AsyncAnthropic
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.agents.model_router import ModelRouter
from app.config import Settings
from app.models.db import Base, LLMUsageLog
from app.schemas.agent import TripIntent


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    return Settings(
        database_url="postgresql+asyncpg://test",
        supabase_jwt_jwks_url="https://test.supabase.co/auth/v1/.well-known/jwks.json",
        supabase_jwt_issuer="https://test.supabase.co/auth/v1",
        supabase_anon_key="test-anon",
        supabase_service_role_key="test-service",
        anthropic_api_key="test-api-key",
        webhook_url="https://discord.com/api/webhooks/test",
        haiku_model="claude-haiku-4-5-20251001",
        sonnet_model="claude-sonnet-4-6",
        anthropic_timeout_seconds=30,
    )


@pytest.fixture
def mock_client():
    """Create mock AsyncAnthropic client."""
    return AsyncMock(spec=AsyncAnthropic)


@pytest.fixture
def model_router(mock_settings, mock_client):
    """Create ModelRouter instance with mocked dependencies."""
    return ModelRouter(settings=mock_settings, client=mock_client)


@pytest.fixture
async def db_session():
    """Create an in-memory SQLite database session for testing."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        yield session
    await engine.dispose()


class TestModelRouterInit:
    """Test ModelRouter initialization."""

    def test_init_with_settings_and_client(self, mock_settings, mock_client):
        router = ModelRouter(settings=mock_settings, client=mock_client)
        assert router.settings == mock_settings
        assert router.client == mock_client

    def test_init_defaults_to_get_settings(self):
        """ModelRouter can be initialized without explicit settings (uses get_settings)."""
        with patch("app.agents.model_router.get_settings") as mock_get:
            mock_settings = MagicMock()
            mock_get.return_value = mock_settings
            router = ModelRouter(client=AsyncMock())
            # Settings are lazily evaluated; we just verify initialization succeeds
            assert router.settings is not None


class TestEstimateCost:
    """Test cost estimation from token counts."""

    def test_estimate_cost_haiku(self, model_router):
        cost = model_router._estimate_cost(
            model="claude-haiku-4-5-20251001",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        )
        # Haiku: 0.80 per 1M input + 4.00 per 1M output = 4.80
        assert cost == pytest.approx(4.80, rel=0.01)

    def test_estimate_cost_sonnet(self, model_router):
        cost = model_router._estimate_cost(
            model="claude-sonnet-4-6",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        )
        # Sonnet: 3.00 per 1M input + 15.00 per 1M output = 18.00
        assert cost == pytest.approx(18.00, rel=0.01)

    def test_estimate_cost_unknown_model(self, model_router):
        cost = model_router._estimate_cost(
            model="unknown-model",
            input_tokens=1000,
            output_tokens=500,
        )
        assert cost is None


class TestExtractTripIntent:
    """Test extract_trip_intent method."""

    @pytest.mark.asyncio
    async def test_extract_trip_intent_success(self, model_router):
        """Test successful intent extraction."""
        intent_json = {
            "budget_usd": 2000,
            "duration_days": 7,
            "travel_month": "June",
            "preferred_style": "adventure",
            "preferred_activities": ["hiking", "camping"],
            "climate_preference": "cool",
            "candidate_destination": "Banff",
        }

        # Mock the Anthropic response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps(intent_json))]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 150

        model_router.client.messages.create = AsyncMock(return_value=mock_response)

        message = "I want a mountain hiking trip"
        intent = await model_router.extract_trip_intent(message)

        assert isinstance(intent, TripIntent)
        assert intent.budget_usd == 2000
        assert intent.duration_days == 7
        assert intent.preferred_style == "adventure"
        assert intent.candidate_destination == "Banff"

    @pytest.mark.asyncio
    async def test_extract_trip_intent_with_markdown(self, model_router):
        """Test intent extraction when response is in markdown code block."""
        intent_json = {"budget_usd": 1500, "duration_days": 5}

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=f"```json\n{json.dumps(intent_json)}\n```")]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 150

        model_router.client.messages.create = AsyncMock(return_value=mock_response)

        intent = await model_router.extract_trip_intent("test message")
        assert intent.budget_usd == 1500
        assert intent.duration_days == 5

    @pytest.mark.asyncio
    async def test_extract_trip_intent_invalid_json(self, model_router):
        """Test intent extraction with invalid JSON returns empty intent."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="not valid json")]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 150

        model_router.client.messages.create = AsyncMock(return_value=mock_response)

        intent = await model_router.extract_trip_intent("test message")
        # Should return empty TripIntent on parse failure
        assert isinstance(intent, TripIntent)
        assert intent.budget_usd is None

    @pytest.mark.asyncio
    async def test_extract_trip_intent_logs_usage(self, model_router, db_session):
        """Test that LLM usage is logged to database when session/run_id provided."""
        run_id = uuid4()
        intent_json = {"budget_usd": 1000}

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps(intent_json))]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 150

        model_router.client.messages.create = AsyncMock(return_value=mock_response)

        await model_router.extract_trip_intent(
            message="test",
            session=db_session,
            run_id=run_id,
        )

        # Verify LLMUsageLog was created
        from sqlalchemy import select

        stmt = select(LLMUsageLog).where(LLMUsageLog.run_id == run_id)
        result = await db_session.execute(stmt)
        logs = result.scalars().all()
        assert len(logs) == 1
        assert logs[0].step_name == "intent_extraction"
        assert logs[0].input_tokens == 100
        assert logs[0].output_tokens == 150


class TestRewriteRagQuery:
    """Test rewrite_rag_query method."""

    @pytest.mark.asyncio
    async def test_rewrite_rag_query(self, model_router):
        """Test RAG query rewriting."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text="- mountain hiking adventure\n- alpine camp destination\n- cool weather trek"
            )
        ]
        mock_response.usage.input_tokens = 150
        mock_response.usage.output_tokens = 100

        model_router.client.messages.create = AsyncMock(return_value=mock_response)

        intent = TripIntent(budget_usd=2000, duration_days=7, preferred_style="adventure")
        rewritten = await model_router.rewrite_rag_query("I want mountain hiking", intent)

        assert isinstance(rewritten, str)
        assert "mountain" in rewritten.lower() or "hiking" in rewritten.lower()


class TestSelectCandidateDestination:
    """Test select_candidate_destination method."""

    @pytest.mark.asyncio
    async def test_select_candidate_destination(self, model_router):
        """Test destination selection."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Banff")]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50

        model_router.client.messages.create = AsyncMock(return_value=mock_response)

        selection = await model_router.select_candidate_destination(
            "I want mountains",
            ["Banff", "Interlaken", "Kyoto"],
        )

        assert selection == "Banff"

    @pytest.mark.asyncio
    async def test_select_candidate_destination_no_match(self, model_router):
        """Test destination selection returns None for no match."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="null")]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50

        model_router.client.messages.create = AsyncMock(return_value=mock_response)

        selection = await model_router.select_candidate_destination(
            "test",
            ["Banff"],
        )

        assert selection is None

    @pytest.mark.asyncio
    async def test_haiku_returns_bold_formatted_name(self, model_router):
        """Haiku returns '**best match:** Santorini' → normalized to 'Santorini'."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="**best match:** Santorini")]
        mock_response.usage.input_tokens = 50
        mock_response.usage.output_tokens = 10
        model_router.client.messages.create = AsyncMock(return_value=mock_response)

        result = await model_router.select_candidate_destination(
            message="Romantic island trip with sunsets",
            retrieved_destinations=["Santorini", "Bali"],
        )
        assert result == "Santorini"

    @pytest.mark.asyncio
    async def test_haiku_returns_reasoning_paragraph(self, model_router):
        """Haiku returns a reasoning paragraph → first candidate name extracted."""
        paragraph = (
            "Based on the user's request for a romantic island getaway with "
            "white-washed buildings and sunsets, Santorini is the best match "
            "among the retrieved destinations."
        )
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=paragraph)]
        mock_response.usage.input_tokens = 50
        mock_response.usage.output_tokens = 60
        model_router.client.messages.create = AsyncMock(return_value=mock_response)

        result = await model_router.select_candidate_destination(
            message="Romantic island sunset trip",
            retrieved_destinations=["Santorini", "Bali"],
        )
        assert result == "Santorini"

    @pytest.mark.asyncio
    async def test_haiku_returns_unsupported_destination(self, model_router):
        """Haiku returns a name not in the supported list → None."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Paris")]
        mock_response.usage.input_tokens = 50
        mock_response.usage.output_tokens = 5
        model_router.client.messages.create = AsyncMock(return_value=mock_response)

        result = await model_router.select_candidate_destination(
            message="I want to see the Eiffel Tower",
            retrieved_destinations=["Santorini", "Bali"],
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_normalized_name_fits_db_column(self, model_router):
        """Normalized destination name is short enough for String(256) column."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Bali")]
        mock_response.usage.input_tokens = 50
        mock_response.usage.output_tokens = 5
        model_router.client.messages.create = AsyncMock(return_value=mock_response)

        result = await model_router.select_candidate_destination(
            message="Beach relaxation",
            retrieved_destinations=["Bali"],
        )
        assert result is not None
        assert len(result) <= 256


class TestRepairToolArguments:
    """Test repair_tool_arguments method."""

    @pytest.mark.asyncio
    async def test_repair_tool_arguments_success(self, model_router):
        """Test successful tool argument repair."""
        repaired = {"top_k": 5, "destination_filter": "Bali"}

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps(repaired))]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 100

        model_router.client.messages.create = AsyncMock(return_value=mock_response)

        result = await model_router.repair_tool_arguments(
            tool_name="rag_retrieval",
            invalid_payload={"top_k": 20},
            validation_error="top_k must be <= 10",
        )

        assert result == repaired

    @pytest.mark.asyncio
    async def test_repair_tool_arguments_fallback_to_original(self, model_router):
        """Test repair returns original on JSON parse failure."""
        original = {"top_k": 20}

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="invalid json")]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 100

        model_router.client.messages.create = AsyncMock(return_value=mock_response)

        result = await model_router.repair_tool_arguments(
            tool_name="rag_retrieval",
            invalid_payload=original,
            validation_error="error",
        )

        assert result == original


class TestSynthesizeFinalAnswer:
    """Test synthesize_final_answer method (Sonnet)."""

    @pytest.mark.asyncio
    async def test_synthesize_final_answer(self, model_router, mock_settings):
        """Test final answer synthesis uses Sonnet model."""
        expected_answer = "I recommend Banff for your summer hiking trip..."

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=expected_answer)]
        mock_response.usage.input_tokens = 500
        mock_response.usage.output_tokens = 300

        model_router.client.messages.create = AsyncMock(return_value=mock_response)

        answer = await model_router.synthesize_final_answer(
            user_message="I want a hiking trip",
            trip_intent=TripIntent(budget_usd=2000, duration_days=7),
            rag_content="Banff is in the Canadian Rockies...",
            classifier_result="Adventure: 92%",
            weather_summary="Clear skies, 15°C",
        )

        assert answer == expected_answer

        # Verify Sonnet model was used
        call_kwargs = model_router.client.messages.create.call_args[1]
        assert call_kwargs["model"] == mock_settings.sonnet_model

    @pytest.mark.asyncio
    async def test_synthesize_final_answer_logs_usage(self, model_router, db_session):
        """Test synthesis logs usage to database."""
        run_id = uuid4()

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Test answer")]
        mock_response.usage.input_tokens = 500
        mock_response.usage.output_tokens = 300

        model_router.client.messages.create = AsyncMock(return_value=mock_response)

        await model_router.synthesize_final_answer(
            user_message="test",
            trip_intent=TripIntent(),
            rag_content="test",
            classifier_result="test",
            weather_summary="test",
            session=db_session,
            run_id=run_id,
        )

        # Verify LLMUsageLog was created
        from sqlalchemy import select

        stmt = select(LLMUsageLog).where(
            (LLMUsageLog.run_id == run_id) & (LLMUsageLog.step_name == "final_synthesis")
        )
        result = await db_session.execute(stmt)
        logs = result.scalars().all()
        assert len(logs) == 1
        assert logs[0].step_name == "final_synthesis"


class TestHaikuVsSonnetModels:
    """Test that Haiku and Sonnet models are correctly routed."""

    @pytest.mark.asyncio
    async def test_haiku_used_for_intent_extraction(self, model_router, mock_settings):
        """Verify Haiku model is used for mechanical tasks."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="{}")]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 100

        model_router.client.messages.create = AsyncMock(return_value=mock_response)

        await model_router.extract_trip_intent("test")

        call_kwargs = model_router.client.messages.create.call_args[1]
        assert call_kwargs["model"] == mock_settings.haiku_model

    @pytest.mark.asyncio
    async def test_sonnet_used_for_synthesis(self, model_router, mock_settings):
        """Verify Sonnet model is used for synthesis."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="answer")]
        mock_response.usage.input_tokens = 500
        mock_response.usage.output_tokens = 300

        model_router.client.messages.create = AsyncMock(return_value=mock_response)

        await model_router.synthesize_final_answer(
            user_message="test",
            trip_intent=TripIntent(),
            rag_content="test",
            classifier_result="test",
            weather_summary="test",
        )

        call_kwargs = model_router.client.messages.create.call_args[1]
        assert call_kwargs["model"] == mock_settings.sonnet_model
