"""Tests for agent persistence repositories."""
from __future__ import annotations

import uuid

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.db.repositories.agent_runs import (
    create_agent_run,
    get_agent_run_for_user,
    list_agent_runs_for_user,
    log_llm_usage,
    log_tool_call,
    log_trace_event,
    mark_agent_run_completed,
    mark_agent_run_failed,
)
from app.models.db import Base


@pytest.fixture
async def db_session():
    """Create an in-memory SQLite database session."""
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


class TestCreateAgentRun:
    """Test agent run creation."""

    @pytest.mark.asyncio
    async def test_create_agent_run(self, db_session) -> None:
        user_id = uuid.uuid4()
        query = "I want a beach vacation in summer"

        run = await create_agent_run(db_session, user_id, query)

        assert run.id is not None
        assert run.user_id == user_id
        assert run.user_query == query
        assert run.status == "running"
        assert run.final_answer is None
        assert run.created_at is not None
        assert run.completed_at is None


class TestMarkAgentRunCompleted:
    """Test marking agent run as completed."""

    @pytest.mark.asyncio
    async def test_mark_completed(self, db_session) -> None:
        user_id = uuid.uuid4()
        run = await create_agent_run(db_session, user_id, "test query")

        completed = await mark_agent_run_completed(
            db_session,
            run.id,
            final_answer="Here is your plan...",
            recommended_destination="Bali",
            total_cost_usd=0.50,
            webhook_delivered=True,
            webhook_status_code=200,
        )

        assert completed.status == "completed"
        assert completed.final_answer == "Here is your plan..."
        assert completed.recommended_destination == "Bali"
        assert completed.total_cost_usd == 0.50
        assert completed.webhook_delivered is True
        assert completed.webhook_status_code == 200
        assert completed.completed_at is not None


class TestMarkAgentRunFailed:
    """Test marking agent run as failed."""

    @pytest.mark.asyncio
    async def test_mark_failed(self, db_session) -> None:
        user_id = uuid.uuid4()
        run = await create_agent_run(db_session, user_id, "test query")

        failed = await mark_agent_run_failed(
            db_session,
            run.id,
            "LLM API timeout",
        )

        assert failed.status == "failed"
        assert failed.error_message == "LLM API timeout"
        assert failed.completed_at is not None


class TestLogToolCall:
    """Test tool call logging."""

    @pytest.mark.asyncio
    async def test_log_tool_call_success(self, db_session) -> None:
        user_id = uuid.uuid4()
        run = await create_agent_run(db_session, user_id, "test")

        log = await log_tool_call(
            db_session,
            run.id,
            tool_name="destination_knowledge_retrieval",
            input_json={"query": "beaches", "top_k": 5},
            output_summary="Retrieved 3 chunks about beaches",
            status="ok",
            latency_ms=120,
        )

        assert log.run_id == run.id
        assert log.tool_name == "destination_knowledge_retrieval"
        assert log.input_json == {"query": "beaches", "top_k": 5}
        assert log.output_summary == "Retrieved 3 chunks about beaches"
        assert log.status == "ok"
        assert log.latency_ms == 120

    @pytest.mark.asyncio
    async def test_log_tool_call_error(self, db_session) -> None:
        user_id = uuid.uuid4()
        run = await create_agent_run(db_session, user_id, "test")

        log = await log_tool_call(
            db_session,
            run.id,
            tool_name="fetch_live_weather",
            status="error",
            output_summary="Connection timeout",
            latency_ms=3000,
        )

        assert log.status == "error"
        assert log.output_summary == "Connection timeout"


class TestLogLLMUsage:
    """Test LLM usage logging."""

    @pytest.mark.asyncio
    async def test_log_llm_usage(self, db_session) -> None:
        user_id = uuid.uuid4()
        run = await create_agent_run(db_session, user_id, "test")

        log = await log_llm_usage(
            db_session,
            run.id,
            step_name="intent_extraction",
            model="claude-3-haiku-20240307",
            input_tokens=256,
            output_tokens=128,
            cost_usd=0.001,
            latency_ms=450,
        )

        assert log.run_id == run.id
        assert log.step_name == "intent_extraction"
        assert log.model == "claude-3-haiku-20240307"
        assert log.input_tokens == 256
        assert log.output_tokens == 128
        assert log.cost_usd == 0.001
        assert log.latency_ms == 450


class TestLogTraceEvent:
    """Test trace event logging."""

    @pytest.mark.asyncio
    async def test_log_trace_event(self, db_session) -> None:
        user_id = uuid.uuid4()
        run = await create_agent_run(db_session, user_id, "test")

        event = await log_trace_event(
            db_session,
            run.id,
            event_type="tool_call",
            event_name="RAG retrieval started",
            detail_json={"query": "beaches in summer"},
            latency_ms=None,
        )

        assert event.run_id == run.id
        assert event.event_type == "tool_call"
        assert event.event_name == "RAG retrieval started"
        assert event.detail_json == {"query": "beaches in summer"}


class TestListAgentRuns:
    """Test listing agent runs for a user."""

    @pytest.mark.asyncio
    async def test_list_runs_for_user(self, db_session) -> None:
        user_id_1 = uuid.uuid4()
        user_id_2 = uuid.uuid4()

        # Create runs for user 1
        run_1 = await create_agent_run(db_session, user_id_1, "query 1")
        run_2 = await create_agent_run(db_session, user_id_1, "query 2")

        # Create run for user 2
        run_3 = await create_agent_run(db_session, user_id_2, "query 3")

        # List runs for user 1
        user_1_runs = await list_agent_runs_for_user(db_session, user_id_1)

        assert len(user_1_runs) == 2
        assert run_1.id in [r.id for r in user_1_runs]
        assert run_2.id in [r.id for r in user_1_runs]
        assert run_3.id not in [r.id for r in user_1_runs]

    @pytest.mark.asyncio
    async def test_list_runs_respects_limit(self, db_session) -> None:
        user_id = uuid.uuid4()

        # Create 5 runs
        for i in range(5):
            await create_agent_run(db_session, user_id, f"query {i}")

        # List with limit 2
        runs = await list_agent_runs_for_user(db_session, user_id, limit=2)

        assert len(runs) == 2


class TestGetAgentRunForUser:
    """Test getting a specific agent run for a user."""

    @pytest.mark.asyncio
    async def test_get_run_for_correct_user(self, db_session) -> None:
        user_id = uuid.uuid4()
        run = await create_agent_run(db_session, user_id, "test")

        retrieved = await get_agent_run_for_user(db_session, run.id, user_id)

        assert retrieved is not None
        assert retrieved.id == run.id
        assert retrieved.user_id == user_id

    @pytest.mark.asyncio
    async def test_get_run_for_wrong_user(self, db_session) -> None:
        user_id_1 = uuid.uuid4()
        user_id_2 = uuid.uuid4()
        run = await create_agent_run(db_session, user_id_1, "test")

        # Try to get run as different user
        retrieved = await get_agent_run_for_user(db_session, run.id, user_id_2)

        assert retrieved is None

    @pytest.mark.asyncio
    async def test_get_nonexistent_run(self, db_session) -> None:
        user_id = uuid.uuid4()
        fake_run_id = uuid.uuid4()

        retrieved = await get_agent_run_for_user(db_session, fake_run_id, user_id)

        assert retrieved is None
