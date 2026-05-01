"""Tests for agent schemas."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.agent import AgentToolError, PlanTripRequest, TripIntent
from app.schemas.agent_tools import (
    ClassifyDestinationInput,
    DestinationKnowledgeInput,
    LiveWeatherInput,
)


class TestTripIntent:
    """Test TripIntent schema."""

    def test_accepts_all_none_values(self) -> None:
        intent = TripIntent()
        assert intent.budget_usd is None
        assert intent.duration_days is None
        assert intent.preferred_activities == []

    def test_accepts_partial_values(self) -> None:
        intent = TripIntent(budget_usd=2000, preferred_style="adventure")
        assert intent.budget_usd == 2000
        assert intent.preferred_style == "adventure"
        assert intent.duration_days is None

    def test_rejects_negative_budget(self) -> None:
        with pytest.raises(ValidationError):
            TripIntent(budget_usd=-100)

    def test_rejects_zero_duration(self) -> None:
        with pytest.raises(ValidationError):
            TripIntent(duration_days=0)

    def test_accepts_positive_duration(self) -> None:
        intent = TripIntent(duration_days=7)
        assert intent.duration_days == 7

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            TripIntent(unknown_field="value")  # type: ignore[call-arg]


class TestPlanTripRequest:
    """Test PlanTripRequest schema."""

    def test_accepts_valid_message(self) -> None:
        req = PlanTripRequest(message="I want a beach trip")
        assert req.message == "I want a beach trip"

    def test_strips_whitespace(self) -> None:
        req = PlanTripRequest(message="  beach trip  ")
        assert req.message == "beach trip"

    def test_rejects_empty_message(self) -> None:
        with pytest.raises(ValidationError):
            PlanTripRequest(message="")

    def test_rejects_whitespace_only(self) -> None:
        with pytest.raises(ValidationError):
            PlanTripRequest(message="   ")

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            PlanTripRequest(message="trip", extra="field")  # type: ignore[call-arg]


class TestAgentToolError:
    """Test AgentToolError schema."""

    def test_creates_error(self) -> None:
        err = AgentToolError(tool_name="rag_retrieval", error="Not found")
        assert err.tool_name == "rag_retrieval"
        assert err.error == "Not found"
        assert err.retryable is False

    def test_marks_retryable(self) -> None:
        err = AgentToolError(
            tool_name="weather", error="API timeout", retryable=True
        )
        assert err.retryable is True


class TestDestinationKnowledgeInput:
    """Test DestinationKnowledgeInput schema."""

    def test_accepts_query_only(self) -> None:
        inp = DestinationKnowledgeInput(query="hiking destination")
        assert inp.query == "hiking destination"
        assert inp.top_k == 5
        assert inp.destination_filter is None

    def test_top_k_defaults_to_5(self) -> None:
        inp = DestinationKnowledgeInput(query="beaches")
        assert inp.top_k == 5

    def test_top_k_in_valid_range(self) -> None:
        inp = DestinationKnowledgeInput(query="test", top_k=1)
        assert inp.top_k == 1
        inp = DestinationKnowledgeInput(query="test", top_k=10)
        assert inp.top_k == 10

    def test_rejects_top_k_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            DestinationKnowledgeInput(query="test", top_k=0)
        with pytest.raises(ValidationError):
            DestinationKnowledgeInput(query="test", top_k=11)

    def test_normalizes_destination_filter(self) -> None:
        # None stays None
        inp = DestinationKnowledgeInput(query="test", destination_filter=None)
        assert inp.destination_filter is None

        # Empty string becomes None
        inp = DestinationKnowledgeInput(query="test", destination_filter="")
        assert inp.destination_filter is None

        # "none" becomes None (case-insensitive)
        inp = DestinationKnowledgeInput(query="test", destination_filter="none")
        assert inp.destination_filter is None
        inp = DestinationKnowledgeInput(query="test", destination_filter="NONE")
        assert inp.destination_filter is None

        # "null" becomes None
        inp = DestinationKnowledgeInput(query="test", destination_filter="null")
        assert inp.destination_filter is None

        # "all" becomes None
        inp = DestinationKnowledgeInput(query="test", destination_filter="all")
        assert inp.destination_filter is None

        # Real destination names preserved and stripped
        inp = DestinationKnowledgeInput(query="test", destination_filter="  Bali  ")
        assert inp.destination_filter == "Bali"


class TestClassifyDestinationInput:
    """Test ClassifyDestinationInput schema."""

    def test_accepts_destination(self) -> None:
        inp = ClassifyDestinationInput(destination="Bali")
        assert inp.destination == "Bali"

    def test_strips_whitespace(self) -> None:
        inp = ClassifyDestinationInput(destination="  Interlaken  ")
        assert inp.destination == "Interlaken"

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValidationError):
            ClassifyDestinationInput(destination="")


class TestLiveWeatherInput:
    """Test LiveWeatherInput schema."""

    def test_accepts_destination_only(self) -> None:
        inp = LiveWeatherInput(destination="Bali")
        assert inp.destination == "Bali"
        assert inp.forecast_days == 3

    def test_forecast_days_in_valid_range(self) -> None:
        inp = LiveWeatherInput(destination="test", forecast_days=1)
        assert inp.forecast_days == 1
        inp = LiveWeatherInput(destination="test", forecast_days=7)
        assert inp.forecast_days == 7

    def test_rejects_forecast_days_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            LiveWeatherInput(destination="test", forecast_days=0)
        with pytest.raises(ValidationError):
            LiveWeatherInput(destination="test", forecast_days=8)
