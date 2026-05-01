"""Tests for agent tool wrappers."""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.schemas.agent_tools import (
    ClassifyDestinationInput,
    DestinationKnowledgeInput,
    LiveWeatherInput,
)
from app.schemas.classifier import ClassifierPredictionResponse
from app.schemas.rag import ChunkResult
from app.schemas.weather import DailyWeather, WeatherForecastResponse
from app.services.weather_service import WeatherServiceError
from app.tools.base import ToolResult
from app.tools.classifier_tool import classify_destination_style
from app.tools.rag_tool import destination_knowledge_retrieval
from app.tools.weather_tool import fetch_live_weather


class TestDestinationKnowledgeRetrievalTool:
    """Test RAG retrieval tool."""

    @pytest.mark.asyncio
    async def test_successful_retrieval(self) -> None:
        # Setup mock services
        mock_rag_service = MagicMock()
        mock_session = MagicMock()

        # Mock chunks returned from RAG
        mock_chunks = [
            ChunkResult(
                chunk_id=uuid.uuid4(),
                document_id=uuid.uuid4(),
                destination_name="Bali",
                source_title="Bali Travel Guide",
                chunk_text="Bali is a tropical island with beautiful beaches and culture...",
                chunk_index=0,
                cosine_distance=0.15,
            ),
            ChunkResult(
                chunk_id=uuid.uuid4(),
                document_id=uuid.uuid4(),
                destination_name="Bali",
                source_title="Bali Beaches",
                chunk_text="Beautiful beaches with warm water and water sports...",
                chunk_index=2,
                cosine_distance=0.18,
            ),
        ]
        mock_rag_service.retrieve_top_k = AsyncMock(return_value=mock_chunks)

        # Call tool
        tool_input = DestinationKnowledgeInput(
            query="beaches in tropical destinations", top_k=5
        )
        with patch("app.tools.rag_tool.logger"):
            result = await destination_knowledge_retrieval(
                tool_input, mock_rag_service, mock_session
            )

        # Verify result
        assert result.status == "ok"
        assert result.tool_name == "destination_knowledge_retrieval"
        assert result.output["query"] == "beaches in tropical destinations"
        assert result.output["top_k"] == 5
        assert len(result.output["chunks"]) == 2

        # Verify service was called correctly
        mock_rag_service.retrieve_top_k.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_rag_service_error(self) -> None:
        mock_rag_service = MagicMock()
        mock_session = MagicMock()
        mock_rag_service.retrieve_top_k = AsyncMock(
            side_effect=Exception("Connection failed")
        )

        tool_input = DestinationKnowledgeInput(query="test")
        with patch("app.tools.rag_tool.logger"):
            result = await destination_knowledge_retrieval(
                tool_input, mock_rag_service, mock_session
            )

        assert result.status == "error"
        assert "Connection failed" in result.error
        assert result.retryable is True

    @pytest.mark.asyncio
    async def test_with_destination_filter(self) -> None:
        mock_rag_service = MagicMock()
        mock_session = MagicMock()
        mock_rag_service.retrieve_top_k = AsyncMock(return_value=[])

        tool_input = DestinationKnowledgeInput(
            query="adventure activities", destination_filter="Bali", top_k=3
        )
        with patch("app.tools.rag_tool.logger"):
            await destination_knowledge_retrieval(
                tool_input, mock_rag_service, mock_session
            )

        # Verify the tool called RAG service with correct parameters
        mock_rag_service.retrieve_top_k.assert_awaited_once()
        call_args = mock_rag_service.retrieve_top_k.call_args
        assert call_args[1]["query"] == "adventure activities"
        assert call_args[1]["top_k"] == 3
        assert call_args[1]["destination_filter"] == "Bali"


class TestClassifyDestinationStyleTool:
    """Test destination style classification tool."""

    @pytest.mark.asyncio
    async def test_successful_classification(self) -> None:
        mock_classifier_service = MagicMock()
        mock_classifier_service.predict = MagicMock(
            return_value=ClassifierPredictionResponse(travel_style="Adventure", confidence=0.92)
        )

        tool_input = ClassifyDestinationInput(destination="Interlaken")
        with patch("app.tools.classifier_tool.logger"):
            result = await classify_destination_style(tool_input, mock_classifier_service)

        assert result.status == "ok"
        assert result.tool_name == "classify_destination_style"
        assert result.output["destination"] == "Interlaken"
        assert result.output["travel_style"] == "Adventure"
        assert result.output["confidence"] == pytest.approx(0.92)

    @pytest.mark.asyncio
    async def test_output_is_flat_dict_with_correct_types(self) -> None:
        """output dict must have scalar values — not tuples or nested objects."""
        mock_classifier_service = MagicMock()
        mock_classifier_service.predict = MagicMock(
            return_value=ClassifierPredictionResponse(travel_style="Family", confidence=0.998)
        )

        tool_input = ClassifyDestinationInput(destination="Santorini")
        with patch("app.tools.classifier_tool.logger"):
            result = await classify_destination_style(tool_input, mock_classifier_service)

        assert result.status == "ok"
        travel_style = result.output["travel_style"]
        confidence = result.output["confidence"]
        # Must be plain scalars — not tuples, lists, or Pydantic objects
        assert isinstance(travel_style, str), f"travel_style was {type(travel_style)}"
        assert isinstance(confidence, float), f"confidence was {type(confidence)}"
        # Confidence must be formattable as a percentage without crashing
        assert f"{confidence:.0%}"  # would crash if confidence were a tuple

    @pytest.mark.asyncio
    async def test_unsupported_destination(self) -> None:
        mock_classifier_service = MagicMock()

        tool_input = ClassifyDestinationInput(destination="NonExistentPlace")
        with patch("app.tools.classifier_tool.logger"):
            result = await classify_destination_style(tool_input, mock_classifier_service)

        assert result.status == "error"
        assert "not in the supported list" in result.error
        assert result.retryable is False

    @pytest.mark.asyncio
    async def test_classifier_error(self) -> None:
        mock_classifier_service = MagicMock()
        mock_classifier_service.predict = MagicMock(
            side_effect=Exception("Model error")
        )

        tool_input = ClassifyDestinationInput(destination="Bali")
        with patch("app.tools.classifier_tool.logger"):
            result = await classify_destination_style(tool_input, mock_classifier_service)

        assert result.status == "error"
        assert "Model error" in result.error
        assert result.retryable is False


class TestFetchLiveWeatherTool:
    """Test live weather tool."""

    @pytest.mark.asyncio
    async def test_successful_weather_fetch(self) -> None:
        mock_weather_service = MagicMock()

        # Create mock forecast
        mock_forecast = WeatherForecastResponse(
            destination="Bali",
            latitude=0.0,
            longitude=115.0,
            current_temperature_c=28.0,
            current_weather_code=0,
            daily=[
                DailyWeather(
                    date="2025-04-30",
                    temperature_max_c=28,
                    temperature_min_c=24,
                    precipitation_probability_max=5,
                ),
                DailyWeather(
                    date="2025-05-01",
                    temperature_max_c=29,
                    temperature_min_c=25,
                    precipitation_probability_max=2,
                ),
            ],
        )
        mock_weather_service.get_forecast = AsyncMock(return_value=mock_forecast)

        tool_input = LiveWeatherInput(destination="Bali", forecast_days=3)
        with patch("app.tools.weather_tool.logger"):
            result = await fetch_live_weather(tool_input, mock_weather_service)

        assert result.status == "ok"
        assert result.tool_name == "fetch_live_weather"
        assert result.output["destination"] == "Bali"
        assert len(result.output["daily"]) == 2

        # Verify service was called
        mock_weather_service.get_forecast.assert_awaited_once_with(
            destination="Bali", forecast_days=3
        )

    @pytest.mark.asyncio
    async def test_weather_service_error(self) -> None:
        mock_weather_service = MagicMock()
        mock_weather_service.get_forecast = AsyncMock(
            side_effect=WeatherServiceError("API limit exceeded")
        )

        tool_input = LiveWeatherInput(destination="Bali")
        with patch("app.tools.weather_tool.logger"):
            result = await fetch_live_weather(tool_input, mock_weather_service)

        assert result.status == "error"
        assert "API limit exceeded" in result.error
        assert result.retryable is True

    @pytest.mark.asyncio
    async def test_weather_service_generic_error(self) -> None:
        mock_weather_service = MagicMock()
        mock_weather_service.get_forecast = AsyncMock(
            side_effect=Exception("Unexpected error")
        )

        tool_input = LiveWeatherInput(destination="Bali")
        with patch("app.tools.weather_tool.logger"):
            result = await fetch_live_weather(tool_input, mock_weather_service)

        assert result.status == "error"
        assert "Unexpected error" in result.error
        assert result.retryable is False


class TestToolResultHelpers:
    """Test ToolResult helper methods."""

    def test_result_ok_helper(self) -> None:
        output = {"key": "value"}
        result = ToolResult.ok("test_tool", output)

        assert result.status == "ok"
        assert result.tool_name == "test_tool"
        assert result.output == output
        assert result.error is None

    def test_result_fail_helper(self) -> None:
        result = ToolResult.fail("test_tool", "Something went wrong")

        assert result.status == "error"
        assert result.tool_name == "test_tool"
        assert result.error == "Something went wrong"
        assert result.retryable is False

    def test_result_fail_with_retryable(self) -> None:
        result = ToolResult.fail(
            "test_tool", "Network timeout", retryable=True
        )

        assert result.status == "error"
        assert result.retryable is True
