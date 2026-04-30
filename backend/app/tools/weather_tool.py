"""Weather tool for agent."""
from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from app.schemas.agent_tools import LiveWeatherInput
from app.services.weather_service import WeatherServiceError
from app.tools.base import ToolResult

if TYPE_CHECKING:
    from app.services.weather_service import WeatherService

logger = structlog.get_logger(__name__)


async def fetch_live_weather(
    tool_input: LiveWeatherInput,
    weather_service: WeatherService,
) -> ToolResult:
    """Fetch live weather forecast for a destination.

    Args:
        tool_input: Destination and forecast days.
        weather_service: Weather service dependency.

    Returns:
        ToolResult with forecast if successful, error otherwise.
    """
    try:
        # Validate input
        tool_input = LiveWeatherInput(**tool_input.model_dump())

        logger.info(
            "weather_tool.fetch_start",
            destination=tool_input.destination,
            forecast_days=tool_input.forecast_days,
        )

        # Call weather service
        forecast = await weather_service.get_forecast(
            destination=tool_input.destination,
            forecast_days=tool_input.forecast_days,
        )

        logger.info(
            "weather_tool.fetch_success",
            destination=tool_input.destination,
            num_days=len(forecast.daily),
        )

        # Format forecast for agent
        output = {
            "destination": forecast.destination,
            "latitude": forecast.latitude,
            "longitude": forecast.longitude,
            "current_temperature_c": forecast.current_temperature_c,
            "daily": [
                {
                    "date": day.date,
                    "temperature_max_c": day.temperature_max_c,
                    "temperature_min_c": day.temperature_min_c,
                    "precipitation_probability_max": day.precipitation_probability_max,
                }
                for day in forecast.daily
            ],
        }

        return ToolResult.ok("fetch_live_weather", output)

    except WeatherServiceError as exc:
        logger.warning(
            "weather_tool.fetch_failure",
            destination=tool_input.destination,
            error=str(exc),
        )
        return ToolResult.fail(
            "fetch_live_weather",
            f"Could not fetch weather: {exc}",
            retryable=True,  # API errors are often transient
        )
    except Exception as exc:
        logger.error(
            "weather_tool.fetch_error",
            destination=tool_input.destination,
            error=str(exc),
            exc_info=True,
        )
        return ToolResult.fail(
            "fetch_live_weather",
            f"Weather tool error: {exc}",
            retryable=False,
        )
