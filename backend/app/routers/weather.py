from __future__ import annotations

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_weather_service
from app.schemas.weather import WeatherForecastRequest, WeatherForecastResponse
from app.services.weather_service import WeatherService, WeatherServiceError

logger = structlog.get_logger(__name__)

# Public endpoint: weather data is already publicly available from Open-Meteo.
# Auth can be added before production if the project requires it.
router = APIRouter(prefix="/weather", tags=["weather"])


@router.post("/forecast", response_model=WeatherForecastResponse)
async def get_weather_forecast(
    body: WeatherForecastRequest,
    weather: Annotated[WeatherService, Depends(get_weather_service)],
) -> WeatherForecastResponse:
    logger.info(
        "weather.forecast.start",
        destination=body.destination,
        forecast_days=body.forecast_days,
    )
    try:
        result = await weather.get_forecast(
            destination=body.destination,
            forecast_days=body.forecast_days,
        )
    except WeatherServiceError as exc:
        logger.warning(
            "weather.forecast.error",
            destination=body.destination,
            error=str(exc),
            retryable=exc.retryable,
        )
        if not exc.retryable:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    logger.info(
        "weather.forecast.success",
        destination=result.destination,
        cached=result.cached,
        current_temp=result.current_temperature_c,
    )
    return result
