from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field


class WeatherForecastRequest(BaseModel):
    destination: Annotated[str, Field(min_length=1)]
    forecast_days: Annotated[int, Field(ge=1, le=7)] = 3


class DailyWeather(BaseModel):
    date: str
    temperature_min_c: float | None
    temperature_max_c: float | None
    precipitation_probability_max: float | None


class WeatherForecastResponse(BaseModel):
    destination: str
    latitude: float
    longitude: float
    current_temperature_c: float | None
    current_weather_code: int | None
    daily: list[DailyWeather]
    source: str = "Open-Meteo"
    cached: bool = False


class WeatherToolError(BaseModel):
    error: str
    retryable: bool
