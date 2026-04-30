from __future__ import annotations

from typing import Any

import httpx
import structlog
from cachetools import TTLCache
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import Settings
from app.schemas.weather import DailyWeather, WeatherForecastResponse
from app.services.destination_coordinates import get_destination_coordinates

logger = structlog.get_logger(__name__)

_REQUEST_TIMEOUT = httpx.Timeout(10.0)


class WeatherServiceError(Exception):
    """Raised when the weather service cannot return a forecast."""

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


class WeatherService:
    """Fetches weather forecasts from Open-Meteo with retry and TTL caching.

    One instance should be created per application lifetime (lifespan singleton).
    The TTL cache is held on the instance; creating a new instance resets the cache.
    """

    def __init__(
        self,
        settings: Settings,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        self._owned_client = http_client is None
        self._http = http_client or httpx.AsyncClient(timeout=_REQUEST_TIMEOUT)
        self._cache: TTLCache[str, WeatherForecastResponse] = TTLCache(
            maxsize=128, ttl=settings.weather_cache_ttl_seconds
        )

    async def close(self) -> None:
        """Release the owned httpx.AsyncClient if we created it."""
        if self._owned_client:
            await self._http.aclose()

    async def get_forecast(
        self,
        destination: str,
        forecast_days: int,
    ) -> WeatherForecastResponse:
        """Return a weather forecast for a supported destination.

        Raises WeatherServiceError if the destination is unsupported or
        if Open-Meteo cannot be reached after retries.
        """
        coords = get_destination_coordinates(destination)
        if coords is None:
            raise WeatherServiceError(
                f"'{destination}' is not a supported destination. "
                f"Supported: Interlaken, Banff, Bali, Santorini, Kyoto, "
                f"Istanbul, Tbilisi, Kraków, Dubai, Singapore.",
                retryable=False,
            )

        cache_key = f"{destination.strip().lower()}:{forecast_days}"
        if cache_key in self._cache:
            logger.debug("weather.cache.hit", destination=destination, forecast_days=forecast_days)
            cached: WeatherForecastResponse = self._cache[cache_key]
            return cached.model_copy(update={"cached": True})

        logger.info(
            "weather.fetch.start",
            destination=destination,
            latitude=coords.latitude,
            longitude=coords.longitude,
            forecast_days=forecast_days,
        )
        data = await self._fetch_with_retry(coords.latitude, coords.longitude, forecast_days)
        response = _parse_open_meteo(destination, coords.latitude, coords.longitude, data)
        self._cache[cache_key] = response

        logger.info(
            "weather.fetch.success",
            destination=destination,
            current_temp=response.current_temperature_c,
        )
        return response

    async def _fetch_with_retry(
        self, latitude: float, longitude: float, forecast_days: int
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,weather_code",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "timezone": "auto",
            "forecast_days": forecast_days,
        }
        data: dict[str, Any] | None = None

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=0.5, max=8),
                retry=retry_if_exception_type(httpx.RequestError),
                reraise=True,
            ):
                with attempt:
                    logger.debug(
                        "weather.fetch.attempt",
                        attempt_number=attempt.retry_state.attempt_number,
                    )
                    resp = await self._http.get(self._settings.weather_api_url, params=params)
                    resp.raise_for_status()
                    data = dict(resp.json())

        except httpx.HTTPStatusError as exc:
            logger.error(
                "weather.fetch.http_error",
                status=exc.response.status_code,
                error=str(exc),
            )
            raise WeatherServiceError(
                f"Open-Meteo returned HTTP {exc.response.status_code}.",
                retryable=exc.response.status_code >= 500,
            ) from exc
        except httpx.RequestError as exc:
            logger.error("weather.fetch.network_error", error=str(exc))
            raise WeatherServiceError(
                "Network error reaching Open-Meteo after retries.",
                retryable=True,
            ) from exc

        if data is None:
            # Should not be reachable: tenacity reraises on failure, so if we
            # get here the loop exited cleanly but data was never set.
            raise WeatherServiceError("No data received from Open-Meteo.", retryable=True)

        return data


def _parse_open_meteo(
    destination: str,
    latitude: float,
    longitude: float,
    data: dict[str, Any],
) -> WeatherForecastResponse:
    """Convert a raw Open-Meteo JSON response into a WeatherForecastResponse."""
    current = data.get("current", {})
    daily_raw = data.get("daily", {})

    dates: list[str] = daily_raw.get("time", [])
    mins: list[Any] = daily_raw.get("temperature_2m_min", [])
    maxs: list[Any] = daily_raw.get("temperature_2m_max", [])
    precips: list[Any] = daily_raw.get("precipitation_probability_max", [])

    daily = [
        DailyWeather(
            date=dates[i],
            temperature_min_c=mins[i] if i < len(mins) else None,
            temperature_max_c=maxs[i] if i < len(maxs) else None,
            precipitation_probability_max=precips[i] if i < len(precips) else None,
        )
        for i in range(len(dates))
    ]

    return WeatherForecastResponse(
        destination=destination,
        latitude=latitude,
        longitude=longitude,
        current_temperature_c=current.get("temperature_2m"),
        current_weather_code=current.get("weather_code"),
        daily=daily,
        cached=False,
    )
