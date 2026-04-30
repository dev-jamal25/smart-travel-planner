"""Tests for weather schemas, destination coordinates, weather service, and endpoint.

No tests call the real Open-Meteo API. The service tests use a mock httpx.AsyncClient
and the endpoint tests override get_weather_service via dependency injection.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from pydantic import ValidationError

from app.dependencies import get_weather_service
from app.main import app
from app.schemas.weather import WeatherForecastRequest, WeatherForecastResponse
from app.services.destination_coordinates import (
    get_destination_coordinates,
    list_supported_destinations,
)
from app.services.weather_service import WeatherService, WeatherServiceError

# ---------------------------------------------------------------------------
# Canned Open-Meteo response used across service tests
# ---------------------------------------------------------------------------

_MOCK_OPEN_METEO = {
    "latitude": 35.0116,
    "longitude": 135.7681,
    "current": {
        "temperature_2m": 18.5,
        "weather_code": 1,
    },
    "daily": {
        "time": ["2024-04-01", "2024-04-02", "2024-04-03"],
        "temperature_2m_max": [22.0, 20.0, 19.5],
        "temperature_2m_min": [14.0, 13.5, 12.0],
        "precipitation_probability_max": [10.0, 30.0, 5.0],
    },
}

_VALID_REQUEST = {
    "destination": "Kyoto",
    "forecast_days": 3,
}


# ---------------------------------------------------------------------------
# Destination coordinate tests
# ---------------------------------------------------------------------------


def test_coordinates_lookup_exact_name() -> None:
    coords = get_destination_coordinates("Kyoto")
    assert coords is not None
    assert coords.name == "Kyoto"
    assert coords.latitude == pytest.approx(35.0116)


def test_coordinates_lookup_case_insensitive() -> None:
    assert get_destination_coordinates("kyoto") is not None
    assert get_destination_coordinates("KYOTO") is not None
    assert get_destination_coordinates("Kyoto") is not None


def test_coordinates_lookup_trims_whitespace() -> None:
    assert get_destination_coordinates("  kyoto  ") is not None


def test_coordinates_unsupported_destination_returns_none() -> None:
    assert get_destination_coordinates("London") is None
    assert get_destination_coordinates("") is None


def test_coordinates_krakow_accepts_ascii_and_unicode() -> None:
    assert get_destination_coordinates("krakow") is not None
    assert get_destination_coordinates("kraków") is not None


def test_list_supported_destinations_contains_all_ten() -> None:
    names = list_supported_destinations()
    assert len(names) == 10
    assert "Kyoto" in names
    assert "Dubai" in names
    assert "Singapore" in names


# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------


def test_schema_accepts_valid_request() -> None:
    req = WeatherForecastRequest(**_VALID_REQUEST)
    assert req.destination == "Kyoto"
    assert req.forecast_days == 3


def test_schema_default_forecast_days_is_3() -> None:
    req = WeatherForecastRequest(destination="Bali")
    assert req.forecast_days == 3


def test_schema_rejects_forecast_days_zero() -> None:
    with pytest.raises(ValidationError):
        WeatherForecastRequest(destination="Bali", forecast_days=0)


def test_schema_rejects_forecast_days_eight() -> None:
    with pytest.raises(ValidationError):
        WeatherForecastRequest(destination="Bali", forecast_days=8)


def test_schema_accepts_forecast_days_boundaries() -> None:
    WeatherForecastRequest(destination="Bali", forecast_days=1)
    WeatherForecastRequest(destination="Bali", forecast_days=7)


def test_schema_rejects_empty_destination() -> None:
    with pytest.raises(ValidationError):
        WeatherForecastRequest(destination="", forecast_days=3)


# ---------------------------------------------------------------------------
# WeatherService unit tests (no real HTTP)
# ---------------------------------------------------------------------------


def _make_mock_client(response_json: dict | None = None) -> httpx.AsyncClient:
    """Return a mock httpx.AsyncClient that returns a canned JSON response."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.raise_for_status.return_value = None
    mock_resp.status_code = 200
    mock_resp.json.return_value = response_json or _MOCK_OPEN_METEO

    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_client.get = AsyncMock(return_value=mock_resp)
    return mock_client  # type: ignore[return-value]


def _make_service(mock_client: httpx.AsyncClient | None = None) -> WeatherService:
    import os

    os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    os.environ.setdefault(
        "SUPABASE_JWT_JWKS_URL",
        "https://test.supabase.co/auth/v1/.well-known/jwks.json",
    )
    os.environ.setdefault("SUPABASE_JWT_ISSUER", "https://test.supabase.co/auth/v1")
    os.environ.setdefault("SUPABASE_JWT_AUDIENCE", "authenticated")
    os.environ.setdefault("SUPABASE_ANON_KEY", "test-anon")
    os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "test-srk")
    os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test")
    os.environ.setdefault("WEBHOOK_URL", "https://test.example.com/webhook")

    from app.config import get_settings

    return WeatherService(settings=get_settings(), http_client=mock_client or _make_mock_client())


async def test_service_parses_mock_response() -> None:
    service = _make_service()
    result = await service.get_forecast("Kyoto", 3)

    assert result.destination == "Kyoto"
    assert result.latitude == pytest.approx(35.0116)
    assert result.current_temperature_c == pytest.approx(18.5)
    assert result.current_weather_code == 1
    assert len(result.daily) == 3
    assert result.daily[0].date == "2024-04-01"
    assert result.daily[0].temperature_max_c == pytest.approx(22.0)
    assert result.daily[0].temperature_min_c == pytest.approx(14.0)
    assert result.daily[0].precipitation_probability_max == pytest.approx(10.0)
    assert result.cached is False
    assert result.source == "Open-Meteo"


async def test_service_second_call_returns_cached_true() -> None:
    mock_client = _make_mock_client()
    service = _make_service(mock_client=mock_client)

    first = await service.get_forecast("Kyoto", 3)
    second = await service.get_forecast("Kyoto", 3)

    assert first.cached is False
    assert second.cached is True
    # HTTP client should only have been called once (cache hit on second call)
    assert mock_client.get.call_count == 1  # type: ignore[union-attr]


async def test_service_raises_for_unsupported_destination() -> None:
    service = _make_service()
    with pytest.raises(WeatherServiceError) as exc_info:
        await service.get_forecast("London", 3)
    assert exc_info.value.retryable is False


async def test_service_raises_on_http_error() -> None:
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 500
    mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500", request=MagicMock(), response=mock_resp
    )
    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_client.get = AsyncMock(return_value=mock_resp)

    service = _make_service(mock_client=mock_client)  # type: ignore[arg-type]
    with pytest.raises(WeatherServiceError) as exc_info:
        await service.get_forecast("Kyoto", 3)
    assert exc_info.value.retryable is True


# ---------------------------------------------------------------------------
# Endpoint tests (dependency override, no real HTTP or lifespan needed)
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_weather_service() -> WeatherService:
    return _make_service()


@pytest.fixture()
def weather_client(client, fake_weather_service: WeatherService):  # type: ignore[no-untyped-def]
    app.dependency_overrides[get_weather_service] = lambda: fake_weather_service
    yield client
    app.dependency_overrides.pop(get_weather_service, None)


def test_forecast_endpoint_success(weather_client) -> None:  # type: ignore[no-untyped-def]
    mock_svc = MagicMock(spec=WeatherService)
    mock_svc.get_forecast = AsyncMock(
        return_value=WeatherForecastResponse(
            destination="Kyoto",
            latitude=35.0116,
            longitude=135.7681,
            current_temperature_c=18.5,
            current_weather_code=1,
            daily=[],
            cached=False,
        )
    )
    app.dependency_overrides[get_weather_service] = lambda: mock_svc

    response = weather_client.post("/weather/forecast", json=_VALID_REQUEST)
    assert response.status_code == 200
    data = response.json()
    assert data["destination"] == "Kyoto"
    assert data["source"] == "Open-Meteo"
    assert data["cached"] is False

    app.dependency_overrides.pop(get_weather_service, None)


def test_forecast_endpoint_unsupported_destination(weather_client) -> None:  # type: ignore[no-untyped-def]
    mock_svc = MagicMock(spec=WeatherService)
    mock_svc.get_forecast = AsyncMock(
        side_effect=WeatherServiceError("Unsupported destination", retryable=False)
    )
    app.dependency_overrides[get_weather_service] = lambda: mock_svc

    response = weather_client.post(
        "/weather/forecast", json={"destination": "London", "forecast_days": 3}
    )
    assert response.status_code == 422

    app.dependency_overrides.pop(get_weather_service, None)


def test_forecast_endpoint_open_meteo_failure_returns_502(weather_client) -> None:  # type: ignore[no-untyped-def]
    mock_svc = MagicMock(spec=WeatherService)
    mock_svc.get_forecast = AsyncMock(
        side_effect=WeatherServiceError("Network error", retryable=True)
    )
    app.dependency_overrides[get_weather_service] = lambda: mock_svc

    response = weather_client.post("/weather/forecast", json=_VALID_REQUEST)
    assert response.status_code == 502

    app.dependency_overrides.pop(get_weather_service, None)


def test_forecast_endpoint_invalid_forecast_days(weather_client) -> None:  # type: ignore[no-untyped-def]
    response = weather_client.post(
        "/weather/forecast", json={"destination": "Kyoto", "forecast_days": 10}
    )
    assert response.status_code == 422
