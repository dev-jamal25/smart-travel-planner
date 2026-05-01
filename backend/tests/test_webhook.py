"""Tests for Discord webhook schemas, service, and endpoint.

No tests call the real Discord API. Service tests use a mock httpx.AsyncClient.
Endpoint tests use dependency overrides so no lifespan setup is required.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from pydantic import ValidationError

from app.dependencies import get_webhook_service
from app.main import app
from app.schemas.webhook import WebhookDeliveryResponse, WebhookTripPlanRequest
from app.services.webhook_service import WebhookService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_REQUEST = {
    "user_query": "I want two weeks in July, warm weather, hiking, and around $1500.",
    "destination": "Interlaken",
    "trip_plan": "Interlaken is a strong adventure match with mountain access and lakes.",
}


def _make_mock_http_client(status_code: int = 204) -> httpx.AsyncClient:
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = status_code
    mock_resp.raise_for_status.return_value = None
    mock_resp.request = MagicMock()

    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_client.post = AsyncMock(return_value=mock_resp)
    return mock_client  # type: ignore[return-value]


def _make_service(mock_client: httpx.AsyncClient | None = None) -> WebhookService:
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
    os.environ.setdefault("WEBHOOK_URL", "https://discord.example.com/webhooks/test")

    from app.config import get_settings

    return WebhookService(settings=get_settings(), http_client=mock_client or _make_mock_http_client())


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


def test_schema_accepts_valid_payload() -> None:
    req = WebhookTripPlanRequest(**_VALID_REQUEST)
    assert req.user_query.startswith("I want")
    assert req.destination == "Interlaken"
    assert req.trip_plan.startswith("Interlaken")


def test_schema_accepts_minimal_fields_only() -> None:
    req = WebhookTripPlanRequest(
        user_query="Give me a beach trip",
        trip_plan="Go to Bali.",
    )
    assert req.destination is None


def test_schema_rejects_empty_user_query() -> None:
    with pytest.raises(ValidationError):
        WebhookTripPlanRequest(user_query="", trip_plan="Some plan")


def test_schema_rejects_whitespace_user_query() -> None:
    with pytest.raises(ValidationError):
        WebhookTripPlanRequest(user_query="   ", trip_plan="Some plan")


def test_schema_rejects_empty_trip_plan() -> None:
    with pytest.raises(ValidationError):
        WebhookTripPlanRequest(user_query="A query", trip_plan="")


def test_discord_payload_excludes_internal_fields() -> None:
    """Verify that tools_used and cost_usd are not in Discord payload."""
    # Schema no longer accepts these fields, so this test confirms the contract
    with pytest.raises(ValidationError):
        WebhookTripPlanRequest(
            user_query="A query",
            trip_plan="A plan",
            tools_used=["tool1"],  # type: ignore[call-arg]
        )
    with pytest.raises(ValidationError):
        WebhookTripPlanRequest(
            user_query="A query",
            trip_plan="A plan",
            cost_usd=0.05,  # type: ignore[call-arg]
        )


# ---------------------------------------------------------------------------
# WebhookService unit tests (no real HTTP)
# ---------------------------------------------------------------------------


async def test_service_sends_correct_discord_payload() -> None:
    mock_client = _make_mock_http_client(status_code=204)
    service = _make_service(mock_client=mock_client)

    req = WebhookTripPlanRequest(**_VALID_REQUEST)
    result = await service.send_trip_plan(req)

    assert result.delivered is True
    assert result.status_code == 204

    # Verify the HTTP client was called once with a JSON body containing key fields
    mock_client.post.assert_awaited_once()  # type: ignore[union-attr]
    call_kwargs = mock_client.post.call_args  # type: ignore[union-attr]
    payload = call_kwargs.kwargs.get("json") or call_kwargs.args[1]
    assert "content" in payload
    assert "Smart Travel Planner" in payload["content"]
    assert req.destination in payload["content"]
    assert req.user_query in payload["content"]


async def test_service_returns_delivered_false_on_network_failure() -> None:
    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_client.post = AsyncMock(
        side_effect=httpx.RequestError("Connection refused", request=MagicMock())
    )

    service = _make_service(mock_client=mock_client)  # type: ignore[arg-type]
    req = WebhookTripPlanRequest(**_VALID_REQUEST)
    result = await service.send_trip_plan(req)

    assert result.delivered is False
    assert result.error is not None


async def test_service_returns_delivered_false_on_4xx() -> None:
    mock_client = _make_mock_http_client(status_code=400)
    service = _make_service(mock_client=mock_client)

    req = WebhookTripPlanRequest(**_VALID_REQUEST)
    result = await service.send_trip_plan(req)

    assert result.delivered is False
    assert result.status_code == 400


async def test_service_does_not_expose_webhook_url_in_response() -> None:
    mock_client = _make_mock_http_client(status_code=204)
    service = _make_service(mock_client=mock_client)

    req = WebhookTripPlanRequest(**_VALID_REQUEST)
    result = await service.send_trip_plan(req)

    # The response object must not contain the webhook URL
    response_dict = result.model_dump()
    for value in response_dict.values():
        if isinstance(value, str):
            assert "discord.example.com" not in value
            assert "webhooks" not in value


# ---------------------------------------------------------------------------
# Endpoint tests (dependency overrides, no lifespan needed)
# ---------------------------------------------------------------------------


@pytest.fixture()
def webhook_client(client):  # type: ignore[no-untyped-def]
    mock_svc = MagicMock(spec=WebhookService)
    mock_svc.send_trip_plan = AsyncMock(
        return_value=WebhookDeliveryResponse(delivered=True, status_code=204)
    )
    app.dependency_overrides[get_webhook_service] = lambda: mock_svc
    yield client, mock_svc
    app.dependency_overrides.pop(get_webhook_service, None)


def test_endpoint_returns_200_on_success(webhook_client) -> None:  # type: ignore[no-untyped-def]
    client, _ = webhook_client
    response = client.post("/webhook/test-discord", json=_VALID_REQUEST)
    assert response.status_code == 200
    data = response.json()
    assert data["delivered"] is True
    assert data["provider"] == "discord"


def test_endpoint_returns_200_with_delivered_false_on_discord_failure(webhook_client) -> None:  # type: ignore[no-untyped-def]
    client, mock_svc = webhook_client
    mock_svc.send_trip_plan = AsyncMock(
        return_value=WebhookDeliveryResponse(
            delivered=False,
            status_code=400,
            error="Discord rejected the payload",
        )
    )
    response = client.post("/webhook/test-discord", json=_VALID_REQUEST)
    # Failure isolation: always 200, but delivered=False
    assert response.status_code == 200
    data = response.json()
    assert data["delivered"] is False
    assert data["error"] is not None


def test_endpoint_returns_401_without_auth(client) -> None:  # type: ignore[no-untyped-def]
    from app.dependencies import get_current_user

    app.dependency_overrides.pop(get_current_user, None)
    response = client.post("/webhook/test-discord", json=_VALID_REQUEST)
    # Missing auth returns 401 (AuthError handler)
    assert response.status_code == 401


def test_endpoint_returns_422_on_invalid_body(webhook_client) -> None:  # type: ignore[no-untyped-def]
    client, _ = webhook_client
    response = client.post(
        "/webhook/test-discord",
        json={"user_query": "", "trip_plan": "Some plan"},
    )
    assert response.status_code == 422
