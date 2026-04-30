from __future__ import annotations

import httpx
import structlog
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import Settings
from app.schemas.webhook import WebhookDeliveryResponse, WebhookTripPlanRequest

logger = structlog.get_logger(__name__)

_REQUEST_TIMEOUT = httpx.Timeout(10.0)
_DISCORD_CONTENT_LIMIT = 2000
_TRIP_PLAN_LIMIT = 1400  # leave room for the rest of the message


def _build_discord_payload(req: WebhookTripPlanRequest) -> dict[str, object]:
    """Build a Discord webhook JSON payload from a trip plan request.
    
    Includes only user-facing information: query, destination, and trip plan.
    Internal details like tool usage and cost are stored in DB/logs, not sent to Discord.
    """
    trip_plan = req.trip_plan
    if len(trip_plan) > _TRIP_PLAN_LIMIT:
        trip_plan = trip_plan[:_TRIP_PLAN_LIMIT] + " … [truncated]"

    lines: list[str] = [
        "## ✈️ Smart Travel Planner — New Trip Plan",
        f"**Query:** {req.user_query}",
    ]
    if req.destination:
        lines.append(f"**Destination:** {req.destination}")
    lines.append(f"**Plan:**\n{trip_plan}")

    content = "\n".join(lines)
    # Hard safety clamp so Discord never rejects the message
    if len(content) > _DISCORD_CONTENT_LIMIT:
        content = content[: _DISCORD_CONTENT_LIMIT - 20] + "\n… [truncated]"

    return {"content": content}


class WebhookService:
    """Delivers trip plan notifications to a Discord webhook.

    One instance should live for the application lifetime (lifespan singleton).
    Failures are isolated: callers always receive a WebhookDeliveryResponse,
    never an unhandled exception.
    """

    def __init__(
        self,
        settings: Settings,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._webhook_url = settings.webhook_url
        self._owned_client = http_client is None
        self._http = http_client or httpx.AsyncClient(timeout=_REQUEST_TIMEOUT)

    async def close(self) -> None:
        if self._owned_client:
            await self._http.aclose()

    async def send_trip_plan(self, req: WebhookTripPlanRequest) -> WebhookDeliveryResponse:
        """Send a trip plan to Discord. Never raises; returns delivered=False on failure."""
        payload = _build_discord_payload(req)
        logger.info(
            "webhook.discord.start",
            destination=req.destination,
        )
        try:
            return await self._send_with_retry(payload)
        except Exception as exc:
            logger.error("webhook.discord.failure", error=str(exc), exc_info=True)
            return WebhookDeliveryResponse(delivered=False, error=str(exc))

    async def _send_with_retry(self, payload: dict[str, object]) -> WebhookDeliveryResponse:
        status_code: int | None = None
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=0.5, max=8),
                # Retry only on transient network errors and 5xx; never on 4xx
                retry=retry_if_exception_type(httpx.RequestError),
                reraise=True,
            ):
                with attempt:
                    logger.debug(
                        "webhook.discord.attempt",
                        attempt_number=attempt.retry_state.attempt_number,
                    )
                    resp = await self._http.post(self._webhook_url, json=payload)
                    status_code = resp.status_code
                    if resp.status_code >= 500:
                        # Treat 5xx as retryable by re-raising a RequestError equivalent
                        raise httpx.RequestError(
                            f"Discord returned HTTP {resp.status_code}",
                            request=resp.request,
                        )
                    if resp.status_code >= 400:
                        # 4xx is a client error — do not retry, return delivered=False
                        logger.warning(
                            "webhook.discord.client_error",
                            status=resp.status_code,
                        )
                        return WebhookDeliveryResponse(
                            delivered=False,
                            status_code=resp.status_code,
                            error=f"Discord rejected the payload with HTTP {resp.status_code}",
                        )

        except httpx.RequestError as exc:
            logger.error("webhook.discord.network_error", error=str(exc))
            return WebhookDeliveryResponse(
                delivered=False,
                status_code=status_code,
                error=f"Network error after retries: {exc}",
            )

        logger.info("webhook.discord.success", status=status_code)
        return WebhookDeliveryResponse(delivered=True, status_code=status_code)
