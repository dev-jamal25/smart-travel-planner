from __future__ import annotations

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends

from app.dependencies import get_current_user, get_webhook_service
from app.schemas.auth import CurrentUser
from app.schemas.webhook import WebhookDeliveryResponse, WebhookTripPlanRequest
from app.services.webhook_service import WebhookService

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/webhook", tags=["webhook"])


@router.post("/test-discord", response_model=WebhookDeliveryResponse)
async def test_discord_webhook(
    body: WebhookTripPlanRequest,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    webhook: Annotated[WebhookService, Depends(get_webhook_service)],
) -> WebhookDeliveryResponse:
    """Send a trip plan to the configured Discord webhook.

    Returns delivered=True on success. On Discord failure, returns delivered=False
    with the error detail — the endpoint itself always returns HTTP 200 to demonstrate
    failure isolation.
    """
    logger.info(
        "webhook.test_discord.start",
        user_id=str(current_user.user_id),
        destination=body.destination,
    )
    result = await webhook.send_trip_plan(body)
    if result.delivered:
        logger.info(
            "webhook.test_discord.success",
            user_id=str(current_user.user_id),
            status_code=result.status_code,
        )
    else:
        logger.warning(
            "webhook.test_discord.failed",
            user_id=str(current_user.user_id),
            error=result.error,
            status_code=result.status_code,
        )
    return result
