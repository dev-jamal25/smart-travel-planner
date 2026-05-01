"""Chat router — user-facing travel planning endpoint."""
from __future__ import annotations

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_agent_service, get_current_user, get_session
from app.schemas.agent import PlanTripRequest, PlanTripResponse
from app.schemas.auth import CurrentUser
from app.services.agent_service import AgentService

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/plan-trip", response_model=PlanTripResponse)
async def plan_trip(
    body: PlanTripRequest,
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
    agent_service: Annotated[AgentService, Depends(get_agent_service)],
) -> PlanTripResponse:
    """Run the travel-planning agent and return a synthesized trip plan.

    Internal costs, token counts, tool call logs, and trace events are persisted
    to the database but are never included in the API response.
    """
    logger.info(
        "chat.plan_trip.start",
        user_id=str(current_user.user_id),
        message_length=len(body.message),
    )

    try:
        response = await agent_service.plan_trip(
            session=session,
            current_user=current_user,
            request=body,
        )
    except Exception as exc:
        logger.error(
            "chat.plan_trip.failed",
            user_id=str(current_user.user_id),
            error=str(exc),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Travel planning failed. Please try again.",
        ) from exc

    logger.info(
        "chat.plan_trip.success",
        user_id=str(current_user.user_id),
        run_id=str(response.run_id),
        destination=response.recommended_destination,
        webhook_delivered=response.webhook_delivered,
    )
    return response
