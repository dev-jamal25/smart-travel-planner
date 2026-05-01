"""AgentService — orchestrates the LangGraph travel-planning graph."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.graph import build_travel_graph
from app.agents.state import AgentGraphState
from app.db.repositories.agent_runs import (
    create_agent_run,
    log_trace_event,
    mark_agent_run_completed,
    mark_agent_run_failed,
)
from app.schemas.agent import PlanTripRequest, PlanTripResponse
from app.tracing import plan_trip_trace

if TYPE_CHECKING:
    from app.agents.model_router import ModelRouter
    from app.schemas.auth import CurrentUser
    from app.services.classifier_service import ClassifierService
    from app.services.rag_service import RagService
    from app.services.weather_service import WeatherService
    from app.services.webhook_service import WebhookService

logger = structlog.get_logger(__name__)


class AgentService:
    """Orchestrates the LangGraph travel-planning graph.

    One instance is created per request (because ``rag_service`` and
    ``classifier_service`` are per-request wrappers).  The graph is compiled
    in ``__init__`` — compilation is fast (no network I/O).
    """

    def __init__(
        self,
        model_router: ModelRouter,
        rag_service: RagService,
        classifier_service: ClassifierService,
        weather_service: WeatherService,
        webhook_service: WebhookService,
    ) -> None:
        self._model_router = model_router
        self._rag_service = rag_service
        self._classifier_service = classifier_service
        self._weather_service = weather_service
        self._webhook_service = webhook_service

        self._graph: Any = build_travel_graph(
            model_router=model_router,
            rag_service=rag_service,
            classifier_service=classifier_service,
            weather_service=weather_service,
            webhook_service=webhook_service,
        )

    @plan_trip_trace
    async def plan_trip(
        self,
        session: AsyncSession,
        current_user: CurrentUser,
        request: PlanTripRequest,
    ) -> PlanTripResponse:
        """Run the full travel-planning graph and return a user-facing response.

        Creates an AgentRun at the start, invokes the LangGraph graph, then
        marks the run completed or failed.  Commits the session once at the
        service boundary.

        Webhook failure does NOT raise — the response is still returned.
        Unhandled graph errors are re-raised after marking the run failed.
        """
        run = await create_agent_run(
            session=session,
            user_id=current_user.user_id,
            user_query=request.message,
        )

        await log_trace_event(
            session=session,
            run_id=run.id,
            event_type="graph",
            event_name="graph_start",
            detail_json={"user_query": request.message},
        )

        # Commit the AgentRun row before entering the graph so that a later
        # rollback (on graph failure) does not delete it and cause FK violations
        # when we try to persist the graph_error trace event.
        await session.commit()

        initial_state: AgentGraphState = {
            "run_id": run.id,
            "user_id": current_user.user_id,
            "user_query": request.message,
            "intent": None,
            "rewritten_query": None,
            "rag_result": None,
            "retrieved_destinations": [],
            "selected_destination": None,
            "classifier_result": None,
            "weather_result": None,
            "final_answer": None,
            "webhook_delivered": None,
            "webhook_status_code": None,
            "errors": [],
            "total_cost_usd": None,
            "session": session,
        }

        try:
            final_state: AgentGraphState = await self._graph.ainvoke(initial_state)

            final_answer: str = (
                final_state.get("final_answer")
                or "I was unable to generate a travel plan. Please try again."
            )

            await mark_agent_run_completed(
                session=session,
                run_id=run.id,
                final_answer=final_answer,
                recommended_destination=final_state.get("selected_destination"),
                total_cost_usd=final_state.get("total_cost_usd"),
                webhook_delivered=final_state.get("webhook_delivered"),
                webhook_status_code=final_state.get("webhook_status_code"),
            )

            await session.commit()

            logger.info(
                "agent_service.plan_trip.completed",
                run_id=str(run.id),
                destination=final_state.get("selected_destination"),
                webhook_delivered=final_state.get("webhook_delivered"),
            )

            return PlanTripResponse(
                run_id=run.id,
                answer=final_answer,
                recommended_destination=final_state.get("selected_destination"),
                webhook_delivered=final_state.get("webhook_delivered"),
            )

        except Exception as exc:
            logger.error(
                "agent_service.plan_trip.failed",
                run_id=str(run.id),
                error=str(exc),
                exc_info=True,
            )

            try:
                # Rollback any partial write (e.g. failed flush in mark_agent_run_completed)
                # so the session is clean before writing the failure state.
                await session.rollback()
                await log_trace_event(
                    session=session,
                    run_id=run.id,
                    event_type="graph",
                    event_name="graph_error",
                    detail_json={"error_type": type(exc).__name__},
                )
                await mark_agent_run_failed(
                    session=session,
                    run_id=run.id,
                    error_message=str(exc),
                )
                await session.commit()
            except Exception as persist_exc:
                logger.error(
                    "agent_service.persist_failure_state_failed",
                    run_id=str(run.id),
                    error=str(persist_exc),
                )

            raise
