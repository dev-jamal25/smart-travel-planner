"""Controlled LangGraph travel-planning graph.

This is NOT a ReAct agent.  The graph is a deterministic, linear StateGraph
where every node and transition is explicit.  The LLM cannot invent tool names
or call anything outside the three tools on the allowlist.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, cast

from langgraph.graph import END, START, StateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.state import AgentGraphState
from app.db.repositories.agent_runs import log_tool_call, log_trace_event
from app.schemas.agent import TripIntent
from app.schemas.agent_tools import (
    ClassifyDestinationInput,
    DestinationKnowledgeInput,
    LiveWeatherInput,
)
from app.schemas.webhook import WebhookTripPlanRequest
from app.tools.base import ToolResult
from app.tools.classifier_tool import classify_destination_style
from app.tools.rag_tool import destination_knowledge_retrieval
from app.tools.weather_tool import fetch_live_weather

if TYPE_CHECKING:
    from app.agents.model_router import ModelRouter
    from app.services.classifier_service import ClassifierService
    from app.services.rag_service import RagService
    from app.services.weather_service import WeatherService
    from app.services.webhook_service import WebhookService

logger = logging.getLogger(__name__)


def _elapsed_ms(start: float) -> int:
    """Return milliseconds elapsed since ``start`` (from time.monotonic)."""
    return int((time.monotonic() - start) * 1000)


def build_travel_graph(
    model_router: ModelRouter,
    rag_service: RagService,
    classifier_service: ClassifierService,
    weather_service: WeatherService,
    webhook_service: WebhookService,
) -> Any:  # CompiledStateGraph — avoid version-specific import
    """Build and compile the controlled travel-planning LangGraph graph.

    All service dependencies are captured as closures so nodes never call
    internal HTTP endpoints and the LLM cannot reference tools outside the
    explicit allowlist in app/tools/registry.py.
    """

    # ── Node: extract_intent ──────────────────────────────────────────────────

    async def extract_intent(state: AgentGraphState) -> dict[str, Any]:
        """Extract structured TripIntent from the user message (Haiku)."""
        session = cast(AsyncSession, state["session"])
        run_id = state["run_id"]
        start = time.monotonic()

        intent = await model_router.extract_trip_intent(
            message=state["user_query"],
            session=session,
            run_id=run_id,
        )

        await log_trace_event(
            session=session,
            run_id=run_id,
            event_type="llm_call",
            event_name="intent_extracted",
            detail_json={
                "preferred_style": intent.preferred_style,
                "candidate_destination": intent.candidate_destination,
            },
            latency_ms=_elapsed_ms(start),
        )

        return {"intent": intent}

    # ── Node: rewrite_rag_query ───────────────────────────────────────────────

    async def rewrite_rag_query(state: AgentGraphState) -> dict[str, Any]:
        """Rewrite the user query for optimised RAG retrieval (Haiku)."""
        session = cast(AsyncSession, state["session"])
        run_id = state["run_id"]
        intent = state["intent"] or TripIntent()
        start = time.monotonic()

        rewritten = await model_router.rewrite_rag_query(
            message=state["user_query"],
            intent=intent,
            session=session,
            run_id=run_id,
        )

        await log_trace_event(
            session=session,
            run_id=run_id,
            event_type="llm_call",
            event_name="rag_query_rewritten",
            detail_json={"preview": rewritten[:120]},
            latency_ms=_elapsed_ms(start),
        )

        return {"rewritten_query": rewritten}

    # ── Node: retrieve_knowledge ──────────────────────────────────────────────

    async def retrieve_knowledge(state: AgentGraphState) -> dict[str, Any]:
        """Retrieve destination knowledge chunks from the RAG store."""
        session = cast(AsyncSession, state["session"])
        run_id = state["run_id"]

        # Take only the first line of the (potentially multi-line) rewritten query
        raw = state["rewritten_query"] or state["user_query"]
        query = (raw.split("\n")[0].strip()) or state["user_query"]

        tool_input = DestinationKnowledgeInput(query=query, top_k=5)
        start = time.monotonic()

        result = await destination_knowledge_retrieval(
            tool_input=tool_input,
            rag_service=rag_service,
            session=session,
        )
        latency = _elapsed_ms(start)

        chunks: list[dict[str, Any]] = (
            result.output.get("chunks", []) if result.status == "ok" else []
        )
        output_summary = (
            f"{len(chunks)} chunks retrieved"
            if result.status == "ok"
            else (result.error or "error")
        )

        await log_tool_call(
            session=session,
            run_id=run_id,
            tool_name="destination_knowledge_retrieval",
            input_json=tool_input.model_dump(),
            output_summary=output_summary,
            status=result.status,
            latency_ms=latency,
        )

        # Deduplicate destination names from returned chunks
        retrieved: list[str] = []
        seen: set[str] = set()
        for chunk in chunks:
            dest = chunk.get("destination")
            if dest and dest not in seen:
                seen.add(dest)
                retrieved.append(dest)

        await log_trace_event(
            session=session,
            run_id=run_id,
            event_type="tool_call",
            event_name="rag_retrieved",
            detail_json={"num_destinations": len(retrieved), "status": result.status},
            latency_ms=latency,
        )

        errors = state["errors"]
        if result.status == "error":
            errors = [*errors, f"RAG retrieval failed: {result.error}"]

        return {
            "rag_result": result,
            "retrieved_destinations": retrieved,
            "errors": errors,
        }

    # ── Node: select_destination ──────────────────────────────────────────────

    async def select_destination(state: AgentGraphState) -> dict[str, Any]:
        """Select the best destination from RAG candidates (Haiku)."""
        session = cast(AsyncSession, state["session"])
        run_id = state["run_id"]
        retrieved = state["retrieved_destinations"]

        if not retrieved:
            # Fall back to the intent's candidate_destination when RAG found nothing
            fallback = (
                state["intent"].candidate_destination if state["intent"] else None
            )
            await log_trace_event(
                session=session,
                run_id=run_id,
                event_type="decision",
                event_name="destination_selected",
                detail_json={"selected": fallback, "reason": "no_rag_results"},
            )
            return {"selected_destination": fallback}

        start = time.monotonic()
        selected = await model_router.select_candidate_destination(
            message=state["user_query"],
            retrieved_destinations=retrieved,
            session=session,
            run_id=run_id,
        )

        await log_trace_event(
            session=session,
            run_id=run_id,
            event_type="llm_call",
            event_name="destination_selected",
            detail_json={"selected": selected, "candidates": retrieved},
            latency_ms=_elapsed_ms(start),
        )

        return {"selected_destination": selected}

    # ── Node: classify_destination ────────────────────────────────────────────

    async def classify_destination(state: AgentGraphState) -> dict[str, Any]:
        """Classify destination travel style using the ML classifier."""
        session = cast(AsyncSession, state["session"])
        run_id = state["run_id"]
        destination = state["selected_destination"]

        if not destination:
            skip_result = ToolResult.fail(
                "classify_destination_style",
                "No destination selected — classification skipped.",
            )
            return {
                "classifier_result": skip_result,
                "errors": [*state["errors"], "No destination for classification"],
            }

        tool_input = ClassifyDestinationInput(destination=destination)
        start = time.monotonic()

        result = await classify_destination_style(
            tool_input=tool_input,
            classifier_service=classifier_service,
        )
        latency = _elapsed_ms(start)

        if result.status == "ok":
            style = result.output.get("travel_style", "unknown")
            conf = result.output.get("confidence")
            summary = (
                f"style={style} confidence={conf:.0%}" if conf is not None else f"style={style}"
            )
        else:
            summary = result.error or "error"

        await log_tool_call(
            session=session,
            run_id=run_id,
            tool_name="classify_destination_style",
            input_json=tool_input.model_dump(),
            output_summary=summary,
            status=result.status,
            latency_ms=latency,
        )

        await log_trace_event(
            session=session,
            run_id=run_id,
            event_type="tool_call",
            event_name="classification_done",
            detail_json={"destination": destination, "status": result.status},
            latency_ms=latency,
        )

        errors = state["errors"]
        if result.status == "error":
            errors = [*errors, f"Classification failed: {result.error}"]

        return {"classifier_result": result, "errors": errors}

    # ── Node: fetch_weather ───────────────────────────────────────────────────

    async def fetch_weather(state: AgentGraphState) -> dict[str, Any]:
        """Fetch live weather forecast for the selected destination."""
        session = cast(AsyncSession, state["session"])
        run_id = state["run_id"]
        destination = state["selected_destination"]

        if not destination:
            skip_result = ToolResult.fail(
                "fetch_live_weather",
                "No destination selected — weather lookup skipped.",
            )
            return {
                "weather_result": skip_result,
                "errors": [*state["errors"], "No destination for weather lookup"],
            }

        tool_input = LiveWeatherInput(destination=destination)
        start = time.monotonic()

        result = await fetch_live_weather(
            tool_input=tool_input,
            weather_service=weather_service,
        )
        latency = _elapsed_ms(start)

        if result.status == "ok":
            temp = result.output.get("current_temperature_c")
            summary = f"temp={temp}°C" if temp is not None else "fetched"
        else:
            summary = result.error or "error"

        await log_tool_call(
            session=session,
            run_id=run_id,
            tool_name="fetch_live_weather",
            input_json=tool_input.model_dump(),
            output_summary=summary,
            status=result.status,
            latency_ms=latency,
        )

        await log_trace_event(
            session=session,
            run_id=run_id,
            event_type="tool_call",
            event_name="weather_done",
            detail_json={"destination": destination, "status": result.status},
            latency_ms=latency,
        )

        errors = state["errors"]
        if result.status == "error":
            errors = [*errors, f"Weather fetch failed: {result.error}"]

        return {"weather_result": result, "errors": errors}

    # ── Node: synthesize_answer ───────────────────────────────────────────────

    async def synthesize_answer(state: AgentGraphState) -> dict[str, Any]:
        """Synthesize the final travel plan from all tool outputs (Sonnet)."""
        session = cast(AsyncSession, state["session"])
        run_id = state["run_id"]
        intent = state["intent"] or TripIntent()

        # Build RAG content string
        rag_result = state["rag_result"]
        if rag_result and rag_result.status == "ok":
            chunks: list[dict[str, Any]] = rag_result.output.get("chunks", [])
            rag_content = "\n\n".join(
                f"[{c.get('destination', 'Unknown')}] {c.get('chunk_text', '')}"
                for c in chunks
            )
        else:
            rag_content = "Destination knowledge unavailable — data could not be retrieved."

        # Build classifier summary string
        clf_result = state["classifier_result"]
        if clf_result and clf_result.status == "ok":
            style = clf_result.output.get("travel_style", "unknown")
            conf = clf_result.output.get("confidence")
            conf_str = f"{conf:.0%}" if conf is not None else "n/a"
            classifier_summary = f"Travel style: {style} (confidence: {conf_str})"
        else:
            classifier_summary = "Travel style classification unavailable."

        # Build weather summary string
        weather_result = state["weather_result"]
        if weather_result and weather_result.status == "ok":
            dest_name = weather_result.output.get("destination", "")
            temp = weather_result.output.get("current_temperature_c")
            daily: list[dict[str, Any]] = weather_result.output.get("daily", [])
            daily_str = ", ".join(
                f"{d.get('date')}: {d.get('temperature_min_c', '?')}–"
                f"{d.get('temperature_max_c', '?')}°C"
                for d in daily[:3]
            )
            weather_summary = f"{dest_name}: current {temp}°C. Forecast: {daily_str}"
        else:
            weather_summary = (
                "Live weather data unavailable — check local forecasts before travelling."
            )

        start = time.monotonic()
        final_answer = await model_router.synthesize_final_answer(
            user_message=state["user_query"],
            trip_intent=intent,
            rag_content=rag_content,
            classifier_result=classifier_summary,
            weather_summary=weather_summary,
            session=session,
            run_id=run_id,
        )

        await log_trace_event(
            session=session,
            run_id=run_id,
            event_type="llm_call",
            event_name="synthesis_done",
            detail_json={
                "destination": state["selected_destination"],
                "answer_length": len(final_answer),
            },
            latency_ms=_elapsed_ms(start),
        )

        return {"final_answer": final_answer}

    # ── Node: deliver_webhook ─────────────────────────────────────────────────

    async def deliver_webhook(state: AgentGraphState) -> dict[str, Any]:
        """Deliver the trip plan to the configured Discord webhook."""
        session = cast(AsyncSession, state["session"])
        run_id = state["run_id"]

        trip_plan = state["final_answer"] or (
            "Travel plan could not be generated. Please try again."
        )

        webhook_req = WebhookTripPlanRequest(
            user_query=state["user_query"],
            destination=state["selected_destination"],
            trip_plan=trip_plan,
        )

        start = time.monotonic()
        delivery = await webhook_service.send_trip_plan(webhook_req)
        latency = _elapsed_ms(start)

        await log_trace_event(
            session=session,
            run_id=run_id,
            event_type="webhook",
            event_name="webhook_done",
            detail_json={
                "delivered": delivery.delivered,
                "status_code": delivery.status_code,
            },
            latency_ms=latency,
        )

        return {
            "webhook_delivered": delivery.delivered,
            "webhook_status_code": delivery.status_code,
        }

    # ── Assemble and compile ──────────────────────────────────────────────────

    graph: StateGraph[AgentGraphState] = StateGraph(AgentGraphState)

    graph.add_node("extract_intent", extract_intent)
    graph.add_node("rewrite_rag_query", rewrite_rag_query)
    graph.add_node("retrieve_knowledge", retrieve_knowledge)
    graph.add_node("select_destination", select_destination)
    graph.add_node("classify_destination", classify_destination)
    graph.add_node("fetch_weather", fetch_weather)
    graph.add_node("synthesize_answer", synthesize_answer)
    graph.add_node("deliver_webhook", deliver_webhook)

    graph.add_edge(START, "extract_intent")
    graph.add_edge("extract_intent", "rewrite_rag_query")
    graph.add_edge("rewrite_rag_query", "retrieve_knowledge")
    graph.add_edge("retrieve_knowledge", "select_destination")
    graph.add_edge("select_destination", "classify_destination")
    graph.add_edge("classify_destination", "fetch_weather")
    graph.add_edge("fetch_weather", "synthesize_answer")
    graph.add_edge("synthesize_answer", "deliver_webhook")
    graph.add_edge("deliver_webhook", END)

    return graph.compile()
