"""Model router for Haiku (mechanical) and Sonnet (synthesis) LLM calls."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from anthropic import APIError, APITimeoutError, AsyncAnthropic
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.prompts.agent import (
    DESTINATION_NORMALIZATION_PROMPT,
    INTENT_EXTRACTION_PROMPT,
    RAG_QUERY_REWRITING_PROMPT,
    TOOL_ARGUMENT_REPAIR_PROMPT,
)
from app.prompts.synthesis import SYNTHESIS_PROMPT
from app.schemas.agent import TripIntent

# Canonical destination names exactly as stored in destination_profiles and DB.
SUPPORTED_DESTINATIONS: tuple[str, ...] = (
    "Interlaken",
    "Banff",
    "Bali",
    "Santorini",
    "Kyoto",
    "Istanbul",
    "Tbilisi",
    "Kraków",
    "Dubai",
    "Singapore",
)

# Alternative spellings that map to the canonical name.
_DEST_ALIASES: dict[str, str] = {
    "krakow": "Kraków",
    "cracow": "Kraków",
}


def _normalize_destination_response(raw: str, candidates: list[str]) -> str | None:
    """Extract the first valid destination name from raw LLM output.

    Checks candidates (from RAG) first, then the full supported set, then
    aliases.  Returns the canonical properly-cased name, or None if no match.
    """
    raw_lower = raw.lower().strip()
    if not raw_lower or raw_lower in ("null", "none"):
        return None

    # Build lookup: lowercase → canonical name.
    lookup: dict[str, str] = {d.lower(): d for d in SUPPORTED_DESTINATIONS}
    lookup.update(_DEST_ALIASES)
    # Candidate names from RAG take precedence (they may be the exact DB strings).
    for c in candidates:
        lookup[c.lower()] = c

    # 1. Exact match.
    if raw_lower in lookup:
        return lookup[raw_lower]

    # 2. Substring match: candidates first (they are the most relevant).
    for dest in candidates:
        if dest.lower() in raw_lower:
            return lookup.get(dest.lower(), dest)

    # 3. Substring match: full supported set and aliases.
    for dest_lower, dest_proper in lookup.items():
        if dest_lower in raw_lower:
            return dest_proper

    return None

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# Anthropic pricing (USD per 1M tokens) as of May 2026
# Used for cost estimation; can be updated or moved to config if prices change frequently
ANTHROPIC_PRICING = {
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
}


@dataclass
class ModelResult:
    """Result from an LLM call with usage tracking."""

    content: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    latency_ms: int | None = None


class ModelRouter:
    """Routes LLM calls to Haiku (mechanical) or Sonnet (synthesis) with token/cost tracking."""

    def __init__(self, settings: Any = None, client: AsyncAnthropic | None = None) -> None:
        """Initialize model router.

        Args:
            settings: Settings object with model names and timeout. Defaults to get_settings().
            client: AsyncAnthropic client. Defaults to new client from settings.anthropic_api_key.
        """
        self.settings = settings or get_settings()
        self.client = client or AsyncAnthropic(api_key=self.settings.anthropic_api_key)
        logger.info(
            "ModelRouter initialized: Haiku=%s, Sonnet=%s",
            self.settings.haiku_model,
            self.settings.sonnet_model,
        )

    def _estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float | None:
        """Estimate cost in USD based on model and token counts.

        Args:
            model: Model name (e.g., "claude-haiku-4-5-20251001").
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Estimated cost in USD, or None if model pricing is unknown.
        """
        pricing = ANTHROPIC_PRICING.get(model)
        if not pricing:
            logger.warning("Unknown model pricing for %s", model)
            return None
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    async def _call_model(
        self,
        model: str,
        system: str,
        user_message: str,
        session: AsyncSession | None = None,
        run_id: UUID | None = None,
        step_name: str | None = None,
    ) -> ModelResult:
        """Call Anthropic model with retry and cost tracking.

        Args:
            model: Model name (haiku or sonnet).
            system: System prompt.
            user_message: User message.
            session: Optional database session for logging usage.
            run_id: Optional agent run ID for logging.
            step_name: Optional step name for logging (e.g., "intent_extraction").

        Returns:
            ModelResult with content, usage, cost, and latency.
        """
        start_time = datetime.now(UTC)

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=0.5, max=8),
                reraise=True,
            ):
                with attempt:
                    response = await self.client.messages.create(
                        model=model,
                        max_tokens=2048,
                        system=system,
                        messages=[{"role": "user", "content": user_message}],
                        timeout=self.settings.anthropic_timeout_seconds,
                    )
        except (APITimeoutError, APIError) as e:
            logger.error(
                "LLM call failed after retries: model=%s, error=%s",
                model,
                str(e),
            )
            raise

        end_time = datetime.now(UTC)
        latency_ms = int((end_time - start_time).total_seconds() * 1000)

        # Extract usage
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost_usd = self._estimate_cost(model, input_tokens, output_tokens)

        # Get text content
        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        result = ModelResult(
            content=content,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
        )

        # Log to database if session and run_id provided
        if session is not None and run_id is not None and step_name is not None:
            from app.db.repositories.agent_runs import log_llm_usage

            try:
                await log_llm_usage(
                    session=session,
                    run_id=run_id,
                    step_name=step_name,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                    latency_ms=latency_ms,
                )
                await session.flush()
            except Exception as e:
                logger.warning("Failed to log LLM usage: %s", str(e))

        return result

    async def extract_trip_intent(
        self,
        message: str,
        session: AsyncSession | None = None,
        run_id: UUID | None = None,
    ) -> TripIntent:
        """Extract structured trip intent from user message using Haiku.

        Args:
            message: User's travel request.
            session: Optional database session for logging.
            run_id: Optional agent run ID for logging.

        Returns:
            Parsed TripIntent object.
        """
        user_prompt = f"""User message: "{message}"

{INTENT_EXTRACTION_PROMPT}"""

        result = await self._call_model(
            model=self.settings.haiku_model,
            system="You are a travel intent extractor. Return valid JSON only.",
            user_message=user_prompt,
            session=session,
            run_id=run_id,
            step_name="intent_extraction",
        )

        try:
            # Extract JSON from response (may be wrapped in markdown code block)
            json_str = result.content
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
            intent_dict = json.loads(json_str.strip())
            return TripIntent(**intent_dict)
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(
                "Failed to parse trip intent: %s. Returning empty intent.",
                str(e),
            )
            return TripIntent()

    async def rewrite_rag_query(
        self,
        message: str,
        intent: TripIntent,
        session: AsyncSession | None = None,
        run_id: UUID | None = None,
    ) -> str:
        """Rewrite user query for RAG retrieval using Haiku.

        Args:
            message: User's original message.
            intent: Extracted trip intent.
            session: Optional database session for logging.
            run_id: Optional agent run ID for logging.

        Returns:
            Rewritten query optimized for vector search.
        """
        user_prompt = f"""Original user message: "{message}"

Extracted intent:
- Budget: ${intent.budget_usd or 'not specified'}
- Duration: {intent.duration_days or 'not specified'} days
- Style: {intent.preferred_style or 'not specified'}
- Activities: {', '.join(intent.preferred_activities) or 'not specified'}

{RAG_QUERY_REWRITING_PROMPT}"""

        result = await self._call_model(
            model=self.settings.haiku_model,
            system="You are a query rewriting expert. Provide 2-3 optimized search queries, one per line.",
            user_message=user_prompt,
            session=session,
            run_id=run_id,
            step_name="rag_query_rewriting",
        )

        return result.content.strip()

    async def select_candidate_destination(
        self,
        message: str,
        retrieved_destinations: list[str],
        session: AsyncSession | None = None,
        run_id: UUID | None = None,
    ) -> str | None:
        """Select best candidate destination from retrieved list using Haiku.

        Args:
            message: User's travel request.
            retrieved_destinations: List of destination names from RAG.
            session: Optional database session for logging.
            run_id: Optional agent run ID for logging.

        Returns:
            Selected destination name, or None if no good match.
        """
        user_prompt = f"""User request: "{message}"

Retrieved destinations:
{chr(10).join(f"- {d}" for d in retrieved_destinations)}

{DESTINATION_NORMALIZATION_PROMPT}"""

        result = await self._call_model(
            model=self.settings.haiku_model,
            system="You are a destination selector. Return only the destination name or 'null' if no match.",
            user_message=user_prompt,
            session=session,
            run_id=run_id,
            step_name="destination_selection",
        )

        raw = result.content.strip()
        normalized = _normalize_destination_response(raw, retrieved_destinations)
        if normalized is None:
            logger.warning(
                "select_destination: unrecognized response '%s' — returning None",
                raw[:120],
            )
        return normalized

    async def repair_tool_arguments(
        self,
        tool_name: str,
        invalid_payload: dict[str, Any],
        validation_error: str,
        session: AsyncSession | None = None,
        run_id: UUID | None = None,
    ) -> dict[str, Any]:
        """Repair invalid tool arguments using Haiku.

        Args:
            tool_name: Name of the tool with invalid arguments.
            invalid_payload: The invalid arguments dict.
            validation_error: The validation error message.
            session: Optional database session for logging.
            run_id: Optional agent run ID for logging.

        Returns:
            Repaired arguments dict, or original if not repairable.
        """
        user_prompt = f"""Tool: {tool_name}

Invalid payload:
{json.dumps(invalid_payload, indent=2)}

Validation error:
{validation_error}

{TOOL_ARGUMENT_REPAIR_PROMPT}"""

        result = await self._call_model(
            model=self.settings.haiku_model,
            system="You are a tool argument repair specialist. Return valid JSON only.",
            user_message=user_prompt,
            session=session,
            run_id=run_id,
            step_name="tool_argument_repair",
        )

        try:
            json_str = result.content
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
            return json.loads(json_str.strip())  # type: ignore[no-any-return]
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(
                "Failed to repair tool arguments: %s. Returning original payload.",
                str(e),
            )
            return invalid_payload

    async def synthesize_final_answer(
        self,
        user_message: str,
        trip_intent: TripIntent,
        rag_content: str,
        classifier_result: str,
        weather_summary: str,
        session: AsyncSession | None = None,
        run_id: UUID | None = None,
    ) -> str:
        """Synthesize final travel plan using Sonnet.

        Args:
            user_message: Original user request.
            trip_intent: Extracted trip intent.
            rag_content: Knowledge chunks about destination from RAG.
            classifier_result: Destination style match result.
            weather_summary: Current/forecast weather summary.
            session: Optional database session for logging.
            run_id: Optional agent run ID for logging.

        Returns:
            Final user-facing travel plan.
        """
        user_prompt = f"""
**User request**: {user_message}

**User preferences** (extracted):
- Budget: ${trip_intent.budget_usd or 'not specified'}/trip
- Duration: {trip_intent.duration_days or 'not specified'} days
- Preferred style: {trip_intent.preferred_style or 'flexible'}
- Activities: {', '.join(trip_intent.preferred_activities) or 'flexible'}
- Climate: {trip_intent.climate_preference or 'any'}

**Destination knowledge** (from RAG):
{rag_content}

**Destination style match**:
{classifier_result}

**Weather forecast**:
{weather_summary}

---

{SYNTHESIS_PROMPT}"""

        result = await self._call_model(
            model=self.settings.sonnet_model,
            system="You are an expert travel advisor. Synthesize a genuine, personalized travel plan.",
            user_message=user_prompt,
            session=session,
            run_id=run_id,
            step_name="final_synthesis",
        )

        return result.content.strip()
