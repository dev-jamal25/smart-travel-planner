"""Classifier tool for agent."""
from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from app.schemas.agent_tools import ClassifyDestinationInput
from app.services.destination_profiles import get_destination_profile
from app.tools.base import ToolResult
from app.tracing import tool_call_trace

if TYPE_CHECKING:
    from app.services.classifier_service import ClassifierService

logger = structlog.get_logger(__name__)


@tool_call_trace
async def classify_destination_style(
    tool_input: ClassifyDestinationInput,
    classifier_service: ClassifierService,
) -> ToolResult:
    """Classify a destination by travel style.

    Args:
        tool_input: Destination to classify.
        classifier_service: Classifier service dependency.

    Returns:
        ToolResult with style classification if successful, error otherwise.
    """
    try:
        # Validate input
        tool_input = ClassifyDestinationInput(**tool_input.model_dump())

        logger.info(
            "classifier_tool.classify_start",
            destination=tool_input.destination,
        )

        # Look up destination profile
        profile = get_destination_profile(tool_input.destination)
        if profile is None:
            logger.warning(
                "classifier_tool.unsupported_destination",
                destination=tool_input.destination,
            )
            return ToolResult.fail(
                "classify_destination_style",
                f"Destination '{tool_input.destination}' is not in the supported list. "
                f"Supported: Interlaken, Banff, Bali, Santorini, Kyoto, Istanbul, "
                f"Tbilisi, Kraków, Dubai, Singapore.",
                retryable=False,
            )

        # Call classifier service with profile
        prediction = classifier_service.predict(profile)
        style: str = prediction.travel_style
        confidence: float | None = prediction.confidence

        logger.info(
            "classifier_tool.classify_success",
            destination=tool_input.destination,
            style=style,
            confidence=confidence,
        )

        output = {
            "destination": tool_input.destination,
            "travel_style": style,
            "confidence": confidence,
        }

        return ToolResult.ok("classify_destination_style", output)

    except Exception as exc:
        logger.error(
            "classifier_tool.classify_failure",
            destination=tool_input.destination,
            error=str(exc),
            exc_info=True,
        )
        return ToolResult.fail(
            "classify_destination_style",
            f"Classification failed: {exc}",
            retryable=False,
        )
