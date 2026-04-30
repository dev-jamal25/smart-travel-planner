"""Shared retry configuration for external API calls."""
from __future__ import annotations

from tenacity import (
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Configuration for retrying external calls
# Used by WeatherService, WebhookService, and future LLM/model router calls

# Standard exponential backoff: 0.5s → 1s → 2s → 4s → 8s (max)
DEFAULT_WAIT = wait_exponential(multiplier=0.5, max=8)

# Try up to 3 times
DEFAULT_STOP = stop_after_attempt(3)

# Only retry on network-level errors (httpx.RequestError, etc)
# Not on 4xx client errors (validation) or 5xx server errors if caught separately
DEFAULT_RETRY = retry_if_exception_type(Exception)  # will be narrowed in specific contexts


def _log_before_sleep(retry_state: RetryCallState) -> None:
    """Log retry attempt details (optional helper for structured logging)."""
    if retry_state.attempt_number > 1:
        if retry_state.outcome is not None:
            exception = retry_state.outcome.exception()
            print(
                f"Retry attempt {retry_state.attempt_number}: "
                f"{type(exception).__name__}: {exception}"
            )


# Future: These will be used by:
# - LLM client retry logic (Anthropic API calls)
# - Model router fallback logic
# - Any additional external service calls
