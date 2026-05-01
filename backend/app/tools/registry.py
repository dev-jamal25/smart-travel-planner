"""Tool registry and allowlist for agent access."""
from __future__ import annotations  # noqa: F401

# Explicit allowlist of tools the agent is permitted to call
ALLOWED_AGENT_TOOLS = {
    "destination_knowledge_retrieval",
    "classify_destination_style",
    "fetch_live_weather",
}


def is_allowed_tool(tool_name: str) -> bool:
    """Check if a tool is in the allowlist.

    Args:
        tool_name: Name of the tool to check.

    Returns:
        True if the tool is allowed, False otherwise.
    """
    return tool_name in ALLOWED_AGENT_TOOLS


def get_allowed_tools() -> set[str]:
    """Return the set of allowed tools."""
    return ALLOWED_AGENT_TOOLS.copy()
