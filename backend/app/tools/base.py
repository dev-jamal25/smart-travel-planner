"""Tool execution base types."""
from __future__ import annotations

from typing import Any, Literal


class ToolResult:
    """Result from a tool execution.

    Attributes:
        tool_name: Name of the tool that executed.
        status: "ok" for success, "error" for failure.
        output: Tool output data (if status="ok").
        error: Error message (if status="error").
        retryable: Whether the error is retryable (transient).
    """

    def __init__(
        self,
        tool_name: str,
        status: Literal["ok", "error"],
        output: dict[str, Any] | None = None,
        error: str | None = None,
        retryable: bool = False,
    ) -> None:
        self.tool_name = tool_name
        self.status = status
        self.output = output or {}
        self.error = error
        self.retryable = retryable

    @staticmethod
    def ok(tool_name: str, output: dict[str, Any]) -> ToolResult:
        """Create a successful tool result."""
        return ToolResult(tool_name=tool_name, status="ok", output=output)

    @staticmethod
    def fail(
        tool_name: str, error: str, retryable: bool = False
    ) -> ToolResult:
        """Create a failed tool result."""
        return ToolResult(
            tool_name=tool_name,
            status="error",
            error=error,
            retryable=retryable,
        )

    def __repr__(self) -> str:
        if self.status == "ok":
            return f"ToolResult({self.tool_name}, ok, keys={list(self.output.keys())})"
        return f"ToolResult({self.tool_name}, error: {self.error})"
