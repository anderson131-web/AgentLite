"""Helpers shared by the agents: JSON extraction, trace entries, text truncation."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from agentlite.state import ToolCallTrace, TraceEntry


class JSONParseError(ValueError):
    pass


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model reply.

    Small models often wrap JSON in ```json fences or add a sentence before it,
    so we take everything from the first '{' to the last '}'.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise JSONParseError("no JSON object found in the reply")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise JSONParseError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise JSONParseError("expected a JSON object")
    return data


def truncate(text: str, limit: int) -> str:
    """Keep prompts inside the small context window."""
    return text if len(text) <= limit else text[:limit] + " ...[truncated]"


def trace_entry(
    agent: str,
    input: str,
    output: str,
    provider: str,
    model: str,
    latency_ms: int,
    tool_calls: list[ToolCallTrace] | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    error: str | None = None,
) -> TraceEntry:
    return TraceEntry(
        agent=agent,
        input=input,
        output=output,
        tool_calls=tool_calls or [],
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        error=error,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
