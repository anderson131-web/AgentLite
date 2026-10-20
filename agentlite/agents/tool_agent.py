"""A single tool-calling agent: ask the model, run any tools it requests, repeat.

This is the classic "ReAct-style" loop. In Phase 2 it becomes the Executor that
runs one plan step at a time.

Small local models (3B) make predictable mistakes, so the loop handles them
explicitly instead of crashing:
- Tool call written as JSON text instead of a real tool call -> parse it anyway.
- Unknown tool name or bad arguments -> send the error back as the tool result,
  with the correct schema, so the model can fix it on the next turn.
- Empty reply -> one corrective nudge.
- Too many consecutive tool errors or iterations -> stop and report clearly.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from agentlite.llm import LLM, LLMError, LLMResponse

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a careful assistant that completes tasks using tools.
Rules:
- Call a tool whenever you need facts you do not already know. Call one tool at a time.
- Base your answer on tool results. Never invent numbers, names, or URLs.
- If a tool returns an ERROR, read it, fix your arguments, and try again or use another approach.
- When you have enough information, reply with the final answer as plain text and do not call a tool."""

MAX_CONSECUTIVE_TOOL_ERRORS = 3


@dataclass
class ToolCallRecord:
    name: str
    args: dict[str, Any]
    output: str
    error: bool
    latency_ms: int


@dataclass
class AgentResult:
    answer: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    error: str | None = None  # set when the agent gave up; `answer` then explains why


def run_tool_agent(
    task: str,
    llm: LLM,
    tools: Sequence[BaseTool],
    max_iterations: int = 6,
    system_prompt: str = SYSTEM_PROMPT,
) -> AgentResult:
    """Run the tool loop until the model answers in plain text or we hit a limit."""
    start = time.perf_counter()
    tools_by_name = {t.name: t for t in tools}
    messages: list[BaseMessage] = [SystemMessage(system_prompt), HumanMessage(task)]
    result = AgentResult(answer="")
    consecutive_errors = 0
    nudged_empty = False

    def finish(answer: str, error: str | None = None) -> AgentResult:
        result.answer = answer
        result.error = error
        result.latency_ms = int((time.perf_counter() - start) * 1000)
        return result

    try:
        for _ in range(max_iterations):
            response = _call(llm, messages, tools, result)
            calls = response.tool_calls or _parse_text_tool_call(response.content, tools_by_name)

            if not calls:
                answer = _strip_thinking(response.content)
                if answer:
                    return finish(answer)
                if nudged_empty:
                    return finish("The model returned an empty answer.", error="empty_answer")
                nudged_empty = True
                messages.append(response.message)
                messages.append(HumanMessage("Your reply was empty. Give the final answer as plain text."))
                continue

            # Re-wrap the calls so text-parsed calls look like real ones to the next turn.
            messages.append(AIMessage(content="", tool_calls=calls))
            for call in calls:
                record = _execute_tool_call(call, tools_by_name)
                result.tool_calls.append(record)
                messages.append(ToolMessage(content=record.output, tool_call_id=call["id"]))
                consecutive_errors = consecutive_errors + 1 if record.error else 0

            if consecutive_errors >= MAX_CONSECUTIVE_TOOL_ERRORS:
                return finish(
                    f"Stopped after {consecutive_errors} failed tool calls in a row. "
                    f"Last error: {result.tool_calls[-1].output}",
                    error="tool_errors",
                )

        # Out of iterations: ask for the best answer with the information gathered so far.
        messages.append(HumanMessage("Stop calling tools. Give your best final answer now using the results above."))
        response = _call(llm, messages, None, result)
        answer = _strip_thinking(response.content)
        if answer:
            return finish(answer)
        return finish("Reached the iteration limit without an answer.", error="max_iterations")
    except LLMError as exc:
        return finish(f"LLM error: {exc}", error="llm_error")


def _call(llm: LLM, messages: list[BaseMessage], tools: Sequence[BaseTool] | None, result: AgentResult) -> LLMResponse:
    response = llm.invoke(messages, tools)
    result.llm_calls += 1
    result.input_tokens += response.input_tokens or 0
    result.output_tokens += response.output_tokens or 0
    log.info("LLM call %d: %d ms, %d tool call(s)", result.llm_calls, response.latency_ms, len(response.tool_calls))
    return response


def _execute_tool_call(call: dict[str, Any], tools_by_name: dict[str, BaseTool]) -> ToolCallRecord:
    name, args = call["name"], call.get("args") or {}
    start = time.perf_counter()
    tool = tools_by_name.get(name)

    if tool is None:
        output = f"ERROR: unknown tool '{name}'. Available tools: {', '.join(tools_by_name)}."
        error = True
    else:
        try:
            output = str(tool.invoke(args))
            error = output.startswith("ERROR:")
        except Exception as exc:  # usually argument validation errors from a small model
            schema = json.dumps(tool.args)
            output = f"ERROR: invalid arguments for '{name}': {exc}. Expected arguments: {schema}"
            error = True

    latency_ms = int((time.perf_counter() - start) * 1000)
    log.info("Tool %s(%s) -> %s", name, args, "ERROR" if error else f"{len(output)} chars")
    return ToolCallRecord(name=name, args=args, output=output, error=error, latency_ms=latency_ms)


def _parse_text_tool_call(text: str, tools_by_name: dict[str, BaseTool]) -> list[dict[str, Any]]:
    """Small models sometimes print a tool call as JSON text, e.g.
    {"name": "web_search", "arguments": {"query": "..."}}, instead of using the
    tool-calling format. If the text is clearly that, treat it as a real call."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict) or data.get("name") not in tools_by_name:
        return []
    args = data.get("arguments", data.get("args", data.get("parameters", {})))
    if not isinstance(args, dict):
        return []
    log.info("Recovered a tool call the model wrote as text: %s", data["name"])
    return [{"name": data["name"], "args": args, "id": f"call_{uuid.uuid4().hex[:8]}", "type": "tool_call"}]


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks that reasoning models like qwen3 may emit."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
