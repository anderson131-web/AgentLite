"""Executor agent: runs ONE plan step per graph visit, using tools as needed.

It reuses the Phase 1 tool loop (`run_tool_agent`). The step prompt includes
the overall task and the results of earlier steps, so each step can build on
the last. Every tool call (input, output, latency, error) goes into the trace.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict

from langchain_core.tools import BaseTool

from agentlite.agents.common import trace_entry, truncate
from agentlite.agents.tool_agent import run_tool_agent
from agentlite.llm import LLM
from agentlite.state import AgentState

PREVIOUS_RESULT_CHARS = 1500  # per earlier step, to fit the 8k context


def build_step_prompt(state: AgentState) -> str:
    plan, index = state["plan"], state["current_step"]
    lines = [f"Overall task: {state['task']}", ""]
    if state["step_results"]:
        lines.append("Results of earlier steps:")
        for i, result in enumerate(state["step_results"], start=1):
            lines.append(f"Step {i} result: {truncate(result, PREVIOUS_RESULT_CHARS)}")
        lines.append("")
    lines.append(f"Your current step ({index + 1} of {len(plan)}): {plan[index]}")
    if index == len(plan) - 1:
        lines.append(
            "This is the final step. Your reply is shown to the user as the final answer, "
            "so make it complete and self-contained, using the earlier results."
        )
    else:
        lines.append("Do only this step. Reply with what you found or did, including key facts and URLs.")
    return "\n".join(lines)


def make_executor_node(llm: LLM, tools: Sequence[BaseTool]) -> Callable[[AgentState], dict]:
    def executor(state: AgentState) -> dict:
        prompt = build_step_prompt(state)
        result = run_tool_agent(prompt, llm, tools)

        entry = trace_entry(
            agent="executor",
            input=prompt,
            output=result.answer,
            provider=llm.provider,
            model=llm.model,
            latency_ms=result.latency_ms,
            tool_calls=[asdict(c) for c in result.tool_calls],
            input_tokens=result.input_tokens or None,
            output_tokens=result.output_tokens or None,
            error=result.error,
        )
        return {
            "step_results": state["step_results"] + [result.answer],
            "current_step": state["current_step"] + 1,
            "trace": [entry],
        }

    return executor
