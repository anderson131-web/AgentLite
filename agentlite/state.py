"""The shared state that flows through the LangGraph graph.

Each node receives the full state and returns only the keys it changed.
LangGraph merges those updates. `trace` uses an "add" reducer, so a node returns
`{"trace": [entry]}` and LangGraph appends it rather than overwriting the list.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict


class ToolCallTrace(TypedDict):
    name: str
    args: dict[str, Any]
    output: str
    error: bool
    latency_ms: int


class TraceEntry(TypedDict):
    agent: str                   # "planner" | "executor" | "reviewer" | "finalize"
    input: str
    output: str
    tool_calls: list[ToolCallTrace]
    provider: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: int
    error: str | None            # short failure description, None if the node succeeded
    timestamp: str               # ISO 8601, UTC


class Review(TypedDict):
    verdict: Literal["APPROVED", "REVISE"]
    feedback: str


class AgentState(TypedDict):
    task: str
    plan: list[str]
    current_step: int            # index into plan of the next step to execute
    step_results: list[str]      # one entry per executed step, in order
    review: Review | None
    final_answer: str
    trace: Annotated[list[TraceEntry], operator.add]
    retries: int                 # how many times the reviewer sent work back


def initial_state(task: str) -> AgentState:
    return AgentState(
        task=task,
        plan=[],
        current_step=0,
        step_results=[],
        review=None,
        final_answer="",
        trace=[],
        retries=0,
    )
