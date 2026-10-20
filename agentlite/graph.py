"""The LangGraph workflow: planner -> executor (once per step) -> reviewer -> finalize.

    START -> planner -> executor --(more steps?)--> executor
                            |
                            +--(all steps done)--> reviewer
    reviewer --(REVISE, retries left: a fix-up step was appended)--> executor
    reviewer --(APPROVED, or out of retries)--> finalize -> END

Both routers use the same rule: "is there a plan step not yet executed?" The
reviewer requests a revision by appending a step, so no extra flags are needed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Any

from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from agentlite.agents.common import trace_entry
from agentlite.agents.executor import make_executor_node
from agentlite.agents.planner import make_planner_node
from agentlite.agents.reviewer import MAX_RETRIES, make_reviewer_node
from agentlite.llm import LLM
from agentlite.state import AgentState, TraceEntry, initial_state
from agentlite.tools import get_tools

# The executor runs once per step, plus reviewer hops; this is a safety net only.
RECURSION_LIMIT = 40


def route_after_executor(state: AgentState) -> str:
    return "executor" if state["current_step"] < len(state["plan"]) else "reviewer"


def route_after_reviewer(state: AgentState) -> str:
    return "executor" if state["current_step"] < len(state["plan"]) else "finalize"


def finalize(state: AgentState) -> dict:
    """No LLM call: the last executed step already wrote the user-facing answer."""
    answer = state["step_results"][-1] if state["step_results"] else "No answer was produced."
    review = state["review"]
    if review and review["verdict"] == "REVISE":
        answer += (
            f"\n\n_Note: the reviewer still had concerns after {MAX_RETRIES} revisions: "
            f"{review['feedback']}_"
        )
    entry = trace_entry(
        agent="finalize",
        input=f"{len(state['step_results'])} step results, {state['retries']} retries",
        output=answer,
        provider="-",
        model="-",
        latency_ms=0,
    )
    return {"final_answer": answer, "trace": [entry]}


def build_graph(llm: LLM, tools: Sequence[BaseTool] | None = None):
    tools = list(tools) if tools is not None else get_tools()
    graph = StateGraph(AgentState)
    graph.add_node("planner", make_planner_node(llm, tools))
    graph.add_node("executor", make_executor_node(llm, tools))
    graph.add_node("reviewer", make_reviewer_node(llm))
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_conditional_edges("executor", route_after_executor, ["executor", "reviewer"])
    graph.add_conditional_edges("reviewer", route_after_reviewer, ["executor", "finalize"])
    graph.add_edge("finalize", END)
    return graph.compile()


def stream_run(
    task: str,
    llm: LLM,
    tools: Sequence[BaseTool] | None = None,
    plan: list[str] | None = None,
) -> Iterator[tuple[str, Any]]:
    """Yield ("trace", TraceEntry) as each node finishes, ("plan", list[str]) whenever
    the plan changes, and finally ("done", AgentState).

    The CLI prints entries live; the API (Phase 4) forwards them over SSE.
    Pass `plan` to skip the LLM planner.
    """
    app = build_graph(llm, tools)
    state: dict[str, Any] = dict(initial_state(task, plan))
    for update in app.stream(state, {"recursion_limit": RECURSION_LIMIT}, stream_mode="updates"):
        for node_update in update.values():
            if "plan" in node_update:
                yield "plan", node_update["plan"]
            for entry in node_update.get("trace", []):
                yield "trace", entry
            # Mirror LangGraph's merge (trace appends, everything else replaces).
            for key, value in node_update.items():
                state[key] = state[key] + value if key == "trace" else value
    yield "done", AgentState(**state)


def run_task(
    task: str,
    llm: LLM,
    tools: Sequence[BaseTool] | None = None,
    on_trace: Callable[[TraceEntry], None] | None = None,
    plan: list[str] | None = None,
) -> AgentState:
    """Run the whole graph and return the final state."""
    for kind, payload in stream_run(task, llm, tools, plan):
        if kind == "trace" and on_trace:
            on_trace(payload)
        elif kind == "done":
            return payload
    raise RuntimeError("graph finished without a final state")
