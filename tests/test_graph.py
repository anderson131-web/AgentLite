"""Graph routing tests: planner -> executor(s) -> reviewer -> finalize, with retries."""

from __future__ import annotations

from langchain_core.tools import tool

from agentlite.agents.planner import parse_plan
from agentlite.agents.reviewer import MAX_RETRIES, parse_review
from agentlite.graph import route_after_executor, route_after_reviewer, run_task
from agentlite.state import initial_state
from tests.conftest import FakeLLM, tool_call


@tool
def echo(text: str) -> str:
    """Echo the text back."""
    return f"echo: {text}"


APPROVE = '{"verdict": "APPROVED", "feedback": ""}'
REVISE = '{"verdict": "REVISE", "feedback": "mention the year"}'


def test_happy_path_two_steps_with_tool():
    llm = FakeLLM([
        '{"steps": ["look it up", "write the answer"]}',  # planner
        tool_call("echo", text="hi"),                      # step 1: tool call
        "found hi",                                        # step 1: result
        "Final: hi",                                       # step 2: answer
        APPROVE,                                           # reviewer
    ])
    state = run_task("say hi", llm, [echo])
    assert state["plan"] == ["look it up", "write the answer"]
    assert state["final_answer"] == "Final: hi"
    assert state["retries"] == 0
    agents = [e["agent"] for e in state["trace"]]
    assert agents == ["planner", "executor", "executor", "reviewer", "finalize"]
    assert state["trace"][1]["tool_calls"][0]["output"] == "echo: hi"


def test_planner_retries_once_on_bad_json():
    llm = FakeLLM(["Sure! Here is the plan: step one", '{"steps": ["answer"]}', "done", APPROVE])
    state = run_task("x", llm, [echo])
    assert state["plan"] == ["answer"]
    assert state["trace"][0]["error"] is None


def test_planner_falls_back_to_single_step():
    llm = FakeLLM(["not json", "still not json", "done", APPROVE])
    state = run_task("the task", llm, [echo])
    assert state["plan"] == ["the task"]
    assert "Fell back" in state["trace"][0]["error"]


def test_revise_routes_back_to_executor_then_approves():
    llm = FakeLLM(['{"steps": ["answer"]}', "v1", REVISE, "v2 with year", APPROVE])
    state = run_task("x", llm, [echo])
    assert state["final_answer"] == "v2 with year"
    assert state["retries"] == 1
    assert len(state["plan"]) == 2 and "mention the year" in state["plan"][1]


def test_max_retries_then_finish_with_note():
    replies = ['{"steps": ["answer"]}', "v1"]
    for i in range(MAX_RETRIES):
        replies += [REVISE, f"v{i + 2}"]
    replies.append(REVISE)  # still unhappy after the last retry
    state = run_task("x", FakeLLM(replies), [echo])
    assert state["retries"] == MAX_RETRIES
    assert "reviewer still had concerns" in state["final_answer"]


def test_routers():
    state = initial_state("x")
    state["plan"] = ["a", "b"]
    state["current_step"] = 1
    assert route_after_executor(state) == "executor"
    state["current_step"] = 2
    assert route_after_executor(state) == "reviewer"
    assert route_after_reviewer(state) == "finalize"
    state["plan"].append("fix it")
    assert route_after_reviewer(state) == "executor"


def test_parse_plan_accepts_dict_steps_and_fences():
    plan = parse_plan('```json\n{"steps": [{"step": 1, "description": "search"}, "write"]}\n```')
    assert plan.steps == ["search", "write"]


def test_parse_review_keyword_fallback():
    assert parse_review("Looks good. APPROVED")["verdict"] == "APPROVED"
    assert parse_review("REVISE: missing the cover letter")["verdict"] == "REVISE"
