"""Planner agent: turns the task into a short numbered list of concrete steps.

Output is JSON validated by a Pydantic model. If the model's JSON is bad we
retry once, showing it the exact validation error. If that also fails we fall
back to a one-step plan (the task itself) so the run can still produce an
answer, and the failure is recorded in the trace.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, ValidationError, field_validator

from agentlite.agents.common import JSONParseError, extract_json, trace_entry
from agentlite.llm import LLM, LLMError
from agentlite.state import AgentState

MAX_STEPS = 5

SYSTEM_PROMPT = """You are the Planner. Break the user's task into a short list of concrete steps.

Available tools for the executor:
{tools}

Rules:
- Use between 1 and {max_steps} steps. Fewer is better. Simple questions need 1 or 2 steps.
- Each step is one clear action, ideally using one tool.
- The LAST step must write the complete final answer for the user.
- Reply with JSON only, in exactly this format:
{{"steps": ["first step", "second step"]}}

Example task: "Find the population of Canada and save it to canada.txt"
Example reply: {{"steps": ["Search the web for the current population of Canada", "Write the population to canada.txt and tell the user the result"]}}"""


class Plan(BaseModel):
    steps: list[str] = Field(min_length=1, max_length=MAX_STEPS)

    @field_validator("steps", mode="before")
    @classmethod
    def coerce_steps(cls, value: Any) -> Any:
        # Small models sometimes emit [{"step": 1, "description": "..."}] instead of strings.
        if isinstance(value, list):
            coerced = []
            for item in value:
                if isinstance(item, dict):
                    text = next((v for v in item.values() if isinstance(v, str) and v.strip()), "")
                    coerced.append(text)
                else:
                    coerced.append(str(item))
            return [s.strip() for s in coerced if s.strip()]
        return value


def parse_plan(text: str) -> Plan:
    """Raise JSONParseError or ValidationError if the reply isn't a valid plan."""
    return Plan.model_validate(extract_json(text))


def make_planner_node(llm: LLM, tools: Sequence[BaseTool]) -> Callable[[AgentState], dict]:
    tool_list = "\n".join(f"- {t.name}: {t.description.splitlines()[0]}" for t in tools)
    system = SYSTEM_PROMPT.format(tools=tool_list, max_steps=MAX_STEPS)

    def planner(state: AgentState) -> dict:
        if state["plan"]:  # a predefined plan was supplied (e.g. Job Research)
            entry = trace_entry(
                agent="planner",
                input=state["task"],
                output="Using predefined plan:\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(state["plan"], 1)),
                provider="-",
                model="-",
                latency_ms=0,
            )
            return {"current_step": 0, "step_results": [], "trace": [entry]}

        start = time.perf_counter()
        messages: list[BaseMessage] = [SystemMessage(system), HumanMessage(f"Task: {state['task']}")]
        tokens_in = tokens_out = 0
        error: str | None = None
        reply = ""
        plan: list[str] | None = None

        try:
            for attempt in range(2):  # first try + one corrective retry
                response = llm.invoke(messages)
                tokens_in += response.input_tokens or 0
                tokens_out += response.output_tokens or 0
                reply = response.content
                try:
                    plan = parse_plan(reply).steps
                    break
                except (JSONParseError, ValidationError) as exc:
                    error = f"bad plan JSON (attempt {attempt + 1}): {exc}"
                    messages += [
                        AIMessage(reply),
                        HumanMessage(
                            f"That was not valid. Problem: {exc}. Reply with JSON only, like "
                            '{"steps": ["step one", "step two"]}'
                        ),
                    ]
        except LLMError as exc:
            error = str(exc)

        if plan is None:
            plan = [state["task"]]
            error = f"{error}. Fell back to a single-step plan."
        else:
            error = None  # a successful retry is not a failure

        entry = trace_entry(
            agent="planner",
            input=state["task"],
            output="\n".join(f"{i}. {s}" for i, s in enumerate(plan, start=1)),
            provider=llm.provider,
            model=llm.model,
            latency_ms=int((time.perf_counter() - start) * 1000),
            input_tokens=tokens_in or None,
            output_tokens=tokens_out or None,
            error=error,
        )
        return {"plan": plan, "current_step": 0, "step_results": [], "trace": [entry]}

    return planner
