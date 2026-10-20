"""Reviewer agent: checks the answer against the original task.

Returns APPROVED, or REVISE with feedback. On REVISE we append one extra step
to the plan ("rewrite the answer, fixing X") and send the run back to the
executor. That reuses the normal executor path instead of re-running the
whole plan, which matters when every LLM call takes seconds on a 3B model.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError, field_validator

from agentlite.agents.common import JSONParseError, extract_json, trace_entry, truncate
from agentlite.llm import LLM, LLMError
from agentlite.state import AgentState, Review

SYSTEM_PROMPT = """You are the Reviewer. Check whether the ANSWER fully completes the TASK.

Approve if the answer addresses every part of the task and does not look made up.
Ask for a revision only for a real problem: a missing part of the task, an
obvious error, or an empty or off-topic answer. Do not ask for style changes.

Reply with JSON only:
{"verdict": "APPROVED", "feedback": ""}
or
{"verdict": "REVISE", "feedback": "what exactly is missing or wrong"}"""

MAX_RETRIES = 2  # after this many REVISE rounds we finish anyway, with a note
ANSWER_CHARS = 3000
STEP_CHARS = 800


class ReviewModel(BaseModel):
    verdict: Literal["APPROVED", "REVISE"]
    feedback: str = ""

    @field_validator("verdict", mode="before")
    @classmethod
    def normalize(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value


def parse_review(text: str) -> Review:
    """Parse JSON; if that fails, look for the bare keyword. Raises if neither works."""
    try:
        model = ReviewModel.model_validate(extract_json(text))
        return Review(verdict=model.verdict, feedback=model.feedback.strip())
    except (JSONParseError, ValidationError):
        upper = text.upper()
        if re.search(r"\bREVISE\b", upper):
            return Review(verdict="REVISE", feedback=text.strip())
        if re.search(r"\bAPPROVED\b", upper):
            return Review(verdict="APPROVED", feedback="")
        raise


def build_review_prompt(state: AgentState) -> str:
    steps = "\n".join(
        f"{i}. {step}\n   -> {truncate(result, STEP_CHARS)}"
        for i, (step, result) in enumerate(zip(state["plan"], state["step_results"]), start=1)
    )
    answer = state["step_results"][-1] if state["step_results"] else ""
    return f"TASK: {state['task']}\n\nSTEPS TAKEN:\n{steps}\n\nANSWER:\n{truncate(answer, ANSWER_CHARS)}"


def make_reviewer_node(llm: LLM) -> Callable[[AgentState], dict]:
    def reviewer(state: AgentState) -> dict:
        start = time.perf_counter()
        prompt = build_review_prompt(state)
        messages: list[BaseMessage] = [SystemMessage(SYSTEM_PROMPT), HumanMessage(prompt)]
        tokens_in = tokens_out = 0
        review: Review | None = None
        error: str | None = None

        try:
            for _ in range(2):  # first try + one corrective retry
                response = llm.invoke(messages)
                tokens_in += response.input_tokens or 0
                tokens_out += response.output_tokens or 0
                try:
                    review = parse_review(response.content)
                    break
                except (JSONParseError, ValidationError) as exc:
                    error = f"bad review JSON: {exc}"
                    messages += [
                        AIMessage(response.content),
                        HumanMessage('Reply with JSON only: {"verdict": "APPROVED" or "REVISE", "feedback": "..."}'),
                    ]
        except LLMError as exc:
            error = str(exc)

        if review is None:
            # Don't loop forever on a reviewer that can't produce a verdict.
            review = Review(verdict="APPROVED", feedback="")
            error = f"{error}. Reviewer failed; accepted the answer as-is."
        else:
            error = None

        update: dict = {"review": review}
        if review["verdict"] == "REVISE" and state["retries"] < MAX_RETRIES:
            feedback = review["feedback"] or "The answer is incomplete."
            update["plan"] = state["plan"] + [
                f"Rewrite the complete final answer to the task, fixing this reviewer feedback: {feedback}"
            ]
            update["retries"] = state["retries"] + 1

        update["trace"] = [
            trace_entry(
                agent="reviewer",
                input=prompt,
                output=f"{review['verdict']}: {review['feedback']}".rstrip(": "),
                provider=llm.provider,
                model=llm.model,
                latency_ms=int((time.perf_counter() - start) * 1000),
                input_tokens=tokens_in or None,
                output_tokens=tokens_out or None,
                error=error,
            )
        ]
        return update

    return reviewer
