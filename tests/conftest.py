"""Shared test fixtures. FakeLLM replays scripted replies so graph tests are fast and offline."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from langchain_core.messages import AIMessage, BaseMessage

from agentlite.llm import LLMResponse


class FakeLLM:
    """Stands in for agentlite.llm.LLM. Each invoke() pops the next scripted reply."""

    provider = "fake"
    model = "fake-model"

    def __init__(self, replies: Sequence[str | AIMessage]) -> None:
        self.replies = [AIMessage(r) if isinstance(r, str) else r for r in replies]
        self.prompts: list[list[BaseMessage]] = []

    def invoke(self, messages: Sequence[BaseMessage], tools=None) -> LLMResponse:
        self.prompts.append(list(messages))
        if not self.replies:
            raise AssertionError("FakeLLM ran out of scripted replies")
        return LLMResponse(message=self.replies.pop(0), provider=self.provider, model=self.model,
                           latency_ms=1, input_tokens=10, output_tokens=5)


def tool_call(name: str, **args) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{name}", "type": "tool_call"}])


@pytest.fixture
def fake_llm_factory():
    return FakeLLM
