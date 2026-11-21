"""LLM layer tests: provider selection and enabling, without network calls."""

from __future__ import annotations

import pytest

from agentlite.config import get_settings
from agentlite.llm import LLM, LLMError, enabled_providers
from agentlite.tools import get_tools


@pytest.fixture
def no_cloud_keys(monkeypatch):
    for name in ("GOOGLE_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT",
                 "AZURE_OPENAI_DEPLOYMENT", "GOOGLE_CLOUD_PROJECT"):
        monkeypatch.setenv(name, "")
    get_settings.cache_clear()


def test_only_ollama_without_keys(no_cloud_keys):
    assert enabled_providers() == ["ollama"]
    with pytest.raises(LLMError, match="not configured"):
        LLM(provider="gemini", model="gemini-3.8-flash")


def test_unknown_provider(no_cloud_keys):
    with pytest.raises(LLMError, match="Unknown provider"):
        LLM(provider="nope", model="x")


def test_gemini_builds_with_key(no_cloud_keys, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    get_settings.cache_clear()
    llm = LLM(provider="gemini", model="gemini-3.8-flash")
    assert type(llm._chat).__name__ == "ChatGoogleGenerativeAI"


def test_azure_builds_with_settings(no_cloud_keys, monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4-1-nano")
    get_settings.cache_clear()
    assert "azure_openai" in enabled_providers()
    llm = LLM(provider="azure_openai", model="gpt-4-1-nano")
    assert type(llm._chat).__name__ == "AzureChatOpenAI"


def test_bigquery_tool_only_with_project(no_cloud_keys, monkeypatch):
    assert "bigquery_query" not in [t.name for t in get_tools()]
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-project")
    get_settings.cache_clear()
    assert "bigquery_query" in [t.name for t in get_tools()]


def test_bigquery_rejects_non_select():
    from agentlite.tools.bigquery_query import bigquery_query

    assert bigquery_query.invoke({"query": "DROP TABLE x"}).startswith("ERROR")
