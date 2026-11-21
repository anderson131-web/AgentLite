"""Tool registry. Each tool lives in its own module; its docstring is what the LLM sees."""

from __future__ import annotations

from langchain_core.tools import BaseTool

from agentlite.config import get_settings
from agentlite.tools.bigquery_query import bigquery_query
from agentlite.tools.fetch_page import fetch_page
from agentlite.tools.file_io import list_files, read_file, write_file
from agentlite.tools.knowledge_search import knowledge_search
from agentlite.tools.sql_query import sql_query
from agentlite.tools.web_search import web_search

ALL_TOOLS: list[BaseTool] = [
    web_search,
    fetch_page,
    sql_query,
    read_file,
    write_file,
    list_files,
    knowledge_search,
]


# Tools that need cloud setup, keyed by the check that enables them. Leaving
# them out otherwise keeps the prompt shorter for small local models.
OPTIONAL_TOOLS: list[tuple[BaseTool, str]] = [(bigquery_query, "gcp_project")]


def available_tools() -> list[BaseTool]:
    settings = get_settings()
    return ALL_TOOLS + [t for t, setting in OPTIONAL_TOOLS if getattr(settings, setting)]


def get_tools(names: list[str] | None = None) -> list[BaseTool]:
    """Return all available tools, or only the named ones."""
    if names is None:
        return available_tools()
    by_name = {t.name: t for t in available_tools()}
    unknown = [n for n in names if n not in by_name]
    if unknown:
        raise ValueError(f"Unknown tools: {unknown}. Available: {sorted(by_name)}")
    return [by_name[n] for n in names]
