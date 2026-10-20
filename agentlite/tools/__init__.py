"""Tool registry. Each tool lives in its own module; its docstring is what the LLM sees."""

from __future__ import annotations

from langchain_core.tools import BaseTool

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


def get_tools(names: list[str] | None = None) -> list[BaseTool]:
    """Return all tools, or only the named ones."""
    if names is None:
        return list(ALL_TOOLS)
    by_name = {t.name: t for t in ALL_TOOLS}
    unknown = [n for n in names if n not in by_name]
    if unknown:
        raise ValueError(f"Unknown tools: {unknown}. Available: {sorted(by_name)}")
    return [by_name[n] for n in names]
