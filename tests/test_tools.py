"""Tool tests. All offline except where noted (network tests are skipped by default)."""

from __future__ import annotations

import os

import pytest

from agentlite.agents.tool_agent import _parse_text_tool_call, run_tool_agent
from agentlite.tools import get_tools
from agentlite.tools.fetch_page import fetch_page, html_to_text, is_public_url
from agentlite.tools.file_io import list_files, read_file, write_file
from agentlite.tools.sql_query import check_query, sql_query
from tests.conftest import FakeLLM, tool_call

network = pytest.mark.skipif(not os.getenv("AGENTLITE_NETWORK_TESTS"), reason="set AGENTLITE_NETWORK_TESTS=1")


# --- sql_query -------------------------------------------------------------------

def test_sql_select_works_and_is_deterministic():
    out = sql_query.invoke({"query": "SELECT COUNT(*) AS n FROM employees"})
    assert out.splitlines() == ["n", "40"]


@pytest.mark.parametrize("query", [
    "DELETE FROM employees",
    "UPDATE employees SET salary = 0",
    "DROP TABLE sales",
    "SELECT 1; DROP TABLE sales",
    "INSERT INTO products VALUES (99, 'x', 'y', 1)",
    "PRAGMA table_info(employees)",
])
def test_sql_rejects_non_select(query):
    assert sql_query.invoke({"query": query}).startswith("ERROR")


def test_sql_authorizer_blocks_writes_hidden_in_cte():
    # Passes the text check (starts with WITH) but the authorizer/read-only mode must stop it.
    out = sql_query.invoke({"query": "WITH x AS (SELECT 1) DELETE FROM employees"})
    assert out.startswith("ERROR")
    assert "40" in sql_query.invoke({"query": "SELECT COUNT(*) FROM employees"})


def test_sql_error_includes_schema_hint():
    out = sql_query.invoke({"query": "SELECT nope FROM employees"})
    assert out.startswith("ERROR") and "employees(" in out


def test_check_query_allows_trailing_semicolon():
    assert check_query("SELECT 1;") is None


# --- file_io -------------------------------------------------------------------

def test_write_then_read_and_list():
    assert "Wrote" in write_file.invoke({"path": "notes/a.txt", "content": "hello"})
    assert read_file.invoke({"path": "notes/a.txt"}) == "hello"
    assert read_file.invoke({"path": "workspace/notes/a.txt"}) == "hello"  # common model habit
    assert "notes/a.txt" in list_files.invoke({})


@pytest.mark.parametrize("path", ["../secret.txt", "..\\..\\x", "/etc/passwd", "C:/Windows/win.ini", "a/../../b"])
def test_path_traversal_blocked(path):
    assert read_file.invoke({"path": path}).startswith("ERROR")
    assert write_file.invoke({"path": path, "content": "x"}).startswith("ERROR")


def test_read_missing_file():
    assert "does not exist" in read_file.invoke({"path": "nope.txt"})


# --- fetch_page ----------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "http://127.0.0.1:8000", "http://localhost", "http://169.254.169.254/latest/meta-data",
    "http://10.0.0.1", "file:///etc/passwd", "ftp://example.com", "not a url",
])
def test_fetch_page_blocks_unsafe_urls(url):
    ok, _ = is_public_url(url)
    assert not ok
    assert fetch_page.invoke({"url": url}).startswith("ERROR")


def test_html_to_text_strips_scripts_and_nav():
    html = "<html><title>T</title><nav>menu</nav><script>x()</script><p>Hello</p><p>World</p></html>"
    text = html_to_text(html)
    assert "Hello" in text and "World" in text
    assert "menu" not in text and "x()" not in text


@network
def test_fetch_page_live():
    assert "Example Domain" in fetch_page.invoke({"url": "https://example.com"})


# --- tool agent recovery ---------------------------------------------------------

def test_agent_recovers_from_bad_args_and_text_tool_call():
    tools = get_tools(["list_files"])
    llm = FakeLLM([
        tool_call("no_such_tool"),                                # unknown tool -> error fed back
        '```json\n{"name": "list_files", "arguments": {}}\n```',  # call written as text -> recovered
        "<think>ok</think>The workspace is empty.",               # final answer, think tags stripped
    ])
    result = run_tool_agent("list files", llm, tools)
    assert result.error is None
    assert result.answer == "The workspace is empty."
    assert [c.error for c in result.tool_calls] == [True, False]
    assert "Available tools" in result.tool_calls[0].output


def test_agent_stops_after_repeated_tool_errors():
    llm = FakeLLM([tool_call("nope")] * 3)
    result = run_tool_agent("x", llm, get_tools(["list_files"]))
    assert result.error == "tool_errors"


def test_parse_text_tool_call_ignores_plain_json_answers():
    tools = {t.name: t for t in get_tools()}
    assert _parse_text_tool_call('{"answer": 42}', tools) == []
