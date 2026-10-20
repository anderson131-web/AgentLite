"""web_search tool: DuckDuckGo search via the free `ddgs` package (no API key).

Convention for all AgentLite tools: they never raise. Failures come back as a
string starting with "ERROR:" so the model can read the problem and try again,
and the trace records exactly what went wrong.
"""

from __future__ import annotations

from ddgs import DDGS
from langchain_core.tools import tool

MAX_RESULTS = 5
SNIPPET_CHARS = 300  # keep results short: small models have small context windows
TIMEOUT_S = 10


@tool(parse_docstring=True)
def web_search(query: str) -> str:
    """Search the web and return the top 5 results, each with a title, URL, and snippet.

    Use this for facts you don't know or that may have changed, such as company
    information, rankings, or statistics.

    Args:
        query: A short search query, for example "most popular Python web frameworks".
    """
    query = query.strip()
    if not query:
        return "ERROR: query is empty. Provide a few search keywords."

    try:
        results = DDGS(timeout=TIMEOUT_S).text(query, max_results=MAX_RESULTS)
    except Exception as exc:  # ddgs raises on rate limits, timeouts, and empty results
        return f"ERROR: web search failed for '{query}': {exc}"

    if not results:
        return f"No results found for '{query}'. Try different keywords."

    lines = []
    for i, result in enumerate(results, start=1):
        snippet = (result.get("body") or "")[:SNIPPET_CHARS]
        lines.append(f"{i}. {result.get('title', '')}\n   URL: {result.get('href', '')}\n   {snippet}")
    return "\n".join(lines)
