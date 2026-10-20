"""knowledge_search tool: semantic search over the user's own documents (./docs)."""

from __future__ import annotations

from langchain_core.tools import tool

from agentlite import rag


@tool(parse_docstring=True)
def knowledge_search(query: str) -> str:
    """Search the user's own documents (such as their resume and notes) and return the most relevant passages.

    Args:
        query: What to look for, e.g. "Python and machine learning experience".
    """
    try:
        hits = rag.search(query)
    except Exception as exc:  # e.g. Ollama embedding model unavailable
        return f"ERROR: knowledge search failed: {exc}"
    if not hits:
        return "ERROR: the knowledge base is empty. Run: python -m agentlite.ingest"
    return "\n\n".join(f"[{i}] (from {h.source})\n{h.text}" for i, h in enumerate(hits, start=1))
