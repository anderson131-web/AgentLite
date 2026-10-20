"""Re-index ./docs for knowledge_search:  python -m agentlite.ingest"""

from __future__ import annotations

import sys

from agentlite.config import get_settings
from agentlite.rag import ingest_docs


def main() -> int:
    settings = get_settings()
    print(f"Indexing {settings.docs_dir} with {settings.embed_model} ...")
    try:
        stats = ingest_docs()
    except Exception as exc:
        print(f"Error: {exc}\nIs Ollama running, and did you run: ollama pull {settings.embed_model}?", file=sys.stderr)
        return 1
    print(f"Indexed {stats['chunks']} chunks from {stats['files']} files into {settings.chroma_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
