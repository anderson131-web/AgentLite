"""Retrieval-augmented generation (RAG) over the files in ./docs.

Pipeline: load text -> split into overlapping chunks -> embed each chunk with
nomic-embed-text (via Ollama) -> store in a local persistent ChromaDB
collection. A search embeds the query the same way and returns the nearest
chunks.

We compute embeddings ourselves (instead of letting Chroma do it) so the
embedding model is explicit and comes from the same Ollama server as the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import chromadb
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agentlite.config import get_settings
from agentlite.documents import SUPPORTED_SUFFIXES, load_text

COLLECTION = "docs"
CHUNK_SIZE = 800      # characters; a resume section or a paragraph or two
CHUNK_OVERLAP = 100   # so a sentence cut at a boundary still appears whole once

# nomic-embed-text was trained with task prefixes; using them improves retrieval.
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


@dataclass
class SearchHit:
    source: str
    text: str
    distance: float


@lru_cache(maxsize=1)
def _client() -> chromadb.ClientAPI:
    settings = get_settings()
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(settings.chroma_dir))


def _embedder() -> OllamaEmbeddings:
    settings = get_settings()
    return OllamaEmbeddings(model=settings.embed_model, base_url=settings.ollama_base_url)


def ingest_docs() -> dict[str, int]:
    """Rebuild the index from scratch from every supported file in docs_dir.

    Rebuilding (rather than updating) keeps it simple and means deleted files
    disappear from search results. Returns counts for the CLI and API.
    """
    settings = get_settings()
    files = sorted(
        p for p in settings.docs_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict[str, str | int]] = []
    for path in files:
        source = path.relative_to(settings.docs_dir).as_posix()
        for i, chunk in enumerate(splitter.split_text(load_text(path))):
            ids.append(f"{source}#{i}")
            texts.append(chunk)
            metadatas.append({"source": source, "chunk": i})

    client = _client()
    try:
        client.delete_collection(COLLECTION)
    except Exception:  # collection didn't exist yet
        pass
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    if texts:
        embeddings = _embedder().embed_documents([DOC_PREFIX + t for t in texts])
        collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
    return {"files": len(files), "chunks": len(texts)}


def search(query: str, k: int = 4) -> list[SearchHit]:
    """Return the k chunks closest to the query. Empty list if nothing is indexed."""
    try:
        collection = _client().get_collection(COLLECTION)
    except Exception:
        return []
    if collection.count() == 0:
        return []
    vector = _embedder().embed_query(QUERY_PREFIX + query)
    result = collection.query(query_embeddings=[vector], n_results=min(k, collection.count()))
    return [
        SearchHit(source=meta["source"], text=doc, distance=dist)
        for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
    ]
