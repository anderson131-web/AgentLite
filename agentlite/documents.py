"""Load plain text from .md, .txt, and .pdf files (used by ingest and Job Research)."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

SUPPORTED_SUFFIXES = (".md", ".txt", ".pdf")


def load_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    if suffix in (".md", ".txt"):
        return path.read_text(encoding="utf-8", errors="replace").strip()
    raise ValueError(f"Unsupported file type '{suffix}'. Use one of {SUPPORTED_SUFFIXES}.")
