"""File tools that can only touch the ./workspace sandbox folder.

Small, single-purpose tools (read / write / list) are easier for a 3B model to
call correctly than one tool with a "mode" argument.

Path traversal is blocked by resolving the full path (following "..", symlinks,
and absolute paths) and checking that it is still inside the workspace.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

from agentlite.config import get_settings

MAX_READ_CHARS = 6000
MAX_WRITE_CHARS = 100_000


class SandboxError(ValueError):
    pass


def resolve_in_workspace(path: str) -> Path:
    """Turn a user/model-supplied relative path into a safe absolute path, or raise."""
    workspace = get_settings().workspace_dir.resolve()
    cleaned = path.strip().replace("\\", "/")
    if cleaned.startswith("workspace/"):  # models often include the folder name
        cleaned = cleaned[len("workspace/"):]
    if not cleaned:
        raise SandboxError("path is empty")
    candidate = (workspace / cleaned).resolve()
    if not candidate.is_relative_to(workspace):
        raise SandboxError(f"'{path}' is outside the workspace folder")
    return candidate


@tool(parse_docstring=True)
def read_file(path: str) -> str:
    """Read a text file from the workspace folder.

    Args:
        path: File name relative to the workspace, e.g. "notes.txt".
    """
    try:
        target = resolve_in_workspace(path)
    except SandboxError as exc:
        return f"ERROR: {exc}."
    if not target.is_file():
        return f"ERROR: file '{path}' does not exist. Use list_files to see available files."
    text = target.read_text(encoding="utf-8", errors="replace")
    if len(text) > MAX_READ_CHARS:
        text = text[:MAX_READ_CHARS] + f"\n...[truncated, file has {len(text)} characters]"
    return text


@tool(parse_docstring=True)
def write_file(path: str, content: str) -> str:
    """Write text to a file in the workspace folder, replacing it if it exists.

    Args:
        path: File name relative to the workspace, e.g. "summary.md".
        content: The full text to write.
    """
    try:
        target = resolve_in_workspace(path)
    except SandboxError as exc:
        return f"ERROR: {exc}."
    if len(content) > MAX_WRITE_CHARS:
        return f"ERROR: content is too long ({len(content)} characters, max {MAX_WRITE_CHARS})."
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} characters to {target.relative_to(get_settings().workspace_dir.resolve()).as_posix()}"


@tool
def list_files() -> str:
    """List the files in the workspace folder."""
    workspace = get_settings().workspace_dir.resolve()
    files = sorted(
        p.relative_to(workspace).as_posix()
        for p in workspace.rglob("*")
        if p.is_file() and p.name != ".gitkeep"
    )
    return "\n".join(files) if files else "The workspace is empty."
