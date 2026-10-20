"""Run storage in SQLite: one row per run, one row per trace entry.

Trace entries are written the moment a node finishes. The SSE endpoint reads
new entries from here (by sequence number), which keeps the design simple:
- any number of browser tabs can watch the same run,
- a reconnecting client resumes with Last-Event-ID,
- history survives server restarts.

A new connection per call is the simplest thread-safe pattern for SQLite;
WAL mode lets the worker thread write while API requests read.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from collections.abc import Iterator
from typing import Any

from agentlite.config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id           TEXT PRIMARY KEY,
    task         TEXT NOT NULL,
    mode         TEXT NOT NULL,           -- general | job
    provider     TEXT NOT NULL,
    model        TEXT NOT NULL,
    status       TEXT NOT NULL,           -- queued | running | completed | failed
    plan         TEXT,                    -- JSON list
    final_answer TEXT,
    retries      INTEGER DEFAULT 0,
    error        TEXT,
    created_at   TEXT NOT NULL,
    finished_at  TEXT
);
CREATE TABLE IF NOT EXISTS trace_entries (
    run_id TEXT NOT NULL REFERENCES runs(id),
    seq    INTEGER NOT NULL,
    entry  TEXT NOT NULL,                 -- JSON TraceEntry
    PRIMARY KEY (run_id, seq)
);
"""

FINISHED = ("completed", "failed")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    path = get_settings().runs_db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA)
        # Runs that were in progress when the server stopped will never finish.
        conn.execute(
            "UPDATE runs SET status='failed', error='Server restarted during the run', finished_at=? "
            "WHERE status IN ('queued', 'running')",
            (_now(),),
        )


def create_run(task: str, mode: str, provider: str, model: str) -> str:
    run_id = uuid.uuid4().hex[:12]
    with _connect() as conn:
        conn.execute(
            "INSERT INTO runs (id, task, mode, provider, model, status, created_at) VALUES (?, ?, ?, ?, ?, 'queued', ?)",
            (run_id, task, mode, provider, model, _now()),
        )
    return run_id


def set_running(run_id: str) -> None:
    with _connect() as conn:
        conn.execute("UPDATE runs SET status='running' WHERE id=?", (run_id,))


def add_trace(run_id: str, entry: dict[str, Any]) -> int:
    with _connect() as conn:
        seq = conn.execute("SELECT COUNT(*) FROM trace_entries WHERE run_id=?", (run_id,)).fetchone()[0] + 1
        conn.execute("INSERT INTO trace_entries VALUES (?, ?, ?)", (run_id, seq, json.dumps(entry)))
    return seq


def set_plan(run_id: str, plan: list[str]) -> None:
    with _connect() as conn:
        conn.execute("UPDATE runs SET plan=? WHERE id=?", (json.dumps(plan), run_id))


def finish_run(run_id: str, plan: list[str], final_answer: str, retries: int) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE runs SET status='completed', plan=?, final_answer=?, retries=?, finished_at=? WHERE id=?",
            (json.dumps(plan), final_answer, retries, _now(), run_id),
        )


def fail_run(run_id: str, error: str) -> None:
    with _connect() as conn:
        conn.execute("UPDATE runs SET status='failed', error=?, finished_at=? WHERE id=?", (error, _now(), run_id))


def get_run(run_id: str, include_trace: bool = True) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    if row is None:
        return None
    run = _row_to_run(row)
    run["trace"] = [entry for _, entry in get_trace_since(run_id, 0)] if include_trace else []
    return run


def list_runs(limit: int = 50) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """SELECT r.*, COALESCE(SUM(json_extract(t.entry, '$.latency_ms')), 0) AS total_latency_ms,
                      COUNT(t.seq) AS trace_count
               FROM runs r LEFT JOIN trace_entries t ON t.run_id = r.id
               GROUP BY r.id ORDER BY r.created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [_row_to_run(r) for r in rows]


def get_trace_since(run_id: str, after_seq: int) -> list[tuple[int, dict[str, Any]]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT seq, entry FROM trace_entries WHERE run_id=? AND seq>? ORDER BY seq", (run_id, after_seq)
        ).fetchall()
    return [(r["seq"], json.loads(r["entry"])) for r in rows]


def _row_to_run(row: sqlite3.Row) -> dict[str, Any]:
    run = dict(row)
    run["plan"] = json.loads(run["plan"]) if run.get("plan") else []
    return run
