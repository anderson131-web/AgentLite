"""Runs the agent graph in background worker threads and records it in the store.

Why a thread pool with ONE worker by default: on a single 4 GB GPU, two
concurrent runs just fight over VRAM and both get slower. Runs queue instead.
Set RUN_WORKERS higher for cloud providers that handle concurrency for you.
"""

from __future__ import annotations

import logging
import os
import traceback
from concurrent.futures import ThreadPoolExecutor

from agentlite.graph import stream_run
from agentlite.jobs import build_job_research
from agentlite.llm import LLM
from api import store
from api.schemas import RunRequest

log = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=int(os.getenv("RUN_WORKERS", "1")), thread_name_prefix="run")


def submit_run(run_id: str, request: RunRequest, llm: LLM) -> None:
    _executor.submit(_execute, run_id, request, llm)


def _execute(run_id: str, request: RunRequest, llm: LLM) -> None:
    store.set_running(run_id)
    try:
        task, plan = request.task, None
        if request.mode == "job":
            research = build_job_research(
                posting_url=request.job_posting_url,
                posting_text=request.job_posting_text,
                resume_text=request.resume_text,
                folder=f"run_{run_id}",  # per-run folder so parallel runs can't clash
            )
            task, plan = research.task, research.plan

        for kind, payload in stream_run(task, llm, plan=plan):
            if kind == "trace":
                store.add_trace(run_id, dict(payload))
            elif kind == "plan":
                store.set_plan(run_id, payload)
            else:
                store.finish_run(run_id, payload["plan"], payload["final_answer"], payload["retries"])
    except Exception as exc:  # never let a worker thread die silently
        log.error("Run %s failed:\n%s", run_id, traceback.format_exc())
        store.fail_run(run_id, f"{type(exc).__name__}: {exc}")
