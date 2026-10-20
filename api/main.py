"""AgentLite HTTP API.

Run locally:  uvicorn api.main:app --reload --port 8000
Docs:         http://localhost:8000/docs

All endpoints live under /api so the same server can also serve the built
React app at / (one container, one Cloud Run service).
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, FastAPI, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from agentlite.config import PROJECT_ROOT, get_settings
from agentlite.documents import SUPPORTED_SUFFIXES, load_text
from agentlite.llm import LLM, SUPPORTED_PROVIDERS, LLMError, available_models, check_ollama_model
from agentlite.rag import ingest_docs
from api import runner, store
from api.schemas import IngestResponse, ModelsResponse, ResumeUploadResponse, Run, RunCreated, RunRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

SSE_POLL_S = 0.5
SSE_HEARTBEAT_S = 15  # keeps proxies (and Cloud Run) from closing an idle stream
MAX_UPLOAD_BYTES = 5_000_000
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    store.init_db()
    yield


app = FastAPI(title="AgentLite API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,  # only needed when the Vite dev server (port 5173) calls the API directly
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
api = APIRouter(prefix="/api")


@api.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@api.post("/runs", response_model=RunCreated, status_code=202)
def create_run(request: RunRequest) -> RunCreated:
    """Validate the model choice, queue the run, and return its id immediately."""
    settings = get_settings()
    provider = request.provider or settings.provider
    model = request.model or settings.model
    if provider not in SUPPORTED_PROVIDERS:
        raise HTTPException(400, f"Provider '{provider}' is not enabled. Enabled: {list(SUPPORTED_PROVIDERS)}")
    try:
        if provider == "ollama":
            check_ollama_model(model)
        llm = LLM(provider=provider, model=model)
    except LLMError as exc:
        raise HTTPException(400, str(exc)) from exc

    label = request.task if request.mode == "general" else f"Job research: {request.job_posting_url or 'pasted posting'}"
    run_id = store.create_run(label, request.mode, provider, model)
    runner.submit_run(run_id, request, llm)
    return RunCreated(run_id=run_id, status="queued")


@api.get("/runs", response_model=list[Run])
def list_runs(limit: int = 50) -> list[dict]:
    return store.list_runs(min(limit, 200))


@api.get("/runs/{run_id}", response_model=Run)
def get_run(run_id: str) -> dict:
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(404, "Run not found")
    return run


@api.get("/runs/{run_id}/stream")
async def stream_run(run_id: str, request: Request, last_event_id: str | None = Header(None)) -> StreamingResponse:
    """Server-Sent Events: one `trace` event per agent step, then a `done` event.

    Reconnecting clients send Last-Event-ID (the browser does this automatically)
    and only receive the steps they missed.
    """
    if store.get_run(run_id, include_trace=False) is None:
        raise HTTPException(404, "Run not found")

    async def events() -> AsyncIterator[str]:
        seq = int(last_event_id) if last_event_id and last_event_id.isdigit() else 0
        since_heartbeat = 0.0
        while not await request.is_disconnected():
            # Read status BEFORE entries: the worker writes all entries before
            # marking the run finished, so this order never misses the last step.
            run = await asyncio.to_thread(store.get_run, run_id, False)
            for seq, entry in await asyncio.to_thread(store.get_trace_since, run_id, seq):
                yield f"id: {seq}\nevent: trace\ndata: {json.dumps(entry)}\n\n"
                since_heartbeat = 0
            if run and run["status"] in store.FINISHED:
                yield f"event: done\ndata: {json.dumps(run)}\n\n"
                return
            if since_heartbeat >= SSE_HEARTBEAT_S:
                yield ": heartbeat\n\n"
                since_heartbeat = 0
            await asyncio.sleep(SSE_POLL_S)
            since_heartbeat += SSE_POLL_S

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(events(), media_type="text/event-stream", headers=headers)


@api.get("/models", response_model=ModelsResponse)
def models() -> ModelsResponse:
    settings = get_settings()
    return ModelsResponse(
        default={"provider": settings.provider, "model": settings.model},
        providers=available_models(),
    )


@api.post("/ingest", response_model=IngestResponse)
def ingest() -> dict[str, int]:
    """Re-index ./docs into ChromaDB (takes a few seconds per document)."""
    try:
        return ingest_docs()
    except Exception as exc:
        raise HTTPException(500, f"Ingest failed: {exc}") from exc


@api.post("/uploads/resume", response_model=ResumeUploadResponse)
async def upload_resume(file: UploadFile) -> ResumeUploadResponse:
    """Extract text from an uploaded resume (.pdf/.md/.txt). The file is not kept."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(400, f"Unsupported file type. Use {', '.join(SUPPORTED_SUFFIXES)}")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (max 5 MB)")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"upload{suffix}"
        path.write_bytes(data)
        try:
            text = load_text(path)
        except Exception as exc:
            raise HTTPException(400, f"Could not read the file: {exc}") from exc
    if not text:
        raise HTTPException(400, "No text found in the file (scanned PDFs are not supported).")
    return ResumeUploadResponse(filename=file.filename or "", text=text)


app.include_router(api)

# Serve the built React app (Phase 5) if it exists. Unknown paths return
# index.html so client-side routes like /runs work on refresh.
if FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        file = (FRONTEND_DIST / path).resolve()
        if path and file.is_file() and file.is_relative_to(FRONTEND_DIST.resolve()):
            return FileResponse(file)
        return FileResponse(FRONTEND_DIST / "index.html")
