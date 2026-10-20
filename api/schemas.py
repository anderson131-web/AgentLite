"""Request/response models. FastAPI validates requests against these automatically."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class RunRequest(BaseModel):
    mode: Literal["general", "job"] = "general"
    task: str = Field("", max_length=4000, description="Required in general mode.")
    provider: str | None = None
    model: str | None = None
    # Job Research mode
    job_posting_url: str | None = Field(None, max_length=2000)
    job_posting_text: str | None = Field(None, max_length=30_000)
    resume_text: str | None = Field(None, max_length=30_000, description="If empty, uses knowledge_search over ./docs.")

    @model_validator(mode="after")
    def check_mode_fields(self) -> "RunRequest":
        if self.mode == "general" and not self.task.strip():
            raise ValueError("task is required in general mode")
        if self.mode == "job" and not (self.job_posting_url or self.job_posting_text):
            raise ValueError("job mode needs job_posting_url or job_posting_text")
        return self


class RunCreated(BaseModel):
    run_id: str
    status: str


class Run(BaseModel):
    id: str
    task: str
    mode: str
    provider: str
    model: str
    status: str
    plan: list[str]
    final_answer: str | None
    retries: int | None
    error: str | None
    created_at: str
    finished_at: str | None
    trace: list[dict[str, Any]] = []
    total_latency_ms: int | None = None
    trace_count: int | None = None


class ModelsResponse(BaseModel):
    default: dict[str, str]
    providers: dict[str, list[str]]


class IngestResponse(BaseModel):
    files: int
    chunks: int


class ResumeUploadResponse(BaseModel):
    filename: str
    text: str
