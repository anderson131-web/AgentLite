"""API tests with the graph replaced by a scripted FakeLLM (no Ollama needed)."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

import api.main as main
from tests.conftest import FakeLLM


@pytest.fixture
def client(monkeypatch):
    # Accept any model name and hand the runner a FakeLLM instead of Ollama.
    monkeypatch.setattr(main, "check_ollama_model", lambda model: None)
    monkeypatch.setattr(
        main, "LLM",
        lambda provider, model: FakeLLM(['{"steps": ["answer"]}', "The answer is 42.",
                                         '{"verdict": "APPROVED", "feedback": ""}']),
    )
    with TestClient(main.app) as c:
        yield c


def wait_for(client: TestClient, run_id: str, timeout: float = 10) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        run = client.get(f"/api/runs/{run_id}").json()
        if run["status"] in ("completed", "failed"):
            return run
        time.sleep(0.05)
    raise AssertionError("run did not finish")


def test_run_lifecycle_and_history(client):
    response = client.post("/api/runs", json={"task": "What is 6 x 7?", "model": "fake"})
    assert response.status_code == 202
    run = wait_for(client, response.json()["run_id"])
    assert run["status"] == "completed"
    assert run["final_answer"] == "The answer is 42."
    assert run["plan"] == ["answer"]
    assert [e["agent"] for e in run["trace"]] == ["planner", "executor", "reviewer", "finalize"]
    assert client.get("/api/runs").json()[0]["id"] == run["id"]


def test_stream_sends_trace_then_done(client):
    run_id = client.post("/api/runs", json={"task": "x"}).json()["run_id"]
    wait_for(client, run_id)
    body = client.get(f"/api/runs/{run_id}/stream").text
    assert body.count("event: trace") == 4
    assert body.rstrip().splitlines()[-2] == "event: done"


def test_stream_resumes_from_last_event_id(client):
    run_id = client.post("/api/runs", json={"task": "x"}).json()["run_id"]
    wait_for(client, run_id)
    body = client.get(f"/api/runs/{run_id}/stream", headers={"Last-Event-ID": "3"}).text
    assert body.count("event: trace") == 1


def test_validation_errors(client):
    assert client.post("/api/runs", json={"task": ""}).status_code == 422
    assert client.post("/api/runs", json={"mode": "job"}).status_code == 422
    assert client.post("/api/runs", json={"task": "x", "provider": "nope"}).status_code == 400
    assert client.get("/api/runs/missing").status_code == 404


def test_resume_upload(client):
    ok = client.post("/api/uploads/resume", files={"file": ("cv.md", b"# Me\nPython, SQL", "text/markdown")})
    assert ok.status_code == 200 and "Python" in ok.json()["text"]
    bad = client.post("/api/uploads/resume", files={"file": ("cv.exe", b"MZ", "application/octet-stream")})
    assert bad.status_code == 400
