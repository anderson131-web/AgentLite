"""Run the benchmark: every task in tasks.yaml against every model.

    python -m evals.run_evals                                   # default local models
    python -m evals.run_evals --models ollama/qwen2.5:3b --tasks sql_top_seller
    python -m evals.run_evals --judge ollama/qwen2.5:7b
    python -m evals.run_evals --models gemini/gemini-3.8-flash --pause 10   # stay under free-tier RPM

Two phases, on purpose:
1. RUN all tasks for one model, then the next model. Ollama keeps one model in
   VRAM at a time, so grouping by model avoids reloading weights between tasks.
2. GRADE everything afterwards. LLM-as-judge tasks all use the judge model in
   one batch, for the same reason.

Everything runs in a sandbox (evals/.sandbox) with its own workspace, demo DB,
and vector index built only from the sample resume, so your own files never
affect the scores and the scores never touch your files.

Outputs in evals/results/: results_<timestamp>.csv, latest.json (read by the
dashboard's Evals page), and latest.md (table + failure analysis for the README).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

EVALS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVALS_DIR.parent
SANDBOX = EVALS_DIR / ".sandbox"
RESULTS_DIR = EVALS_DIR / "results"

# Point AgentLite at the sandbox BEFORE importing anything that reads settings.
os.environ["WORKSPACE_DIR"] = str(SANDBOX / "workspace")
os.environ["DATA_DIR"] = str(SANDBOX / "data")
os.environ["DOCS_DIR"] = str(SANDBOX / "docs")

from agentlite.config import get_settings  # noqa: E402
from agentlite.documents import load_text  # noqa: E402
from agentlite.graph import run_task  # noqa: E402
from agentlite.jobs import build_job_research  # noqa: E402
from agentlite.llm import LLM, LLMError, check_ollama_model  # noqa: E402
from agentlite.rag import ingest_docs  # noqa: E402
from evals.checks import check_contains, check_exact, check_file, check_judge  # noqa: E402
from evals.pricing import estimate_cost  # noqa: E402

DEFAULT_MODELS = ["ollama/qwen2.5:3b", "ollama/qwen2.5:7b", "ollama/llama3.2:3b"]
DEFAULT_JUDGE = "ollama/qwen2.5:7b"


@dataclass
class Record:
    task_id: str
    category: str
    provider: str
    model: str
    check: str
    passed: bool = False
    detail: str = ""
    failure_type: str | None = None
    latency_s: float = 0.0
    steps: int = 0
    retries: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    answer: str = ""
    errors: list[str] = field(default_factory=list)  # node-level errors from the trace


def parse_model(spec: str) -> tuple[str, str]:
    """'ollama/qwen2.5:3b' -> ('ollama', 'qwen2.5:3b'). Provider defaults to ollama."""
    provider, _, model = spec.partition("/")
    return (provider, model) if model else ("ollama", provider)


def prepare_sandbox() -> None:
    shutil.rmtree(SANDBOX, ignore_errors=True)
    (SANDBOX / "workspace").mkdir(parents=True)
    (SANDBOX / "docs").mkdir(parents=True)
    shutil.copy(PROJECT_ROOT / "docs" / "sample_resume.md", SANDBOX / "docs" / "sample_resume.md")
    get_settings.cache_clear()
    stats = ingest_docs()
    print(f"Sandbox ready: indexed {stats['chunks']} chunks from the sample resume.")


def run_one(task: dict[str, Any], llm: LLM) -> Record:
    record = Record(task["id"], task["category"], llm.provider, llm.model, task["check"]["type"])
    workspace = get_settings().workspace_dir
    if task["check"]["type"] == "file":
        (workspace / task["check"]["path"]).unlink(missing_ok=True)  # a stale file must not count

    start = time.perf_counter()
    try:
        prompt, plan = task.get("task", ""), None
        if task.get("mode") == "job":
            research = build_job_research(
                posting_text=load_text(PROJECT_ROOT / task["job_file"]),
                resume_path=PROJECT_ROOT / task["resume"] if task.get("resume") else None,
                folder=f"eval_{task['id']}",
            )
            prompt, plan = research.task, research.plan
            task["task"] = prompt  # the judge needs to see the task text
        state = run_task(prompt, llm, plan=plan)
    except Exception as exc:  # a crash is a result too; keep going
        record.latency_s = round(time.perf_counter() - start, 1)
        record.failure_type, record.detail = "crash", f"{type(exc).__name__}: {exc}"
        return record

    record.latency_s = round(time.perf_counter() - start, 1)
    record.answer = state["final_answer"]
    record.steps = len(state["plan"])
    record.retries = state["retries"]
    for entry in state["trace"]:
        record.tool_calls += len(entry["tool_calls"])
        record.tool_errors += sum(1 for c in entry["tool_calls"] if c["error"])
        record.input_tokens += entry["input_tokens"] or 0
        record.output_tokens += entry["output_tokens"] or 0
        if entry["error"]:
            record.errors.append(f"{entry['agent']}: {entry['error']}")
    record.cost_usd = estimate_cost(llm.provider, llm.model, record.input_tokens, record.output_tokens)
    return record


def grade(record: Record, task: dict[str, Any], judge: LLM | None) -> None:
    if record.failure_type == "crash":
        return
    spec = task["check"]
    kind = spec["type"]
    if kind == "exact":
        record.passed, record.detail = check_exact(record.answer, spec)
    elif kind == "contains":
        record.passed, record.detail = check_contains(record.answer, spec)
    elif kind == "file":
        record.passed, record.detail = check_file(get_settings().workspace_dir, spec)
    elif kind == "judge":
        if judge is None:
            record.passed, record.detail = False, "no judge model available"
        else:
            record.passed, record.detail = check_judge(task["task"], record.answer, spec, judge)
    else:
        raise ValueError(f"Unknown check type {kind}")
    if not record.passed:
        record.failure_type = classify_failure(record, kind)


def classify_failure(record: Record, check_type: str) -> str:
    """Name the most likely root cause, checking the most specific signals first."""
    errors = " ".join(record.errors).lower()
    if "llm error" in errors or "call failed" in errors:
        return "llm_error"
    if "fell back to a single-step plan" in errors:
        return "planner_bad_json"
    if "tool_errors" in errors:
        return "tool_error_loop"
    if "max_iterations" in errors:
        return "max_iterations"
    if "reviewer failed" in errors:
        return "reviewer_bad_json"
    if check_type == "file" and "was not written" in record.detail:
        return "file_not_written"
    if record.tool_calls == 0 and record.category not in ("reasoning",):
        return "no_tool_used"
    if record.tool_errors:
        return "wrong_answer_after_tool_errors"
    return "wrong_answer"


def summarize(records: list[Record]) -> list[dict[str, Any]]:
    summary = []
    for key in dict.fromkeys((r.provider, r.model) for r in records):
        group = [r for r in records if (r.provider, r.model) == key]
        n = len(group)
        total_calls = sum(r.tool_calls for r in group)
        summary.append({
            "provider": key[0],
            "model": key[1],
            "tasks": n,
            "success_rate": round(sum(r.passed for r in group) / n, 3),
            "avg_latency_s": round(sum(r.latency_s for r in group) / n, 1),
            "avg_steps": round(sum(r.steps for r in group) / n, 2),
            "avg_retries": round(sum(r.retries for r in group) / n, 2),
            "tool_error_rate": round(sum(r.tool_errors for r in group) / total_calls, 3) if total_calls else 0.0,
            "est_cost_usd": round(sum(r.cost_usd for r in group), 4),
        })
    return summary


def failure_analysis(records: list[Record]) -> dict[str, list[dict[str, Any]]]:
    analysis = {}
    for key in dict.fromkeys(f"{r.provider}/{r.model}" for r in records):
        counts = Counter(r.failure_type for r in records if f"{r.provider}/{r.model}" == key and not r.passed)
        analysis[key] = [{"type": t, "count": c} for t, c in counts.most_common()]
    return analysis


def markdown_report(summary: list[dict[str, Any]], analysis: dict[str, list[dict[str, Any]]],
                    records: list[Record], generated_at: str) -> str:
    lines = [
        f"_Generated {generated_at} · {len(records)} task runs · hardware: GTX 1650 4 GB, 16 GB RAM_",
        "",
        "| Model | Success | Avg latency | Avg steps | Avg retries | Tool error rate | Est. cost |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in summary:
        lines.append(
            f"| {s['provider']}/{s['model']} | {s['success_rate']:.0%} | {s['avg_latency_s']:.1f} s | "
            f"{s['avg_steps']:.1f} | {s['avg_retries']:.2f} | {s['tool_error_rate']:.0%} | ${s['est_cost_usd']:.4f} |"
        )

    categories = sorted({r.category for r in records})
    lines += ["", "**Success by category**", "",
              "| Model | " + " | ".join(categories) + " |", "|---|" + "---|" * len(categories)]
    for s in summary:
        cells = []
        for category in categories:
            group = [r for r in records if r.model == s["model"] and r.provider == s["provider"] and r.category == category]
            cells.append(f"{sum(r.passed for r in group)}/{len(group)}")
        lines.append(f"| {s['model']} | " + " | ".join(cells) + " |")

    lines += ["", "**Failure analysis** (most common failure types per model)", ""]
    for model, failures in analysis.items():
        text = ", ".join(f"{f['type']} ×{f['count']}" for f in failures) or "no failures"
        lines.append(f"- `{model}`: {text}")
    failed = [r for r in records if not r.passed]
    if failed:
        lines += ["", "<details><summary>Failed runs</summary>", "", "| Task | Model | Failure | Detail |", "|---|---|---|---|"]
        for r in failed:
            detail = r.detail.replace("|", "/").replace("\n", " ")[:120]
            lines.append(f"| {r.task_id} | {r.model} | {r.failure_type} | {detail} |")
        lines += ["", "</details>"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the AgentLite benchmark.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="provider/model, e.g. ollama/qwen2.5:3b")
    parser.add_argument("--tasks", nargs="+", help="Only run these task ids.")
    parser.add_argument("--judge", default=DEFAULT_JUDGE, help="provider/model used for LLM-as-judge checks.")
    parser.add_argument("--pause", type=float, default=0, help="Seconds to wait between tasks (for rate-limited APIs).")
    args = parser.parse_args(argv)

    tasks = yaml.safe_load((EVALS_DIR / "tasks.yaml").read_text(encoding="utf-8"))["tasks"]
    if args.tasks:
        tasks = [t for t in tasks if t["id"] in args.tasks]
    if not tasks:
        parser.error("no matching tasks")

    llms: list[LLM] = []
    for spec in args.models:
        provider, model = parse_model(spec)
        try:
            if provider == "ollama":
                check_ollama_model(model)
            llms.append(LLM(provider=provider, model=model))
        except LLMError as exc:
            print(f"Skipping {spec}: {exc}", file=sys.stderr)
    if not llms:
        print("No usable models.", file=sys.stderr)
        return 1

    prepare_sandbox()

    # Phase 1: run.
    records: list[tuple[Record, dict[str, Any]]] = []
    for llm in llms:
        print(f"\n=== {llm.provider}/{llm.model}: {len(tasks)} tasks ===")
        for i, task in enumerate(tasks, start=1):
            if i > 1 and args.pause and llm.provider != "ollama":
                time.sleep(args.pause)
            task = dict(task)  # run_one may add the job task text
            record = run_one(task, llm)
            records.append((record, task))
            status = "crash" if record.failure_type == "crash" else "done"
            print(f"  [{i:2}/{len(tasks)}] {task['id']:<30} {status:<6} {record.latency_s:6.1f}s  "
                  f"steps={record.steps} tools={record.tool_calls} errors={record.tool_errors}")

    # Phase 2: grade.
    judge: LLM | None = None
    if any(t["check"]["type"] == "judge" for t in tasks):
        provider, model = parse_model(args.judge)
        try:
            if provider == "ollama":
                check_ollama_model(model)
            judge = LLM(provider=provider, model=model)
        except LLMError as exc:
            print(f"Judge unavailable ({exc}); judge tasks will fail.", file=sys.stderr)
    print(f"\nGrading (judge: {args.judge}) ...")
    for record, task in records:
        grade(record, task, judge)

    # Reports.
    results = [r for r, _ in records]
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary = summarize(results)
    analysis = failure_analysis(results)
    RESULTS_DIR.mkdir(exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"results_{stamp}.csv"
    columns = [f for f in Record.__dataclass_fields__ if f not in ("answer", "errors")]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in results:
            writer.writerow({k: v for k, v in asdict(r).items() if k in columns})

    report = {
        "generated_at": generated_at,
        "judge": args.judge,
        "summary": summary,
        "results": [{k: v for k, v in asdict(r).items() if k not in ("answer", "errors")} for r in results],
        "failure_analysis": analysis,
    }
    (RESULTS_DIR / "latest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown = markdown_report(summary, analysis, results, generated_at)
    (RESULTS_DIR / "latest.md").write_text(markdown, encoding="utf-8")
    # Full answers, for reading failures by hand.
    (RESULTS_DIR / f"answers_{stamp}.json").write_text(
        json.dumps([{"task_id": r.task_id, "model": r.model, "passed": r.passed, "detail": r.detail,
                     "answer": r.answer, "errors": r.errors} for r in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + markdown)
    print(f"Saved {csv_path.name}, latest.json, latest.md in {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
