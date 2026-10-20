"""Command-line entry point.

    python -m agentlite "your task"
    python -m agentlite --model qwen2.5:7b "your task"
    python -m agentlite --list-models
    python -m agentlite --json "your task"      # full final state incl. trace
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from agentlite.config import get_settings
from agentlite.graph import run_task
from agentlite.llm import LLM, RECOMMENDED_OLLAMA_MODELS, LLMError, available_models, check_ollama_model
from agentlite.state import TraceEntry


def print_entry(entry: TraceEntry) -> None:
    """One trace entry as a compact block, printed as soon as the node finishes."""
    tokens = ""
    if entry["input_tokens"] or entry["output_tokens"]:
        tokens = f", {entry['input_tokens'] or 0}/{entry['output_tokens'] or 0} tok"
    print(f"\n--- {entry['agent'].upper()} ({entry['latency_ms']} ms{tokens}) ---")
    if entry["agent"] == "executor":
        step_line = next((l for l in entry["input"].splitlines() if l.startswith("Your current step")), "")
        print(step_line)
    for call in entry["tool_calls"]:
        status = "ERROR" if call["error"] else "ok"
        print(f"  tool: {call['name']}({json.dumps(call['args'])}) [{status}, {call['latency_ms']} ms]")
    if entry["agent"] != "finalize":
        output = entry["output"]
        print(output if len(output) < 600 else output[:600] + " ...")
    if entry["error"]:
        print(f"  ! {entry['error']}")


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(prog="agentlite", description="Plan, execute, and review a task with local LLM agents.")
    parser.add_argument("task", nargs="?", help="The task to run, in quotes.")
    parser.add_argument("--provider", default=settings.provider, help=f"Default: {settings.provider}")
    parser.add_argument("--model", default=settings.model, help=f"Default: {settings.model}")
    parser.add_argument("--list-models", action="store_true", help="Show models available per provider.")
    parser.add_argument("--json", action="store_true", help="Print the final state (with full trace) as JSON.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show every LLM and tool call.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="  [%(name)s] %(message)s")
    for noisy in ("httpx", "httpcore", "primp", "ddgs"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.list_models:
        for provider, models in available_models().items():
            print(f"{provider}:")
            for model in models or ["(none found)"]:
                tag = "  (recommended)" if model in RECOMMENDED_OLLAMA_MODELS else ""
                print(f"  {model}{tag}")
        return 0

    if not args.task:
        parser.error('give a task, e.g. python -m agentlite "What is LangGraph?"')

    try:
        if args.provider == "ollama":
            check_ollama_model(args.model)
        llm = LLM(provider=args.provider, model=args.model)
    except LLMError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        state = run_task(args.task, llm)
        print(json.dumps(state, indent=2, ensure_ascii=False))
        return 0

    print(f"Running with {llm.provider}/{llm.model}")
    state = run_task(args.task, llm, on_trace=print_entry)
    total_ms = sum(e["latency_ms"] for e in state["trace"])
    print("\n=================== FINAL ANSWER ===================")
    print(state["final_answer"])
    print(f"\n[{len(state['plan'])} steps, {state['retries']} retries, {total_ms / 1000:.1f}s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
