"""Command-line entry point: python -m agentlite "your task"

Phase 1: runs a single tool-calling agent with web_search.
Phase 2 replaces this with the planner -> executor -> reviewer graph.
"""

from __future__ import annotations

import argparse
import logging
import sys

from agentlite.agents.tool_agent import run_tool_agent
from agentlite.config import get_settings
from agentlite.llm import LLM, RECOMMENDED_OLLAMA_MODELS, LLMError, available_models, check_ollama_model
from agentlite.tools import get_tools


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(prog="agentlite", description="Run a task with a tool-using agent.")
    parser.add_argument("task", nargs="?", help="The task to run, in quotes.")
    parser.add_argument("--provider", default=settings.provider, help=f"Default: {settings.provider}")
    parser.add_argument("--model", default=settings.model, help=f"Default: {settings.model}")
    parser.add_argument("--list-models", action="store_true", help="Show models available per provider.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Hide step-by-step progress logs.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="  [%(name)s] %(message)s")
    # Keep third-party HTTP chatter out of the progress log.
    for noisy in ("httpx", "httpcore", "primp", "ddgs"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.list_models:
        for provider, models in available_models().items():
            print(f"{provider}:")
            for model in models or ["(none found - is the server running?)"]:
                tag = "  (recommended)" if model in RECOMMENDED_OLLAMA_MODELS else ""
                print(f"  {model}{tag}")
        return 0

    if not args.task:
        parser.error("give a task, e.g. python -m agentlite \"What is LangGraph?\"")

    try:
        if args.provider == "ollama":
            check_ollama_model(args.model)
        llm = LLM(provider=args.provider, model=args.model)
    except LLMError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Running with {llm.provider}/{llm.model} ...\n")
    result = run_tool_agent(args.task, llm, get_tools())

    print("\n=== Tool calls ===")
    for i, call in enumerate(result.tool_calls, start=1):
        status = "ERROR" if call.error else "ok"
        print(f"{i}. {call.name}({call.args}) [{status}, {call.latency_ms} ms]")
    if not result.tool_calls:
        print("(none)")

    print("\n=== Answer ===")
    print(result.answer)
    print(
        f"\n[{result.llm_calls} LLM calls, {result.input_tokens} in / {result.output_tokens} out tokens, "
        f"{result.latency_ms / 1000:.1f}s total]"
    )
    return 1 if result.error else 0


if __name__ == "__main__":
    sys.exit(main())
