"""Graders for benchmark answers. Each returns (passed, detail)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agentlite.agents.common import JSONParseError, extract_json
from agentlite.llm import LLM, LLMError


def normalize(text: str) -> str:
    """Lowercase and make numbers comparable: "$82,125.00" -> "82125"."""
    text = text.lower().replace("$", "")
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)
    return re.sub(r"(\d+)\.0+\b", r"\1", text)


def _has(text: str, keyword: str) -> bool:
    # Word boundaries so "14" doesn't match "140" and "MIT" doesn't match "submit".
    return re.search(rf"(?<!\w){re.escape(normalize(keyword))}(?!\w)", text) is not None


def _keyword_hits(text: str, keywords: list[Any]) -> tuple[int, list[str]]:
    hits, missing = 0, []
    for keyword in keywords:
        options = keyword["any_of"] if isinstance(keyword, dict) else [keyword]
        if any(_has(text, option) for option in options):
            hits += 1
        else:
            missing.append(" / ".join(options))
    return hits, missing


def check_exact(answer: str, spec: dict[str, Any]) -> tuple[bool, str]:
    ok = _has(normalize(answer), spec["expected"])
    return ok, "" if ok else f"expected '{spec['expected']}'"


def check_contains(answer: str, spec: dict[str, Any]) -> tuple[bool, str]:
    keywords = spec["keywords"]
    hits, missing = _keyword_hits(normalize(answer), keywords)
    needed = spec.get("min_matches", len(keywords))
    return hits >= needed, "" if hits >= needed else f"missing: {', '.join(missing)}"


def check_file(workspace: Path, spec: dict[str, Any]) -> tuple[bool, str]:
    path = workspace / spec["path"]
    if not path.is_file():
        return False, f"{spec['path']} was not written"
    content = path.read_text(encoding="utf-8", errors="replace")
    if not content.strip():
        return False, f"{spec['path']} is empty"
    return check_contains(content, spec)


JUDGE_PROMPT = """You are a fair grader. Decide whether the ANSWER satisfies the RUBRIC for the TASK.
Judge only what the rubric asks for. Accept common simplifications and different wording;
fail the answer only if it misses a rubric requirement or states something clearly false.
Reply with JSON only: {"pass": true or false, "reason": "one short sentence"}"""


def check_judge(task: str, answer: str, spec: dict[str, Any], judge: LLM) -> tuple[bool, str]:
    prompt = f"TASK: {task}\n\nRUBRIC: {spec['rubric']}\n\nANSWER:\n{answer[:6000]}"
    try:
        reply = judge.invoke([SystemMessage(JUDGE_PROMPT), HumanMessage(prompt)]).content
        verdict = extract_json(reply)
    except (LLMError, JSONParseError) as exc:
        return False, f"judge error: {exc}"
    passed = verdict.get("pass") in (True, "true", "True")
    return passed, str(verdict.get("reason", ""))
