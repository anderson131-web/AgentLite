"""Tests for the eval graders and failure classification."""

from __future__ import annotations

from evals.checks import check_contains, check_exact, check_file, normalize
from evals.run_evals import Record, classify_failure, parse_model


def test_normalize_numbers():
    assert normalize("Average: $82,125.00") == "average: 82125"


def test_exact_uses_word_boundaries():
    assert check_exact("There are 14 people.", {"expected": "14"})[0]
    assert not check_exact("There are 140 people.", {"expected": "14"})[0]
    assert check_exact("The average is $82,125.", {"expected": "82125"})[0]


def test_contains_any_of_and_min_matches():
    spec = {"keywords": [{"any_of": ["Ramírez", "tiangolo"]}]}
    assert check_contains("Created by Sebastián Ramírez", spec)[0]
    spec = {"keywords": ["Django", "Flask", "FastAPI"], "min_matches": 2}
    assert check_contains("Django and FastAPI", spec)[0]
    assert not check_contains("Only Django", spec)[0]


def test_contains_does_not_match_inside_words():
    assert not check_contains("Please submit the form", {"keywords": ["MIT"]})[0]


def test_file_check(tmp_path):
    (tmp_path / "out.md").write_text("London 13", encoding="utf-8")
    assert check_file(tmp_path, {"path": "out.md", "keywords": ["London"]})[0]
    ok, detail = check_file(tmp_path, {"path": "missing.md", "keywords": []})
    assert not ok and "not written" in detail


def test_classify_failure_priorities():
    record = Record("t", "sql", "ollama", "m", "exact")
    record.errors = ["planner: bad plan JSON. Fell back to a single-step plan."]
    assert classify_failure(record, "exact") == "planner_bad_json"
    record.errors, record.tool_calls = [], 0
    assert classify_failure(record, "exact") == "no_tool_used"
    record.tool_calls, record.tool_errors = 2, 1
    assert classify_failure(record, "exact") == "wrong_answer_after_tool_errors"


def test_parse_model():
    assert parse_model("ollama/qwen2.5:3b") == ("ollama", "qwen2.5:3b")
    assert parse_model("qwen2.5:3b") == ("ollama", "qwen2.5:3b")
