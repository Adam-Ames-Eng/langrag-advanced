"""
A small, dependency-free evaluation harness for the retrieval step.

This is deliberately simple rather than pulling in a full framework like
RAGAS (which needs its own LLM-as-judge calls and API keys) — it answers
the one question a client actually asks in a proposal: "does it find the
right source document for a question?"

For each labeled question in eval_questions.json, it runs the same
retrieval path as app.py and checks whether the expected source file is
among the ones returned. Report: per-question hit/miss + overall hit rate.

Run:
    python eval.py
"""

import json
import pathlib

from app import _retrieve_and_format, _resolve_question

EVAL_FILE = pathlib.Path(__file__).parent / "eval_questions.json"


def load_eval_set() -> list[dict]:
    if not EVAL_FILE.exists():
        raise SystemExit(
            f"{EVAL_FILE.name} not found — create it with entries like:\n"
            '[{"question": "...", "expected_source": "pricing.txt"}]'
        )
    return json.loads(EVAL_FILE.read_text(encoding="utf-8"))


def main() -> None:
    cases = load_eval_set()
    hits = 0

    print(f"Running {len(cases)} evaluation case(s)\n")
    for i, case in enumerate(cases, start=1):
        question = case["question"]
        expected = case["expected_source"]

        standalone_question = _resolve_question(question, [])
        _, sources = _retrieve_and_format(standalone_question)
        hit = expected in sources

        hits += hit
        status = "HIT " if hit else "MISS"
        print(f"[{status}] {i}. {question!r}")
        print(f"        expected: {expected} | retrieved: {sources}\n")

    total = len(cases)
    rate = (hits / total * 100) if total else 0.0
    print(f"Retrieval hit rate: {hits}/{total} ({rate:.0f}%)")


if __name__ == "__main__":
    main()
