"""BABILong adapter tests; the integration check needs a generated set at $BABILONG_DIR."""

from __future__ import annotations

import os
from pathlib import Path
from unittest import SkipTest

from cora_bench.data.babilong import BabiLongAdapter

DIR = os.environ.get("BABILONG_DIR", "cora_bench/outputs/materialized/babilong")


def test_identity_and_evidence():
    if not (Path(DIR) / "input.jsonl").exists():
        raise SkipTest("Set BABILONG_DIR to run the materialized-data integration check")
    adapter = BabiLongAdapter(DIR)
    examples = list(adapter.load(limit=None))
    assert examples, f"no examples found in {DIR}"
    for e in examples:
        assert e.gold_evidence, f"{e.example_id} has no support facts"
        # full context retains every support fact
        ev = adapter.evidence_retained(e, e.context, set())
        assert ev["fraction_retained"] == 1.0, (e.example_id, ev["fraction_retained"])
        assert ev["all_retained"] is True, e.example_id
        # empty context retains nothing
        ev0 = adapter.evidence_retained(e, "", set())
        assert ev0["fraction_retained"] == 0.0 and ev0["all_retained"] is False
        # the gold answer scores as correct; an unrelated answer does not
        assert adapter.score_answer(e, f"The answer is {e.answer}.").correct is True
        assert adapter.score_answer(e, "definitely-not-the-answer").correct is False
    # support-fact counts match task family (qa1 -> 1, qa2 -> 2, qa3 -> 3)
    by_task = {}
    for e in examples:
        by_task.setdefault(e.metadata["task"], set()).add(len(e.gold_evidence))
    for task, expected in [("qa1", 1), ("qa2", 2), ("qa3", 3)]:
        if task in by_task:
            assert by_task[task] == {expected}, (task, by_task[task])
    print(f"PASS test_babilong ({len(examples)} examples across {sorted(by_task)})")


def test_official_scorer():
    """The gold must be the only task label in the answer."""
    from cora_bench.data.babilong import _official_compare

    L = ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"]
    q = "Where is Mary?"
    assert _official_compare("kitchen", "The answer is kitchen.", q, L) is True
    assert (
        _official_compare("kitchen", "bathroom and kitchen", q, L) is False
    )  # two labels -> reject
    assert _official_compare("kitchen", "kitchen, but actually bathroom", q, L) is False
    assert _official_compare("kitchen", "office", q, L) is False
    print("PASS test_official_scorer")


if __name__ == "__main__":
    test_identity_and_evidence()
    test_official_scorer()
