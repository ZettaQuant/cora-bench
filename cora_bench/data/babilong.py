"""BABILong adapter over the output of scripts/prepare_babilong.py.

Answers use the BABILong label metric; evidence is the annotated bAbI supporting facts.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from cora_bench.data.babilong_prompts import BABILONG_PROMPTS
from cora_bench.data.base import AnswerScore, DatasetAdapter, Example, GoldEvidence

# Vendored from booydar/babilong @7a6efee, babilong/metrics.py::TASK_LABELS (qa1-5).
TASK_LABELS = {
    "qa1": ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"],
    "qa2": ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"],
    "qa3": ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"],
    "qa4": ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"],
    "qa5": ["Bill", "Fred", "Jeff", "Mary", "apple", "football", "milk"],
}


def _official_compare(target: str, output: str, question: str, task_labels: list[str]) -> bool:
    """BABILong compare_answers (babilong/metrics.py): gold must be the only task label left."""
    out = (
        output.lower()
        .split(".")[0]
        .split("<context>")[0]
        .split("<example>")[0]
        .split("Question")[0]
    )
    target = target.lower()
    labels = {l.lower() for l in task_labels}
    in_q = {l for l in labels if l in question.lower()}
    in_out = {l for l in labels if l in out} - in_q
    if "," in target and len(target) > 3:  # multi-subtarget (qa8 etc.)
        subs = target.split(",")
        return all(t in in_out for t in subs) and len(in_out) == len(subs)
    return target in in_out and len(in_out) == 1


class BabiLongAdapter(DatasetAdapter):
    name = "babilong"
    has_gold_evidence = True

    def __init__(self, data_dir: str = "cora_bench/outputs/materialized/babilong"):
        self.dir = Path(data_dir)

    def load(
        self, split: str = "test", subset: str | None = None, limit: int | None = None
    ) -> Iterator[Example]:
        rows = [
            json.loads(l) for l in (self.dir / "input.jsonl").read_text().splitlines() if l.strip()
        ]
        if subset:  # subset selects a task, e.g. "qa2"
            rows = [r for r in rows if r["task"] == subset]
        if limit:
            rows = rows[:limit]
        for r in rows:
            eid = r["example_id"]
            ctx = (self.dir / "ctx" / f"{eid}.txt").read_text()
            ev = json.loads((self.dir / "evidence" / f"{eid}.json").read_text())
            gold = [
                GoldEvidence(
                    text=sf["text"],
                    metadata={
                        "char_start": sf["char_start"],
                        "char_end": sf["char_end"],
                        "fact_id": sf["fact_id"],
                    },
                )
                for sf in ev["support_facts"]
            ]
            pr = BABILONG_PROMPTS[r["task"]]
            yield Example(
                example_id=eid,
                query=str(r["query"]),
                context=ctx,
                answer=str(r["answer"]),
                gold_evidence=gold,
                metadata={
                    "dataset": "babilong",
                    "task": r["task"],
                    "n_support": r["n_support"],
                    "source_tokens": r["source_tokens"],
                    # upstream per-task instructions; output rules differ across qa1-5
                    "reader_instruction": pr["instruction"] + "\n\n" + pr["examples"],
                    "reader_post_prompt": pr["post_prompt"],
                },
            )

    def score_answer(self, example: Example, prediction: str) -> AnswerScore:
        task = example.metadata.get("task")
        labels = TASK_LABELS.get(task, [])
        ok = bool(_official_compare(example.answer, prediction or "", example.query, labels))
        return AnswerScore(correct=ok, score=float(ok), metric="babilong_official_accuracy")

    def evidence_retained(
        self,
        example: Example,
        selected_context: str,
        selected_chunk_ids: set[str],
        chunk_size: int = 512,
        overlap: int = 64,
        source_token_spans=None,
    ) -> dict | None:
        """Support-fact retention.

        With source spans, the annotated occurrence itself must be covered; a duplicate elsewhere
        does not count. Otherwise the exact sentence must survive verbatim, which is stricter.
        """
        facts = example.gold_evidence
        if not facts:
            return None  # no annotated support
        norm = lambda s: " ".join(s.split())
        if selected_chunk_ids or source_token_spans is not None:
            from cora_bench.evaluation.provenance import (
                char_token_span,
                chunk_intervals,
                covers,
                source_layout,
            )

            _, offs = source_layout(example.context, example.example_id, chunk_size, overlap)
            intervals = (
                source_token_spans
                if source_token_spans is not None
                else chunk_intervals(example, selected_chunk_ids, chunk_size, overlap)
            )
            kept = 0
            for ge in facts:
                cs = (ge.metadata or {}).get("char_start")
                ce = (ge.metadata or {}).get("char_end")
                if cs is None or ce is None or example.context[cs:ce] != ge.text:
                    raise ValueError("Missing or mismatched BABILong gold occurrence")
                kept += int(covers(intervals, *char_token_span(offs, cs, ce)))
            metric = "support_fact_recall_full_span_v2"
        else:  # verbatim survival (compressors)
            ctx_n = norm(selected_context)
            kept = sum(1 for ge in facts if norm(ge.text) in ctx_n)
            metric = "verbatim_support_fact_recall"
        return {
            "metric": metric,
            "fraction_retained": kept / len(facts),
            "all_retained": kept == len(facts),
            "n_support": len(facts),
            "occurrence_aware": source_token_spans is not None or bool(selected_chunk_ids),
        }

    def expected_count(self, split: str = "test", subset: str | None = None) -> int | None:
        return None
