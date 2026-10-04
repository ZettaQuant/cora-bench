"""NoLiMa adapter: one needle per example with minimal lexical overlap with the question.

NoLiMa data is under an Adobe Research non-commercial license.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

from cora_bench.data.base import AnswerScore, DatasetAdapter, Example, GoldEvidence


class NoLiMaAdapter(DatasetAdapter):
    name = "nolima"
    has_gold_evidence = True

    def __init__(self, data_dir: str = "cora_bench/outputs/materialized/nolima"):
        self.dir = Path(data_dir)

    def load(
        self, split: str = "test", subset: str | None = None, limit: int | None = None
    ) -> Iterator[Example]:
        rows = [
            json.loads(l) for l in (self.dir / "input.jsonl").read_text().splitlines() if l.strip()
        ]
        if subset:  # subset selects a hop ('onehop'/'twohop') or reasoning_type
            rows = [r for r in rows if subset in (r["hop"], r["reasoning_type"])]
        if limit:
            rows = rows[:limit]
        for r in rows:
            eid = r["example_id"]
            ctx = (self.dir / "ctx" / f"{eid}.txt").read_text()
            ev = json.loads((self.dir / "evidence" / f"{eid}.json").read_text())
            gold = [
                GoldEvidence(
                    text=ev["needle"],
                    metadata={"char_start": ev["char_start"], "char_end": ev["char_end"]},
                )
            ]
            yield Example(
                example_id=eid,
                query=str(r["query"]),
                context=ctx,
                answer=str(r["answer"]),
                gold_evidence=gold,
                metadata={
                    "dataset": "nolima",
                    "hop": r["hop"],
                    "depth": r["depth"],
                    "reasoning_type": r["reasoning_type"],
                    "source_tokens": r["source_tokens"],
                    "needle_id": r["needle_id"],
                    "test_id": r["test_id"],
                    "haystack": r["haystack"],
                },
            )

    def score_answer(self, example: Example, prediction: str) -> AnswerScore:
        # whole-word match on the character name
        ok = (
            bool(prediction)
            and re.search(rf"\b{re.escape(example.answer.lower())}\b", prediction.lower())
            is not None
        )
        return AnswerScore(correct=ok, score=float(ok), metric="nolima_char_match")

    def evidence_retained(
        self,
        example: Example,
        selected_context: str,
        selected_chunk_ids: set[str],
        chunk_size=512,
        overlap=64,
        source_token_spans=None,
    ) -> dict | None:
        # The needle occurs once, so verbatim survival is unambiguous.
        needle = example.gold_evidence[0].text
        survived = " ".join(needle.split()) in " ".join(selected_context.split())
        return {
            "metric": "verbatim_needle_survival",
            "fraction_retained": float(survived),
            "all_retained": survived,
            "n_support": 1,
        }

    def expected_count(self, split: str = "test", subset: str | None = None) -> int | None:
        return None
