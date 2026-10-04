"""LOONG Financial Spotlight over materialized contexts (raw dataset files are not bundled).

Evidence is reviewed source spans when available, otherwise an automatic candidate-line diagnostic.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

from cora_bench.data.base import AnswerScore, DatasetAdapter, Example, GoldEvidence


class LoongAdapter(DatasetAdapter):
    name = "loong"
    has_gold_evidence = True

    def __init__(self, data_dir="cora_bench/outputs/materialized/loong", evidence_dir=None):
        self.dir = Path(data_dir)
        self.evidence_dir = Path(evidence_dir) if evidence_dir else self.dir / "reviewed_evidence"

    def _gold(self, eid, context):
        path = self.evidence_dir / f"{eid}.json"
        if not path.exists():
            return []
        data = json.loads(path.read_text())
        if data.get("status") != "reviewed" or not data.get("reviewer"):
            raise ValueError(f"LOONG annotation is not reviewed: {eid}")
        if data.get("source_sha256") != hashlib.sha256(context.encode()).hexdigest():
            raise ValueError(f"LOONG annotation source changed: {eid}")
        gold = []
        for span in data["support_spans"]:
            if context[span["char_start"] : span["char_end"]] != span["text"]:
                raise ValueError(f"LOONG annotation span mismatch: {eid}")
            gold.append(GoldEvidence(text=span["text"], metadata=span))
        if not gold:
            raise ValueError(f"LOONG annotation has no support: {eid}")
        return gold

    def load(
        self, split: str = "spotlight", subset: str | None = None, limit: int | None = None
    ) -> Iterator[Example]:
        if (self.dir / "input.jsonl").exists():
            rows = [
                json.loads(l)
                for l in (self.dir / "input.jsonl").read_text().splitlines()
                if l.strip()
            ]
            if subset:
                rows = [r for r in rows if r["set"] == int(subset.lower().replace("set", ""))]
            for r in rows[:limit]:
                ctx = (self.dir / "ctx" / f"{r['example_id']}.txt").read_text()
                yield Example(
                    r["example_id"],
                    r["query"],
                    ctx,
                    str(r["answer"]),
                    gold_evidence=self._gold(r["example_id"], ctx),
                    metadata={
                        **{k: v for k, v in r.items() if k not in {"answer", "query"}},
                        "dataset": "loong",
                    },
                )
            return
        raise FileNotFoundError(
            f"Missing materialized LOONG input: {self.dir / 'input.jsonl'}. "
            "See docs/DATA.md for the required source and evidence format."
        )

    def score_answer(self, example: Example, prediction: str) -> AnswerScore:
        """Deterministic value/entity match; LOONG's native LLM judge is scored separately."""
        from cora_bench.evaluation.loong_value_match import spotlight_value_match

        ok = bool(spotlight_value_match(example.answer, prediction))
        return AnswerScore(correct=ok, score=float(ok), metric="spotlight_value_match")

    def evidence_retained(
        self,
        example: Example,
        selected_context: str,
        selected_chunk_ids: set[str],
        chunk_size=512,
        overlap=64,
        source_token_spans=None,
    ) -> dict | None:
        from cora_bench.evaluation.loong_evidence import candidate_line_retention, normalize

        if not example.gold_evidence:
            return candidate_line_retention(example.answer, example.context, selected_context)
        from cora_bench.evaluation.provenance import char_token_span, covers, source_layout

        if source_token_spans is not None:
            _, offsets = source_layout(example.context, example.example_id, chunk_size, overlap)
            retained = [
                covers(
                    source_token_spans,
                    *char_token_span(offsets, g.metadata["char_start"], g.metadata["char_end"]),
                )
                for g in example.gold_evidence
            ]
            metric = "loong_reviewed_support_span_recall"
        else:
            retained = [
                normalize(g.text) in normalize(selected_context) for g in example.gold_evidence
            ]
            metric = "loong_reviewed_support_verbatim_recall"
        return {
            "metric": metric,
            "fraction_retained": sum(retained) / len(retained),
            "all_retained": all(retained),
            "n_support": len(retained),
            "annotation_status": "reviewed",
        }

    def expected_count(self, split: str = "spotlight", subset: str | None = None) -> int | None:
        return 75 if not subset else None
