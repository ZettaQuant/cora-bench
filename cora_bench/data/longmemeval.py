"""LongMemEval histories with neutral session IDs and answer-bearing turn spans.

The containment score here is diagnostic; experiment/score_answers.py applies the native judge.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from cora_bench.data.base import AnswerScore, DatasetAdapter, Example, GoldEvidence


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def _selected_intervals(selected_chunk_ids, chunk_size: int, overlap: int):
    """Merge the token intervals of chunk IDs like 'eid::N' into sorted, disjoint ranges."""
    step = chunk_size - overlap
    ivs = []
    for cid in selected_chunk_ids:
        try:
            n = int(str(cid).rsplit("::", 1)[1])
        except (IndexError, ValueError):
            continue
        ivs.append((n * step, n * step + chunk_size))
    ivs.sort()
    merged = []
    for s, e in ivs:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


class LongMemEvalAdapter(DatasetAdapter):
    name = "longmemeval"
    has_gold_evidence = True

    def __init__(self, data_dir: str = "cora_bench/outputs/materialized/longmemeval"):
        self.dir = Path(data_dir)

    def load(
        self, split: str = "test", subset: str | None = None, limit: int | None = None
    ) -> Iterator[Example]:
        rows = [
            json.loads(l) for l in (self.dir / "input.jsonl").read_text().splitlines() if l.strip()
        ]
        if subset:
            rows = [r for r in rows if r["question_type"] == subset]
        if limit:
            rows = rows[:limit]
        import bisect

        from cora_bench.budgets.token_budget import Tokenizer

        tok = Tokenizer()
        for r in rows:
            eid = r["example_id"]
            # Some histories contain CR characters; read_text() would translate them and shift
            # the stored character offsets.
            ctx = (self.dir / "ctx" / f"{eid}.txt").read_bytes().decode("utf-8")
            ev = json.loads((self.dir / "evidence" / f"{eid}.json").read_text())
            # Map char spans to tokens from one full-context encoding so they line up with chunks.
            offs = tok.char_offsets(ctx) if ev.get("answer_turns") else []
            gold = []
            for t in ev["answer_turns"]:
                ts = max(0, bisect.bisect_right(offs, t["char_start"]) - 1)
                te = (
                    max(ts, bisect.bisect_right(offs, max(t["char_end"] - 1, t["char_start"])) - 1)
                    + 1
                )
                gold.append(
                    GoldEvidence(
                        text=t["text"],
                        metadata={
                            "token_start": ts,
                            "token_end": te,
                            "session_neutral": t["session_neutral"],
                            "role": t["role"],
                        },
                    )
                )
            yield Example(
                example_id=eid,
                query=str(r["query"]),
                context=ctx,
                answer=str(r["answer"]),
                gold_evidence=gold,
                metadata={
                    "dataset": "longmemeval",
                    "question_type": r["question_type"],
                    "question_date": r.get("question_date"),
                    "abstention": r["abstention"],
                    "answer_session_neutral": ev.get("answer_session_neutral", []),
                    "source_tokens": r["source_tokens"],
                    "n_evidence_turns": r["n_evidence_turns"],
                    # date matters for temporal reasoning; reader includes it
                    "reader_post_prompt": f"Current date: {r.get('question_date')}.",
                },
            )

    def score_answer(self, example: Example, prediction: str) -> AnswerScore:
        ok = bool(prediction) and _norm(example.answer) in _norm(prediction)
        return AnswerScore(correct=ok, score=float(ok), metric="lme_contains_NONofficial")

    def evidence_retained(
        self,
        example: Example,
        selected_context: str,
        selected_chunk_ids: set[str],
        chunk_size: int = 512,
        overlap: int = 64,
        source_token_spans=None,
    ) -> dict | None:
        if example.metadata.get("abstention") or not example.gold_evidence:
            return None  # abstentions and examples without gold turns are excluded
        turns = example.gold_evidence
        if selected_chunk_ids or source_token_spans is not None:
            from cora_bench.evaluation.provenance import chunk_intervals

            merged = (
                source_token_spans
                if source_token_spans is not None
                else chunk_intervals(example, selected_chunk_ids, chunk_size, overlap)
            )

            def covered(ts, te):
                return any(s <= ts and te <= e for s, e in merged)

            kept = [g for g in turns if covered(g.metadata["token_start"], g.metadata["token_end"])]
            metric = "evidence_turn_recall_span"
        else:  # compressors: verbatim turn survival
            ctx_n = _norm(selected_context)
            kept = [g for g in turns if _norm(g.text) in ctx_n]
            metric = "evidence_turn_recall_verbatim"
        kept_sessions = {g.metadata["session_neutral"] for g in kept}
        # Denominator is sessions with an annotated has_answer turn, not all answer_session_ids:
        # some official answer sessions have no such turn, so full context could not reach 1.0.
        # This differs from LongMemEval's native session-retrieval recall.
        turn_sessions = {g.metadata["session_neutral"] for g in turns}
        return {
            "metric": metric,
            "fraction_retained": len(kept) / len(turns),
            "all_retained": len(kept) == len(turns),
            "n_support": len(turns),
            "session_recall": (len(kept_sessions & turn_sessions) / len(turn_sessions))
            if turn_sessions
            else None,
        }

    def expected_count(self, split: str = "test", subset: str | None = None) -> int | None:
        return 500
