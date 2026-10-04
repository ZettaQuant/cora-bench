"""Evidence scoring for both stored and live selections."""

from cora_bench.budgets.token_budget import Tokenizer
from cora_bench.evaluation.provenance import chunk_intervals, merge_intervals, source_layout

SCORING_VERSION = "cora_evidence_v2"
EXTRACTIVE_METHODS = {"bm25", "dense", "dense_rerank"}


def score_selection(
    adapter,
    example,
    text,
    chunk_ids=(),
    *,
    method="",
    metadata=None,
    budget=None,
    chunk_size=512,
    overlap=64,
):
    md = metadata or {}
    if md.get("n_unscored", 0):
        raise ValueError(f"Incomplete selection: {md['n_unscored']} unscored chunks")
    if (
        md.get("chunk_size_tokens", chunk_size) != chunk_size
        or md.get("chunk_overlap_tokens", overlap) != overlap
    ):
        raise ValueError("Scoring chunk policy differs from selection policy")
    prunes = md.get("prunes_within_chunk", False) or method in {"provence", "llmlingua2"}
    spans = None
    ids = list(chunk_ids) if not prunes else []
    if ids:
        if len(ids) != len(set(ids)):
            raise ValueError("Repeated chunk IDs in a selection")
        chunks, _ = source_layout(example.context, example.example_id, chunk_size, overlap)
        by_id = {c.chunk_id: c for c in chunks}
        spans = chunk_intervals(example, ids, chunk_size, overlap)
        selected = [by_id[c] for c in ids]
        renderings = {
            "\n".join(c.text for c in selected),
            "\n".join(c.text for c in sorted(selected, key=lambda c: c.original_order)),
        }
        if text not in renderings:
            raise ValueError("Selected text does not match claimed whole-chunk provenance")
    elif (
        not prunes
        and (method in EXTRACTIVE_METHODS or md.get("extractive_whole_chunks"))
        and not text
    ):
        spans = []
    elif not prunes and method in {"head", "tail", "head_tail", "full_context"}:
        tok = Tokenizer()
        source_ids = tok.encode(example.context)
        n = len(source_ids)
        b = min(n, budget) if budget is not None else n
        if text == example.context:
            spans = [(0, n)]
        elif method == "head":
            spans = [(0, b)]
        elif method == "tail":
            spans = [(n - b, n)] if b else []
        elif method == "head_tail":
            half = b // 2
            spans = [(0, half), (n - (b - half), n)]
        if spans is not None:
            rendered = tok.decode([t for s, e in spans for t in source_ids[s:e]])
            if rendered != text:
                # A safety clip after selection can make the intervals wrong; fall back to verbatim.
                spans = None
            else:
                spans = merge_intervals(spans)
    result = adapter.evidence_retained(
        example, text, set(ids), chunk_size=chunk_size, overlap=overlap, source_token_spans=spans
    )
    if result is not None:
        result = {
            **result,
            "scoring_version": SCORING_VERSION,
            "provenance_mode": "source_spans" if spans is not None else "verbatim",
        }
    return result
