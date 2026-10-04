"""BM25 (rank_bm25 BM25Okapi) over the shared token chunks, packed to the budget.

With no budget, all chunks are returned in source order.
"""

from __future__ import annotations

import re
import time

from rank_bm25 import BM25Okapi

from cora_bench.budgets.token_budget import BudgetEnforcer, Tokenizer
from cora_bench.chunking.token_chunker import TokenChunker
from cora_bench.methods.base import ContextSelector, SelectionResult

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tok(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Selector(ContextSelector):
    name = "bm25"

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 64,
        chunk_order: str = "source_order",
        tokenizer: Tokenizer | None = None,
    ):
        self.tok = tokenizer or Tokenizer()
        self.chunker = TokenChunker(chunk_size, overlap, self.tok)
        self.enforcer = BudgetEnforcer(self.tok)
        self.chunk_order = chunk_order

    def select(
        self, query: str, context: str, budget_tokens: int | None, metadata: dict | None = None
    ) -> SelectionResult:
        t0 = time.perf_counter()
        chunks = self.chunker.chunk(context, doc_id=(metadata or {}).get("example_id"))
        src = self.tok.count(context)
        scores = BM25Okapi([_tok(c.text) for c in chunks]).get_scores(_tok(query))
        order = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))  # deterministic tie-break
        ranked = [chunks[i] for i in order]
        if budget_tokens is None:
            selected = sorted(chunks, key=lambda c: c.original_order)
            text = "\n".join(c.text for c in selected)
            clipped = False
        else:
            selected, text, clipped = self.enforcer.pack_chunks(
                ranked, budget_tokens, self.chunk_order
            )
        return SelectionResult(
            text=text,
            selected_token_count=self.tok.count(text),
            source_token_count=src,
            selected_chunk_ids=[c.chunk_id for c in selected],
            provenance=[
                {"chunk_id": c.chunk_id, "order": c.original_order, "tokens": c.token_count}
                for c in selected
            ],
            latency_seconds=round(time.perf_counter() - t0, 4),
            clipped=clipped,
            metadata={"n_chunks": len(chunks), "n_selected": len(selected), "lib": "rank_bm25"},
        )
