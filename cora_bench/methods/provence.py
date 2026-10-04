"""Provence: query-conditioned sentence pruning over the shared chunks.

Without a budget, every chunk that keeps at least one sentence is returned. With a budget,
pruned chunks are ordered by Provence's relevance score and packed.
"""

from __future__ import annotations

import time

from cora_bench.budgets.token_budget import BudgetEnforcer, Tokenizer
from cora_bench.chunking.token_chunker import TokenChunker
from cora_bench.methods.base import ContextSelector, SelectionResult

DEFAULT_MODEL = "naver/provence-reranker-debertav3-v1"


class ProvenceSelector(ContextSelector):
    name = "provence"
    has_natural_arm = True

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        chunk_size: int = 512,
        overlap: int = 64,
        threshold: float = 0.1,
        tokenizer: Tokenizer | None = None,
    ):
        self.model_name = model
        self.tok = tokenizer or Tokenizer()
        self.chunker = TokenChunker(chunk_size, overlap, self.tok)
        self.enforcer = BudgetEnforcer(self.tok)
        self.threshold = threshold
        self._model = None

    def _m(self):
        if self._model is None:
            from transformers import AutoModel

            self._model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True).eval()
            try:
                self._model = self._model.to("cuda")
            except Exception:
                pass
        return self._model

    def _prune(self, query: str, passage: str) -> tuple[str, float]:
        # Title retention is off because chunks have no real titles. One passage per call: the
        # batched process() path was slower for this model.
        out = self._m().process(query, passage, threshold=self.threshold, always_select_title=False)
        if isinstance(out, dict):
            return out.get("pruned_context", passage), float(out.get("reranking_score", 0.0))
        return passage, 0.0

    def select(
        self, query: str, context: str, budget_tokens: int | None, metadata: dict | None = None
    ) -> SelectionResult:
        import hashlib

        # Pruning depends on the query but not the budget, so the cache key includes the query.
        key = (
            hashlib.sha1((context + "\x00" + query).encode("utf-8")).hexdigest(),
            (metadata or {}).get("example_id"),
        )
        cached = getattr(self, "_cache", None)
        if cached is None:
            cached = self._cache = {}
        t0 = time.perf_counter()
        if key in cached:
            chunks, pruned, src = cached[key]
            gpu_s = 0.0
        else:
            chunks = self.chunker.chunk(context, doc_id=(metadata or {}).get("example_id"))
            src = self.tok.count(context)
            pruned = [
                (c, *self._prune(query, c.text)) for c in chunks
            ]  # (chunk, pruned_text, score)
            cached[key] = (chunks, pruned, src)
            gpu_s = time.perf_counter() - t0
        if budget_tokens is None:
            kept = [(c, pt) for (c, pt, sc) in pruned if pt.strip()]
            kept.sort(key=lambda x: x[0].original_order)
            text, clipped, chosen = "\n".join(pt for _, pt in kept), False, [c for c, _ in kept]
        else:
            ranked = sorted(pruned, key=lambda x: -x[2])
            chosen, parts, used, clipped = [], [], 0, False
            for c, pt, sc in ranked:
                n = self.tok.count(pt)
                if used + n <= budget_tokens:
                    chosen.append(c)
                    parts.append((c.original_order, pt))
                    used += n
                else:
                    clipped = True
                    break
            parts.sort()
            text = "\n".join(pt for _, pt in parts)
        return SelectionResult(
            text=text,
            selected_token_count=self.tok.count(text),
            source_token_count=src,
            selected_chunk_ids=[c.chunk_id for c in chosen],
            latency_seconds=round(time.perf_counter() - t0, 4),
            gpu_seconds=round(gpu_s, 4),
            clipped=clipped,
            metadata={
                "n_chunks": len(chunks),
                "n_selected": len(chosen),
                "model": self.model_name,
                # Chunk ids over-claim what survived pruning, so scoring uses verbatim matching.
                "prunes_within_chunk": True,
            },
        )
