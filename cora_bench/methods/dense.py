"""Dense retrieval: rank the shared chunks by Qwen3-Embedding cosine similarity, pack to budget.

torch and sentence-transformers are imported lazily.
"""

from __future__ import annotations

import time

from cora_bench.budgets.token_budget import BudgetEnforcer, Tokenizer
from cora_bench.chunking.token_chunker import TokenChunker
from cora_bench.methods.base import ContextSelector, SelectionResult

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-4B"


class DenseSelector(ContextSelector):
    name = "dense"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        chunk_size: int = 512,
        overlap: int = 64,
        chunk_order: str = "source_order",
        batch_size: int = 32,
        tokenizer: Tokenizer | None = None,
    ):
        self.model_name = model
        self.tok = tokenizer or Tokenizer()
        self.chunker = TokenChunker(chunk_size, overlap, self.tok)
        self.enforcer = BudgetEnforcer(self.tok)
        self.chunk_order = chunk_order
        self.batch_size = batch_size
        self._model = None

    def _m(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            dev = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = SentenceTransformer(
                self.model_name, device=dev, model_kwargs={"torch_dtype": "float16"}
            )
        return self._model

    def _encode_query(self, m, query: str):
        try:  # Qwen3-Embedding ships a 'query' instruct prompt
            return m.encode(
                [query], prompt_name="query", normalize_embeddings=True, convert_to_numpy=True
            )[0]
        except Exception:
            instruct = (
                f"Instruct: Given a question, retrieve passages that help answer it\nQuery: {query}"
            )
            return m.encode([instruct], normalize_embeddings=True, convert_to_numpy=True)[0]

    def select(
        self, query: str, context: str, budget_tokens: int | None, metadata: dict | None = None
    ) -> SelectionResult:
        import hashlib

        t0 = time.perf_counter()
        m = self._m()
        key = (
            hashlib.sha1(context.encode("utf-8")).hexdigest(),
            (metadata or {}).get("example_id"),
        )
        cached = getattr(self, "_cache", None)
        if cached is None:
            cached = self._cache = {}
        compute_start = time.perf_counter()
        hit = key in cached
        if key in cached:
            chunks, doc_emb, src = cached[key]
            gpu_s = 0.0  # chunk embeddings don't depend on the budget
        else:
            chunks = self.chunker.chunk(context, doc_id=(metadata or {}).get("example_id"))
            src = self.tok.count(context)
            doc_emb = m.encode(
                [c.text for c in chunks],
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            cached[key] = (chunks, doc_emb, src)
            gpu_s = time.perf_counter() - compute_start
        query_start = time.perf_counter()
        q_emb = self._encode_query(m, query)
        gpu_s += time.perf_counter() - query_start
        scores = doc_emb @ q_emb
        order = sorted(range(len(chunks)), key=lambda i: (-float(scores[i]), i))
        ranked = [chunks[i] for i in order]
        if budget_tokens is None:
            selected = sorted(chunks, key=lambda c: c.original_order)
            text, clipped = "\n".join(c.text for c in selected), False
        else:
            selected, text, clipped = self.enforcer.pack_chunks(
                ranked, budget_tokens, self.chunk_order
            )
        return SelectionResult(
            text=text,
            selected_token_count=self.tok.count(text),
            source_token_count=src,
            selected_chunk_ids=[c.chunk_id for c in selected],
            latency_seconds=round(time.perf_counter() - t0, 4),
            gpu_seconds=round(gpu_s, 4),
            clipped=clipped,
            metadata={
                "n_chunks": len(chunks),
                "n_selected": len(selected),
                "model": self.model_name,
                "cache_hit": hit,
                "timing_version": "full_select_wall_v2",
                "chunk_size_tokens": self.chunker.chunk_size,
                "chunk_overlap_tokens": self.chunker.overlap,
            },
        )
