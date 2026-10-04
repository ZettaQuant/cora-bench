"""Deterministic token-based chunker shared by all retrieval methods."""

from __future__ import annotations

from cora_bench.budgets.token_budget import Tokenizer
from cora_bench.chunking.base import Chunk, Chunker


class TokenChunker(Chunker):
    def __init__(
        self, chunk_size: int = 512, overlap: int = 64, tokenizer: Tokenizer | None = None
    ):
        if overlap >= chunk_size:
            raise ValueError("overlap must be < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tok = tokenizer or Tokenizer()

    def chunk(self, context: str, doc_id: str | None = None) -> list[Chunk]:
        ids = self.tok.encode(context)
        step = self.chunk_size - self.overlap
        out: list[Chunk] = []
        order = 0
        for start in range(0, max(len(ids), 1), step):
            window = ids[start : start + self.chunk_size]
            if not window:
                break
            cid = f"{doc_id or 'ctx'}::{order}"
            out.append(
                Chunk(
                    chunk_id=cid,
                    text=self.tok.decode(window),
                    original_order=order,
                    token_start=start,
                    token_end=start + len(window),
                    doc_id=doc_id,
                )
            )
            order += 1
            if start + self.chunk_size >= len(ids):
                break
        return out
