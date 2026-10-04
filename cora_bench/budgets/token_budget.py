"""Shared tokenizer and budget enforcement; every budget comparison uses this tokenizer."""

from __future__ import annotations

from functools import lru_cache

import tiktoken

from cora_bench.chunking.base import Chunk

DEFAULT_ENCODING = "cl100k_base"


@lru_cache(maxsize=8)
def _enc(name: str):
    return tiktoken.get_encoding(name)


class Tokenizer:
    def __init__(self, encoding: str = DEFAULT_ENCODING):
        self.encoding = encoding
        self._e = _enc(encoding)

    def count(self, text: str) -> int:
        # Some corpora contain literal "<|endoftext|>"-style strings; encode them as plain text
        # instead of letting tiktoken raise.
        return len(self._e.encode(text or "", disallowed_special=()))

    def encode(self, text: str) -> list[int]:
        return self._e.encode(text or "", disallowed_special=())

    def decode(self, tokens: list[int]) -> str:
        return self._e.decode(tokens)

    def char_offsets(self, text: str) -> list[int]:
        """Start char offset of each token, from a single encoding of the full text.

        Counting len(encode(text[:cpos])) instead can be off by one when BPE merges across the cut.
        """
        ids = self._e.encode(text or "", disallowed_special=())
        _, offs = self._e.decode_with_offsets(ids)
        return offs


class BudgetEnforcer:
    def __init__(self, tokenizer: Tokenizer | None = None):
        self.tok = tokenizer or Tokenizer()

    def clip_text(self, text: str, budget_tokens: int) -> tuple[str, bool]:
        """Clip text to the budget.

        Loops because decode(encode(x)[:n]) can re-encode to slightly more than n tokens.
        """
        ids = self.tok.encode(text)
        if len(ids) <= budget_tokens:
            return text, False
        n = budget_tokens
        for _ in range(8):  # usually converges in 1-2 steps
            cand = self.tok.decode(ids[:n])
            over = self.tok.count(cand) - budget_tokens
            if over <= 0 or n <= 0:
                return cand, True
            n -= max(1, over)
        return self.tok.decode(ids[: max(n, 0)]), True

    def pack_chunks(
        self, ranked: list[Chunk], budget_tokens: int, order: str = "source_order"
    ) -> tuple[list[Chunk], str, bool]:
        """Take top-ranked whole chunks until the next one would exceed the budget.

        Returns (chunks in output order, "\\n"-joined text, clipped). `order` is 'source_order'
        or 'rank_order'.
        """

        def render(chs: list[Chunk]) -> tuple[list[Chunk], str]:
            ordered = (
                sorted(chs, key=lambda c: c.original_order) if order == "source_order" else chs
            )
            return ordered, "\n".join(c.text for c in ordered)

        kept: list[Chunk] = []  # rank order
        used = 0
        clipped = False
        for ch in ranked:
            n = self.tok.count(ch.text)
            sep = 1 if kept else 0  # the "\n" joiner
            if used + sep + n <= budget_tokens:
                kept.append(ch)
                used += sep + n
            else:
                clipped = True
                break
        sel, text = render(kept)
        # BPE merges at the joins can push the total a token or two over; drop the lowest-ranked
        # chunks until it fits.
        while kept and self.tok.count(text) > budget_tokens:
            kept.pop()
            clipped = True
            sel, text = render(kept)
        return sel, text, clipped
