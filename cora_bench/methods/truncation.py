"""Deterministic truncation baselines: head, tail, and head_tail."""

from __future__ import annotations

from cora_bench.budgets.token_budget import Tokenizer
from cora_bench.methods.base import ContextSelector, SelectionResult


class Truncation(ContextSelector):
    def __init__(self, mode: str = "head_tail", tokenizer: Tokenizer | None = None):
        if mode not in ("head", "tail", "head_tail"):
            raise ValueError(mode)
        self.mode = mode
        self.name = mode
        self.tok = tokenizer or Tokenizer()

    def select(
        self, query: str, context: str, budget_tokens: int | None, metadata: dict | None = None
    ) -> SelectionResult:
        ids = self.tok.encode(context)
        src = len(ids)
        if budget_tokens is None or src <= budget_tokens:  # unconstrained or already fits
            kept = ids
        elif self.mode == "head":
            kept = ids[:budget_tokens]
        elif self.mode == "tail":
            kept = ids[-budget_tokens:]
        else:
            half = budget_tokens // 2
            tail_n = budget_tokens - half
            kept = ids[:half] + (ids[-tail_n:] if tail_n > 0 else [])
        text = self.tok.decode(kept)
        return SelectionResult(
            text=text,
            selected_token_count=len(kept),
            source_token_count=src,
            clipped=budget_tokens is not None and src > budget_tokens,
            metadata={"mode": self.mode},
        )
