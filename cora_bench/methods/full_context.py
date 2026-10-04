"""Full-context reference: returns the context unchanged and ignores the budget."""

from __future__ import annotations

from cora_bench.budgets.token_budget import Tokenizer
from cora_bench.methods.base import ContextSelector, SelectionResult


class FullContext(ContextSelector):
    name = "full_context"
    is_budgeted = False

    def __init__(self, tokenizer: Tokenizer | None = None):
        self.tok = tokenizer or Tokenizer()

    def select(
        self, query: str, context: str, budget_tokens: int, metadata: dict | None = None
    ) -> SelectionResult:
        n = self.tok.count(context)
        return SelectionResult(
            text=context,
            selected_token_count=n,
            source_token_count=n,
            metadata={"unconstrained": True},
        )
