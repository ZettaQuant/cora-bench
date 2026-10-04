"""Context selector interface. Every method is (query, context, budget) -> selected context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class SelectionResult:
    text: str
    selected_token_count: int
    source_token_count: int
    selected_chunk_ids: list[str] = field(default_factory=list)
    provenance: list[dict] = field(default_factory=list)
    latency_seconds: float = 0.0
    input_tokens_used_by_selector: int = 0
    output_tokens_used_by_selector: int = 0
    selector_api_calls: int = 0
    cost_usd: float | None = None
    gpu_seconds: float = 0.0
    clipped: bool = False
    metadata: dict = field(default_factory=dict)

    @property
    def compression_ratio(self) -> float:
        return (
            self.selected_token_count / self.source_token_count if self.source_token_count else 0.0
        )


class ContextSelector(ABC):
    """Return C' = S(query, context; budget). Budgeted methods must keep tokens(C') <= budget."""

    name: str
    is_budgeted: bool = True
    has_natural_arm: bool = False

    @abstractmethod
    def select(
        self, query: str, context: str, budget_tokens: int, metadata: dict | None = None
    ) -> SelectionResult: ...
