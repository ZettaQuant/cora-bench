"""The shared `Example` record and the dataset adapter interface.

For conversational data, `context` is the full history and `query` is the current question.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field


@dataclass
class GoldEvidence:
    """One piece of gold evidence; datasets fill whichever fields apply."""

    text: str | None = None
    chunk_ids: list[str] = field(default_factory=list)
    doc_id: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Example:
    example_id: str
    query: str
    context: str
    answer: str
    answer_choices: list[str] | None = None
    gold_evidence: list[GoldEvidence] = field(default_factory=list)
    metadata: dict = field(
        default_factory=dict
    )  # dataset, subset, source_tokens, doc/conv ids, ...


@dataclass
class AnswerScore:
    correct: bool
    score: float
    metric: str
    detail: dict = field(default_factory=dict)


class DatasetAdapter(ABC):
    """Loads examples and scores answers with a named, dataset-specific metric."""

    name: str
    has_gold_evidence: bool = False

    @abstractmethod
    def load(
        self, split: str, subset: str | None = None, limit: int | None = None
    ) -> Iterator[Example]: ...

    @abstractmethod
    def score_answer(self, example: Example, prediction: str) -> AnswerScore: ...

    def expected_count(self, split: str, subset: str | None = None) -> int | None:
        """Expected number of examples, or None if not fixed."""
        return None

    def evidence_retained(
        self,
        example: Example,
        selected_context: str,
        selected_chunk_ids: set[str],
        chunk_size=512,
        overlap=64,
        source_token_spans=None,
    ) -> dict | None:
        """Evidence retention under this dataset's definition of a hit; None without gold."""
        return None
