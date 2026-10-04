"""Downstream reader interface that is independent of the selector."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ReaderResult:
    raw_output: str
    parsed_answer: str
    input_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int = 0
    latency_seconds: float = 0.0
    cost_usd: float = 0.0
    reader_api_calls: int = 0
    error: str | None = None
    provider_metadata: dict = field(default_factory=dict)


class Reader(ABC):
    name: str
    model: str

    @abstractmethod
    def generate(
        self, query: str, selected_context: str, example_metadata: dict | None = None
    ) -> ReaderResult: ...
