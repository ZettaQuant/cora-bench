"""Shared chunk representation. All retrieval methods consume identical chunks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Chunk:
    chunk_id: str
    text: str
    original_order: int
    token_start: int
    token_end: int
    doc_id: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def token_count(self) -> int:
        return self.token_end - self.token_start


class Chunker(ABC):
    @abstractmethod
    def chunk(self, context: str, doc_id: str | None = None) -> list[Chunk]: ...
