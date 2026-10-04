"""Run configuration. YAML-loadable dataclass; every run saves its fully resolved config."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class RunConfig:
    experiment_name: str
    dataset: str
    method: str
    reader: str

    dataset_split: str = "test"
    dataset_subset: str | None = None
    seed: int = 0

    source_context_length: int | None = None  # optional dataset-controlled source length
    context_budget_tokens: int | None = None  # None for unconstrained full_context

    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64
    chunk_order: str = "source_order"  # source_order | rank_order

    embedding_model: str | None = None
    reranker_model: str | None = None
    retrieval_candidate_count: int = 100
    compressor_model: str | None = None

    reader_model: str | None = None
    reader_temperature: float = 0.0
    reader_max_output_tokens: int = 2048

    tokenizer_encoding: str = "cl100k_base"
    pricing_config: str = "pricing_2026_09"
    output_dir: str = "cora_bench/outputs/raw"

    batch_size: int = 8
    num_workers: int = 4
    limit: int | None = None
    extra: dict = field(default_factory=dict)  # dataset/method-specific optional fields

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        return cls(**yaml.safe_load(Path(path).read_text()))

    def to_dict(self) -> dict:
        return asdict(self)
