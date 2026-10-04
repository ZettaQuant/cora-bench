"""Dataset adapter registry."""

from __future__ import annotations

from cora_bench.data.base import DatasetAdapter


def get_dataset(name: str, **kwargs) -> DatasetAdapter:
    if name == "loong":
        from cora_bench.data.loong import LoongAdapter

        return LoongAdapter(**kwargs)
    if name == "babilong":
        from cora_bench.data.babilong import BabiLongAdapter

        return BabiLongAdapter(**kwargs)
    if name == "longmemeval":
        from cora_bench.data.longmemeval import LongMemEvalAdapter

        return LongMemEvalAdapter(**kwargs)
    if name == "nolima":
        from cora_bench.data.nolima import NoLiMaAdapter

        return NoLiMaAdapter(**kwargs)
    raise KeyError(f"unknown dataset {name!r}")
