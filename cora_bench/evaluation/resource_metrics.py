"""Resource-metric derivations from a per-example record."""

from __future__ import annotations


def compression_ratio(selected_tokens: int, source_tokens: int) -> float:
    return selected_tokens / source_tokens if source_tokens else 0.0


def unused_budget(budget_tokens: int | None, selected_tokens: int) -> int | None:
    if budget_tokens is None:
        return None
    return max(budget_tokens - selected_tokens, 0)


def total_cost(selector_cost_usd: float, reader_cost_usd: float) -> float:
    return round(selector_cost_usd + reader_cost_usd, 8)


def end_to_end_latency(selector_latency_s: float, reader_latency_s: float) -> float:
    return round(selector_latency_s + reader_latency_s, 4)
