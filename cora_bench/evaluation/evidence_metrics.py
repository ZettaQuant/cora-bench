"""Dataset-neutral evidence metrics. Dataset-specific matching lives in the data adapters."""

from __future__ import annotations

from collections.abc import Callable


def recall(gold_ids: set[str], selected_ids: set[str]) -> float:
    if not gold_ids:
        return float("nan")
    return len(gold_ids & selected_ids) / len(gold_ids)


def precision(gold_ids: set[str], selected_ids: set[str]) -> float:
    if not selected_ids:
        return 0.0
    return len(gold_ids & selected_ids) / len(selected_ids)


def f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def all_retained(gold_ids: set[str], selected_ids: set[str]) -> bool:
    return gold_ids.issubset(selected_ids)


def evidence_survived(
    gold_texts: list[str], selected_context: str, matcher: Callable[[str, str], bool]
) -> dict:
    """Fraction of gold evidence strings the adapter's `matcher(gold, context)` finds present."""
    if not gold_texts:
        return {"fraction_retained": float("nan"), "all_retained": None}
    hits = sum(1 for g in gold_texts if matcher(g, selected_context))
    return {"fraction_retained": hits / len(gold_texts), "all_retained": hits == len(gold_texts)}
