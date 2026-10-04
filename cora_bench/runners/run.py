"""Per-example selection, reading, and scoring for one configuration."""

from __future__ import annotations

from pathlib import Path

from cora_bench.budgets.token_budget import BudgetEnforcer, Tokenizer
from cora_bench.config.schema import RunConfig
from cora_bench.data.base import DatasetAdapter, Example
from cora_bench.evaluation.resource_metrics import (
    compression_ratio,
    end_to_end_latency,
    unused_budget,
)
from cora_bench.evaluation.selection import SCORING_VERSION, score_selection
from cora_bench.instrumentation.cost import Pricing
from cora_bench.instrumentation.selection_cost import round_cost, selection_cost, sum_cost
from cora_bench.methods.base import ContextSelector
from cora_bench.readers.base import Reader
from cora_bench.utils.io import append_jsonl, read_jsonl


def budget_from_label(label: str) -> int | None:
    return None if str(label).lower() in ("inf", "none", "full") else int(label)


def shard_path(cfg: RunConfig, dataset: str, method: str, reader: str, budget_label: str) -> Path:
    d = Path(cfg.output_dir) / cfg.experiment_name
    return d / f"{dataset}__{method}__{reader}__b{budget_label}.jsonl"


def _done_ids(path: Path) -> set[str]:
    """IDs of successful rows; errored examples are retried on resume."""
    if not path.exists():
        return set()
    return {r["example_id"] for r in read_jsonl(path) if not r.get("error_type")}


def run_one(
    adapter: DatasetAdapter,
    selector: ContextSelector,
    reader: Reader,
    example: Example,
    budget: int | None,
    cfg: RunConfig,
    pricing: Pricing,
    enforcer: BudgetEnforcer,
    src_tokens: int,
) -> dict:
    row: dict = {
        "dataset": adapter.name,
        "dataset_subset": cfg.dataset_subset,
        "example_id": example.example_id,
        "method": selector.name,
        "reader": reader.name,
        "reader_model": reader.model,
        "seed": cfg.seed,
        "context_budget_tokens": budget,
        "source_tokens": src_tokens,
        "set": example.metadata.get("set"),
        "error_type": None,
        "error_message": None,
        "scoring_version": SCORING_VERSION,
    }
    try:
        sel = selector.select(
            example.query, example.context, budget, metadata={"example_id": example.example_id}
        )
    except Exception as e:
        row["error_type"] = "selector"
        row["error_message"] = repr(e)[:400]
        return row

    text, clipped = sel.text, sel.clipped
    chunk_ids = list(sel.selected_chunk_ids)
    sel_tokens = enforcer.tok.count(
        text
    )  # recount with the shared tokenizer rather than trusting the selector
    if budget is not None and sel_tokens > budget:
        text, was_clipped = enforcer.clip_text(text, budget)
        clipped = clipped or was_clipped
        sel_tokens = enforcer.tok.count(text)
        chunk_ids = []  # clipped text no longer matches whole chunks
    assert budget is None or sel_tokens <= budget, (sel_tokens, budget)

    try:
        ev = (
            score_selection(
                adapter,
                example,
                text,
                chunk_ids,
                method=selector.name,
                metadata=sel.metadata,
                budget=budget,
                chunk_size=cfg.chunk_size_tokens,
                overlap=cfg.chunk_overlap_tokens,
            )
            or {}
        )
    except ValueError as e:
        row.update(error_type="evidence_validation", error_message=str(e))
        return row
    rr = reader.generate(example.query, text, example_metadata=example.metadata)
    ans = adapter.score_answer(example, rr.parsed_answer)

    sel_cost = selection_cost(
        selector.name,
        sel.metadata,
        pricing,
        sel.input_tokens_used_by_selector,
        sel.output_tokens_used_by_selector,
        sel.gpu_seconds,
        cfg.extra.get("gpu_type"),
        sel.cost_usd,
    )
    try:
        reader_cost = pricing.api_cost(
            reader.model, rr.input_tokens, rr.output_tokens, rr.thought_tokens
        )
    except KeyError as e:
        reader_cost = None
        row["error_message"] = f"pricing: {e}"

    row.update(
        {
            "selected_tokens": sel_tokens,
            "compression_ratio": round(compression_ratio(sel_tokens, src_tokens), 5),
            "unused_budget_tokens": unused_budget(budget, sel_tokens),
            "clipped": clipped,
            "budget_exceeded": bool(budget is not None and sel_tokens > budget),
            "selector_input_tokens": sel.input_tokens_used_by_selector,
            "selector_output_tokens": sel.output_tokens_used_by_selector,
            "selector_api_calls": sel.selector_api_calls,
            "selector_gpu_seconds": sel.gpu_seconds,
            "reader_input_tokens": rr.input_tokens,
            "reader_output_tokens": rr.output_tokens,
            "reader_thought_tokens": rr.thought_tokens,
            "reader_api_calls": rr.reader_api_calls,
            "selector_latency_seconds": sel.latency_seconds,
            "reader_latency_seconds": rr.latency_seconds,
            "end_to_end_latency_seconds": end_to_end_latency(
                sel.latency_seconds, rr.latency_seconds
            ),
            "selector_cost_usd": round_cost(sel_cost),
            "reader_cost_usd": round_cost(reader_cost),
            "total_cost_usd": sum_cost(sel_cost, reader_cost),
            "cost_complete": sel_cost is not None and reader_cost is not None,
            "evidence_details": ev,
            "scoring_version": SCORING_VERSION,
            "answer_metric": ans.metric,
            "answer_correct": ans.correct,
            "answer_score": ans.score,
            "answer_correct_strict": (ans.detail or {}).get("strict"),
            "parsed_answer": rr.parsed_answer,
            "raw_reader_output": rr.raw_output,
            "evidence_metric": ev.get("metric"),
            "evidence_fraction_retained": ev.get("fraction_retained"),
            "evidence_all_retained": ev.get("all_retained"),
            "evidence_n_support": ev.get("n_support"),
            "evidence_support_token_recall": ev.get("support_token_recall"),
            "task": example.metadata.get("task"),
            "strata": {
                k: example.metadata.get(k)
                for k in ("task", "question_type", "hop", "depth", "reasoning_type", "abstention")
                if k in example.metadata
            },
            "selector_metadata": sel.metadata,
            "reader_finish_reason": rr.provider_metadata.get("finish_reason"),
            "prompt_version": rr.provider_metadata.get("prompt_version"),
            "reader_error": rr.error,
        }
    )
    if rr.error:
        row["error_type"] = "reader"
        row["error_message"] = rr.error
    return row


def read_one(
    adapter: DatasetAdapter,
    reader: Reader,
    example: Example,
    sel: dict,
    cfg: RunConfig,
    pricing: Pricing,
    gpu_type: str = "NVIDIA L4",
) -> dict:
    """Like run_one, but reads and scores a stored selection instead of running the selector."""
    budget, src = sel.get("budget"), sel.get("source_tokens")
    text, sel_tokens = sel.get("filtered_text", ""), sel.get("selected_tokens")
    row: dict = {
        "dataset": adapter.name,
        "example_id": example.example_id,
        "method": sel["method"],
        "reader": reader.name,
        "reader_model": reader.model,
        "seed": cfg.seed,
        "context_budget_tokens": budget,
        "source_tokens": src,
        "selected_tokens": sel_tokens,
        "compression_ratio": round(compression_ratio(sel_tokens or 0, src or 0), 5),
        "unused_budget_tokens": unused_budget(budget, sel_tokens or 0),
        "clipped": sel.get("clipped"),
        "set": example.metadata.get("set"),
        "error_type": None,
        "error_message": None,
        "scoring_version": SCORING_VERSION,
    }
    if sel.get("error"):
        row["error_type"] = "selector"
        row["error_message"] = sel["error"]
        return row

    tok = Tokenizer(cfg.tokenizer_encoding)
    import hashlib

    src_actual = tok.count(example.context)
    if (
        src != src_actual
        or sel.get("source_sha256", hashlib.sha256(example.context.encode()).hexdigest())
        != hashlib.sha256(example.context.encode()).hexdigest()
    ):
        row.update(
            error_type="source_mismatch",
            error_message="Stored selection source differs from dataset",
        )
        return row
    sel_tokens = tok.count(text)
    if budget is not None and sel_tokens > budget:
        row.update(error_type="budget", error_message="Stored selection exceeds budget")
        return row
    row.update(
        selected_tokens=sel_tokens,
        compression_ratio=compression_ratio(sel_tokens, src or 0),
        unused_budget_tokens=unused_budget(budget, sel_tokens),
    )
    try:
        from cora_bench.analysis.dataset_report import verified_legacy_ids

        ids, rebased = verified_legacy_ids(adapter, example, sel)
        row["legacy_provenance_rebased"] = rebased
        ev = (
            score_selection(
                adapter,
                example,
                text,
                ids,
                method=sel["method"],
                metadata=sel.get("metadata"),
                budget=budget,
                chunk_size=cfg.chunk_size_tokens,
                overlap=cfg.chunk_overlap_tokens,
            )
            or {}
        )
    except ValueError as e:
        row.update(error_type="evidence_validation", error_message=str(e))
        return row
    rr = reader.generate(example.query, text, example_metadata=example.metadata)
    ans = adapter.score_answer(example, rr.parsed_answer)

    gpu_s = sel.get("gpu_seconds") or 0.0
    sel_in, sel_out = sel.get("selector_input_tokens") or 0, sel.get("selector_output_tokens") or 0
    filter_model = (sel.get("metadata") or {}).get("filter_model")
    sel_cost = selection_cost(
        sel["method"],
        sel.get("metadata"),
        pricing,
        sel_in,
        sel_out,
        gpu_s,
        gpu_type,
        sel.get("selector_cost_usd"),
    )
    try:
        reader_cost = pricing.api_cost(
            reader.model, rr.input_tokens, rr.output_tokens, rr.thought_tokens
        )
    except KeyError as e:
        reader_cost = None
        row["error_message"] = f"pricing: {e}"

    sel_lat = sel.get("selector_latency_s") or 0.0
    row.update(
        {
            "selector_input_tokens": sel_in,
            "selector_output_tokens": sel_out,
            "selector_api_calls": sel.get("selector_api_calls") or 0,
            "selector_gpu_seconds": gpu_s,
            "reader_input_tokens": rr.input_tokens,
            "reader_output_tokens": rr.output_tokens,
            "reader_thought_tokens": rr.thought_tokens,
            "reader_api_calls": rr.reader_api_calls,
            "selector_latency_seconds": sel_lat,
            "reader_latency_seconds": rr.latency_seconds,
            "end_to_end_latency_seconds": end_to_end_latency(sel_lat, rr.latency_seconds),
            "selector_cost_usd": round_cost(sel_cost),
            "reader_cost_usd": round_cost(reader_cost),
            "total_cost_usd": sum_cost(sel_cost, reader_cost),
            "cost_complete": sel_cost is not None and reader_cost is not None,
            "evidence_details": ev,
            "scoring_version": SCORING_VERSION,
            "answer_metric": ans.metric,
            "answer_correct": ans.correct,
            "answer_score": ans.score,
            "answer_correct_strict": (ans.detail or {}).get("strict"),
            "parsed_answer": rr.parsed_answer,
            "raw_reader_output": rr.raw_output,
            "evidence_metric": ev.get("metric"),
            "evidence_fraction_retained": ev.get("fraction_retained"),
            "evidence_all_retained": ev.get("all_retained"),
            "evidence_n_support": ev.get("n_support"),
            "evidence_support_token_recall": ev.get("support_token_recall"),
            "task": example.metadata.get("task"),
            "strata": {
                k: example.metadata.get(k)
                for k in ("task", "question_type", "hop", "depth", "reasoning_type", "abstention")
                if k in example.metadata
            },
            "selector_metadata": sel.get("metadata"),
            "gpu_type": None if filter_model else gpu_type,
            "reader_finish_reason": rr.provider_metadata.get("finish_reason"),
            "prompt_version": rr.provider_metadata.get("prompt_version"),
            "reader_error": rr.error,
        }
    )
    if rr.error:
        row["error_type"] = "reader"
        row["error_message"] = rr.error
    return row
