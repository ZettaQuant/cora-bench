"""Run selectors over materialized contexts and save selections for the evidence experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from cora_bench.budgets.token_budget import BudgetEnforcer, Tokenizer
from cora_bench.config.schema import RunConfig
from cora_bench.instrumentation.manifest import build_manifest, fingerprint, source_hashes
from cora_bench.methods.registry import get_selector
from cora_bench.runners.run import budget_from_label
from cora_bench.utils.io import append_jsonl, read_jsonl, write_json


def _done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        r["example_id"]
        for r in read_jsonl(path)
        if not r.get("error") and not (r.get("metadata") or {}).get("n_unscored", 0)
    }


def input_identity(inp: Path, rows: list[dict]) -> dict:
    """Hash the input file and every source context."""
    contexts = {}
    for row in rows:
        eid = row["example_id"]
        if eid in contexts:
            raise ValueError(f"Duplicate materialized example ID: {eid}")
        digest = hashlib.sha256()
        with (inp / "ctx" / f"{eid}.txt").open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        contexts[eid] = digest.hexdigest()
    return {
        "input_sha256": hashlib.sha256((inp / "input.jsonl").read_bytes()).hexdigest(),
        "context_sha256": contexts,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", required=True)
    ap.add_argument("--budgets", default="inf,1000000,128000,64000,32000")
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--chunk-overlap", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)  # this run takes rows[shard::nshards]
    ap.add_argument(
        "--emb-model", default=None
    )  # embedder for dense and dense_rerank
    ap.add_argument("--rerank-model", default=None)  # dense_rerank cross-encoder
    ap.add_argument("--score-cache-dir", default=None)
    args = ap.parse_args()

    tok = Tokenizer()
    enforcer = BudgetEnforcer(tok)
    inp, out = Path(args.inp), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = list(read_jsonl(inp / "input.jsonl"))
    inputs = input_identity(inp, rows)
    if args.limit:
        rows = rows[: args.limit]
    if args.nshards > 1:
        rows = rows[args.shard :: args.nshards]
    cfg = RunConfig(
        experiment_name="gpu_select",
        dataset="materialized",
        method="(matrix)",
        reader="none",
        chunk_size_tokens=args.chunk_size,
        chunk_overlap_tokens=args.chunk_overlap,
        embedding_model=args.emb_model or "Qwen/Qwen3-Embedding-0.6B",
        reranker_model=args.rerank_model or "Qwen/Qwen3-Reranker-0.6B",
        extra={"score_cache_dir": args.score_cache_dir or str(out / "score_cache")},
    )
    identity = {"config": cfg.to_dict(), "source_hashes": source_hashes(), **inputs}
    config_fingerprint = fingerprint(identity)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()).get("selection_fingerprint") != config_fingerprint:
            raise ValueError(
                "Output directory belongs to a different configuration; use a new directory"
            )
    elif any(out.glob("sel__*.jsonl")):
        raise ValueError("Legacy output has no run fingerprint; use a new directory")
    else:
        write_json(
            manifest_path, {**build_manifest(identity), "selection_fingerprint": config_fingerprint}
        )

    for method in args.methods.split(","):
        selector = get_selector(method, cfg, tok)
        for label in args.budgets.split(","):
            budget = budget_from_label(label)
            if budget is None and not selector.has_natural_arm:
                continue
            if budget is not None and not selector.is_budgeted:
                continue
            path = out / f"sel__{method}__b{label}.jsonl"
            done = _done_ids(path)
            todo = [
                r
                for r in rows
                if r["example_id"] not in done and (budget is None or budget < r["source_tokens"])
            ]
            print(f"[{method} b{label}] todo={len(todo)} done={len(done)}", flush=True)
            for r in todo:
                ctx = (inp / "ctx" / f"{r['example_id']}.txt").read_text()
                t0 = time.perf_counter()
                try:
                    if (
                        hashlib.sha256(ctx.encode()).hexdigest()
                        != inputs["context_sha256"][r["example_id"]]
                    ):
                        raise ValueError("Source changed after manifest creation")
                    if tok.count(ctx) != r["source_tokens"]:
                        raise ValueError(
                            "Materialized source token count does not match shared tokenizer"
                        )
                    sr = selector.select(
                        r["query"], ctx, budget, metadata={"example_id": r["example_id"]}
                    )
                    text = sr.text
                    clipped = sr.clipped
                    chunk_ids = sr.selected_chunk_ids
                    if (
                        budget is not None and tok.count(text) > budget
                    ):  # safety net for free-text selectors
                        text, was = enforcer.clip_text(text, budget)
                        clipped = clipped or was
                        chunk_ids = []  # clipped text no longer matches whole chunks
                    append_jsonl(
                        path,
                        {
                            "config_fingerprint": config_fingerprint,
                            "source_sha256": hashlib.sha256(ctx.encode()).hexdigest(),
                            "query_sha256": hashlib.sha256(r["query"].encode()).hexdigest(),
                            "example_id": r["example_id"],
                            "method": method,
                            "budget": budget,
                            "filtered_text": text,
                            "selected_tokens": tok.count(text),
                            "source_tokens": r["source_tokens"],
                            "selected_chunk_ids": chunk_ids,
                            "selector_latency_s": sr.latency_seconds,
                            "gpu_seconds": sr.gpu_seconds,
                            "selector_input_tokens": sr.input_tokens_used_by_selector,
                            "selector_output_tokens": sr.output_tokens_used_by_selector,
                            "selector_api_calls": sr.selector_api_calls,
                            "selector_cost_usd": sr.cost_usd,
                            "clipped": clipped,
                            "metadata": {
                                **sr.metadata,
                                "chunk_size_tokens": args.chunk_size,
                                "chunk_overlap_tokens": args.chunk_overlap,
                            },
                        },
                    )
                    print(
                        f"  {r['example_id'][:8]} keep={tok.count(text)} "
                        f"{time.perf_counter() - t0:.1f}s",
                        flush=True,
                    )
                except Exception as e:
                    append_jsonl(
                        path,
                        {
                            "example_id": r["example_id"],
                            "method": method,
                            "budget": budget,
                            "config_fingerprint": config_fingerprint,
                            "error": repr(e)[:400],
                        },
                    )
                    print(f"  {r['example_id'][:8]} ERR {repr(e)[:120]}", flush=True)
    print("gpu_select done.", flush=True)


if __name__ == "__main__":
    main()
