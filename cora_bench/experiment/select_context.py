"""Run each selector once over every dataset and budget, with wall-clock timing.

A fixed subset of questions is rerun at the other seeds to confirm selection is deterministic.
"""

import argparse
import gc
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np

from cora_bench.budgets.token_budget import BudgetEnforcer, Tokenizer
from cora_bench.config.schema import RunConfig
from cora_bench.methods.registry import get_selector

from .api import OUT, PROTOCOL, ROOT, digest, save
from .packing import SourceUnionEnforcer

GPU_METHODS = {"dense", "dense_rerank", "llmlingua2", "provence"}


def seeded(seed, gpu):
    random.seed(seed)
    np.random.seed(seed)
    if gpu:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def sync(gpu):
    if gpu:
        import torch

        torch.cuda.synchronize()


def reset_cache(selector):
    for name in ("_cache", "_doc_cache", "_rank_cache"):
        if hasattr(selector, name):
            setattr(selector, name, {})


def initialize(selector, method):
    if method == "dense":
        selector._m()
    elif method == "dense_rerank":
        selector._emb_model()
        selector._reranker()
    elif method == "llmlingua2":
        selector._compressor()
    elif method == "provence":
        selector._m()


def write_gz(p, record):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".partial")
    with gzip.open(tmp, "wt") as f:
        json.dump(record, f, ensure_ascii=False)
    tmp.replace(p)


def upload(p):
    uri = PROTOCOL["gcs_prefix"] + "/" + str(p.relative_to(OUT))
    subprocess.run(
        ["gcloud", "storage", "cp", str(p), uri, "--quiet"], check=True, stdout=subprocess.DEVNULL
    )


def controls(rows, ds):
    max_budget = max(PROTOCOL["datasets"][ds]["budgets"])
    rows = [r for r in rows if r["source_tokens"] > max_budget]
    groups = {}
    for r in rows:
        key = (
            r.get("task")
            or ("abstention" if r.get("abstention") else r.get("question_type"))
            or "all"
        )
        groups.setdefault(key, []).append(r)
    chosen = set()
    ordered = {k: sorted(g, key=lambda r: digest(r["example_id"])) for k, g in groups.items()}
    while len(chosen) < min(30, len(rows)):
        for group in ordered.values():
            if group and len(chosen) < 30:
                chosen.add(group.pop(0)["example_id"])
    return chosen


def native_audit(selector, method, query, context):
    # Diagnostic only: native tokenizer lengths do not affect the shared cl100k chunks or budgets.
    if method not in {"dense", "dense_rerank"}:
        return {"status": "not_instrumented_here", "method": method}
    model = selector._m() if method == "dense" else selector._emb_model()
    chunks = selector.chunker.chunk(context)
    tk = model.tokenizer
    limit = int(model.max_seq_length)
    lengths = [len(tk.encode(c.text, add_special_tokens=True, truncation=False)) for c in chunks]
    result = {
        "embedding_native_max_length": limit,
        "chunk_count": len(chunks),
        "max_native_chunk_length": max(lengths, default=0),
        "chunks_exceeding_embedding_limit": sum(n > limit for n in lengths),
    }
    if method == "dense_rerank":
        from cora_bench.methods.dense_rerank import _INSTRUCT

        rt, _, _, _, pre, suf = selector._reranker()
        lens = [
            len(
                rt.encode(
                    f"<Instruct>: {_INSTRUCT}\n<Query>: {query}\n<Document>: {c.text}",
                    add_special_tokens=False,
                )
            )
            + len(pre)
            + len(suf)
            for c in chunks
        ]
        result.update(
            reranker_native_limit=8192,
            max_native_reranker_length=max(lens, default=0),
            chunks_exceeding_reranker_limit=sum(n > 8192 for n in lens),
        )
    return result


def shard_rows(rows, index, count):
    """Assign rows to a shard by example-id hash, preserving row order."""
    assert count > 0 and 0 <= index < count
    return [r for r in rows if int(digest(r["example_id"]), 16) % count == index]


def run(
    kind, cloud=False, limit=None, worker=None, method_filter=None, shard_index=0, shard_count=1
):
    methods = (
        ["head_tail", "bm25"]
        if kind == "cpu"
        else ["dense", "dense_rerank", "llmlingua2", "provence"]
    )
    if worker:
        assert kind == "gpu" and worker.replace("-", "").isalnum()
        assert method_filter and set(method_filter) <= set(methods)
        methods = [m for m in methods if m in method_filter]
    else:
        assert not method_filter and shard_index == 0 and shard_count == 1
    gpu = kind == "gpu"
    if gpu:
        import torch

        assert torch.cuda.is_available(), "GPU worker must have CUDA"
    tok = Tokenizer()
    enforcer = BudgetEnforcer(tok)
    code = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ["cora_bench/methods", "cora_bench/budgets", "cora_bench/chunking"]
        for p in (ROOT / folder).rglob("*.py")
    }
    code.update(
        {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                Path(__file__).with_name("api.py"),
                Path(__file__).with_name("packing.py"),
            ]
        }
    )
    manifest = {
        "kind": kind,
        "protocol_sha256": digest(PROTOCOL),
        "code": code,
        "started_unix": time.time(),
        "seed": 5768,
        "cold_model_load_separate": True,
    }
    manifest["environment"] = {
        "python": platform.python_version(),
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
    }
    dest = OUT / ("selection_smoke" if limit else "selection") / kind
    dest.mkdir(parents=True, exist_ok=True)
    control = dest / "workers" / worker if worker else dest
    if worker:
        manifest.update(
            worker=worker,
            methods=methods,
            shard_index=shard_index,
            shard_count=shard_count,
            ownership="sha256 JSON example_id modulo shard_count; controls chosen on full cohort",
            historical_manifest="selection/gpu/manifest.json",
        )
    if (control / "manifest.json").exists():
        old = json.loads((control / "manifest.json").read_text())
        assert old["code"] == code and old["protocol_sha256"] == manifest["protocol_sha256"]
        for key in ("worker", "methods", "shard_index", "shard_count"):
            assert old.get(key) == manifest.get(key)
    else:
        save(control / "manifest.json", manifest)
    if cloud:
        upload(control / "manifest.json")
    for method in methods:
        seeded(5768, gpu)
        cfg = RunConfig(
            experiment_name="reader_followup_20261002",
            dataset="matrix",
            method=method,
            reader="none",
            embedding_model=PROTOCOL["models"]["dense"],
            reranker_model=PROTOCOL["models"]["reranker"],
        )
        selector = get_selector(method, cfg, tok)
        if method in {"bm25", "dense", "dense_rerank"}:
            selector.enforcer = SourceUnionEnforcer(tok)
        t = time.perf_counter()
        initialize(selector, method)
        sync(gpu)
        load_seconds = time.perf_counter() - t
        # Synthetic warm-up so first-call overhead is not included in any timing.
        warm = "Example company revenue was 10 dollars. " * 100
        if isinstance(
            selector.enforcer if hasattr(selector, "enforcer") else None, SourceUnionEnforcer
        ):
            selector.enforcer.prepare(warm)
        selector.select("What was revenue?", warm, 512, metadata={"example_id": "warmup"})
        sync(gpu)
        reset_cache(selector)
        save(
            control / f"{method}_load.json",
            {"model_load_seconds": load_seconds, "device": "cuda" if gpu else "cpu"},
        )
        for ds in ["loong", "babilong", "nolima", "longmemeval"]:
            inp = ROOT / "cora_bench/outputs/materialized" / ds
            rows = [json.loads(l) for l in (inp / "input.jsonl").open()]
            assert len(rows) == PROTOCOL["datasets"][ds]["n"]
            control_ids = controls(rows, ds)
            if worker:
                rows = shard_rows(rows, shard_index, shard_count)
            if limit:
                rows = rows[:limit]
            for i, r in enumerate(rows):
                p = dest / ds / method / (r["example_id"] + ".json.gz")
                if p.exists():
                    continue
                context = (inp / "ctx" / (r["example_id"] + ".txt")).read_bytes().decode("utf-8")
                source_count = tok.count(context)
                assert source_count == r["source_tokens"], (ds, r["example_id"])
                record = {
                    "dataset": ds,
                    "example_id": r["example_id"],
                    "source_sha256": hashlib.sha256(context.encode()).hexdigest(),
                    "query_sha256": digest(r["query"]),
                    "source_tokens": source_count,
                    "method": method,
                    "row": r,
                    "selections": [],
                    "seed_controls": [],
                }
                seeded(5768, gpu)
                reset_cache(selector)
                if r["example_id"] in sorted(control_ids)[:5]:
                    record["native_tokenizer_audit"] = native_audit(
                        selector, method, r["query"], context
                    )
                for b in PROTOCOL["datasets"][ds]["budgets"]:
                    if source_count <= b:
                        continue
                    if r["example_id"] in control_ids:
                        reset_cache(selector)
                    sync(gpu)
                    start = time.perf_counter()
                    if isinstance(getattr(selector, "enforcer", None), SourceUnionEnforcer):
                        selector.enforcer.prepare(context)
                    sr = selector.select(
                        r["query"], context, b, metadata={"example_id": r["example_id"]}
                    )
                    sync(gpu)
                    text, clipped = enforcer.clip_text(sr.text, b)
                    n = tok.count(text)
                    assert n <= b
                    elapsed = time.perf_counter() - start
                    if isinstance(getattr(selector, "enforcer", None), SourceUnionEnforcer):
                        assert not clipped, "Union packer must enforce the final cap itself"
                    ids = sr.selected_chunk_ids if not clipped else []
                    record["selections"].append(
                        {
                            "budget": b,
                            "filtered_text": text,
                            "selected_tokens": n,
                            "selected_chunk_ids": ids,
                            "text_sha256": digest(text.encode()),
                            "selector_wall_seconds": elapsed,
                            "model_compute_seconds": sr.gpu_seconds,
                            "metadata": sr.metadata,
                            "clipped": sr.clipped or clipped,
                            "selection_seed": 5768,
                            "standalone_latency_sample": r["example_id"] in control_ids,
                            "source_token_spans": getattr(selector.enforcer, "last_spans", None)
                            if hasattr(selector, "enforcer")
                            else None,
                            "packing": "source_union_v1"
                            if isinstance(getattr(selector, "enforcer", None), SourceUnionEnforcer)
                            else "paper_original",
                        }
                    )
                if r["example_id"] in control_ids:
                    for seed in PROTOCOL["seeds"][1:]:
                        seeded(seed, gpu)
                        reset_cache(selector)
                        for first in record["selections"]:
                            b = first["budget"]
                            reset_cache(selector)
                            sync(gpu)
                            t = time.perf_counter()
                            if isinstance(getattr(selector, "enforcer", None), SourceUnionEnforcer):
                                selector.enforcer.prepare(context)
                            sr = selector.select(
                                r["query"], context, b, metadata={"example_id": r["example_id"]}
                            )
                            sync(gpu)
                            text, _ = enforcer.clip_text(sr.text, b)
                            elapsed = time.perf_counter() - t
                            record["seed_controls"].append(
                                {
                                    "seed": seed,
                                    "budget": b,
                                    "selector_wall_seconds": elapsed,
                                    "same_text": text == first["filtered_text"],
                                    "text_sha256": digest(text.encode()),
                                }
                            )
                    # Reader seeds reuse one selection, which is only valid if it is deterministic.
                    assert all(c["same_text"] for c in record["seed_controls"]), (
                        f"Non-deterministic {ds}/{method}/{r['example_id']}; seed-specific full selections required"
                    )
                write_gz(p, record)
                if cloud:
                    upload(p)
                save(
                    control / "status.json",
                    {
                        "dataset": ds,
                        "method": method,
                        "done_in_group": i + 1,
                        "total_in_group": len(rows),
                        "updated_unix": time.time(),
                    },
                )
                if cloud and (i % 10 == 0 or i + 1 == len(rows)):
                    upload(control / "status.json")
                print(
                    json.dumps(
                        {"dataset": ds, "method": method, "done": i + 1, "total": len(rows)}
                    ),
                    flush=True,
                )
                reset_cache(selector)
        del selector
        gc.collect()
        if gpu:
            torch.cuda.empty_cache()
    save(
        control / "complete.json",
        {
            "complete": True,
            "limit": limit,
            "finished_unix": time.time(),
            "worker": worker,
            "methods": methods,
            "shard_index": shard_index,
            "shard_count": shard_count,
        },
    )
    if cloud:
        upload(control / "complete.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=["cpu", "gpu"])
    p.add_argument("--cloud", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--worker")
    p.add_argument("--methods", nargs="+")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    a = p.parse_args()
    run(a.kind, a.cloud, a.limit, a.worker, a.methods, a.shard_index, a.shard_count)
