"""Re-time the CPU selectors on the control questions once other workloads have finished."""

import gzip
import json
import time

from cora_bench.budgets.token_budget import BudgetEnforcer, Tokenizer
from cora_bench.config.schema import RunConfig
from cora_bench.methods.registry import get_selector

from .api import OUT, PROTOCOL, ROOT, digest, save
from .packing import SourceUnionEnforcer
from .select_context import controls, reset_cache, seeded


def run():
    dest = OUT / "cpu_standalone_latency"
    tok = Tokenizer()
    clip = BudgetEnforcer(tok)
    for method in ["head_tail", "bm25"]:
        selector = get_selector(
            method,
            RunConfig(experiment_name="latency", dataset="matrix", method=method, reader="none"),
            tok,
        )
        if method == "bm25":
            selector.enforcer = SourceUnionEnforcer(tok)
        warm = "A short fixed revenue example. " * 100
        if method == "bm25":
            selector.enforcer.prepare(warm)
        selector.select("revenue", warm, 512)
        for ds in PROTOCOL["datasets"]:
            base = ROOT / "cora_bench/outputs/materialized" / ds
            rows = [json.loads(l) for l in (base / "input.jsonl").open()]
            ids = controls(rows, ds)
            for row in rows:
                if row["example_id"] not in ids:
                    continue
                p = dest / ds / method / (row["example_id"] + ".json")
                if p.exists():
                    continue
                src = OUT / "selection/cpu" / ds / method / (row["example_id"] + ".json.gz")
                with gzip.open(src, "rt") as f:
                    old = json.load(f)
                context = (base / "ctx" / (row["example_id"] + ".txt")).read_bytes().decode("utf8")
                result = []
                for s in old["selections"]:
                    times = []
                    for seed in PROTOCOL["seeds"]:
                        seeded(seed, False)
                        reset_cache(selector)
                        start = time.perf_counter()
                        if method == "bm25":
                            selector.enforcer.prepare(context)
                        out = selector.select(
                            row["query"],
                            context,
                            s["budget"],
                            metadata={"example_id": row["example_id"]},
                        )
                        text, _ = clip.clip_text(out.text, s["budget"])
                        assert tok.count(text) <= s["budget"]
                        elapsed = time.perf_counter() - start
                        assert digest(text.encode()) == s["text_sha256"], (
                            "Latency rerun differs from primary selection"
                        )
                        times.append(elapsed)
                    result.append(
                        {
                            "budget": s["budget"],
                            "seeds": PROTOCOL["seeds"],
                            "wall_seconds": times,
                            "text_sha256": s["text_sha256"],
                        }
                    )
                save(p, result)
    save(
        dest / "complete.json",
        {
            "complete": True,
            "finished_unix": time.time(),
            "purpose": "Fresh-cache CPU timing after the reader and selection workloads finished",
        },
    )


if __name__ == "__main__":
    run()
