"""Aggregate per-example runner shards into a summary table.

Usage: python -m cora_bench.analysis.aggregate --experiment NAME [--by-task]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

from cora_bench.utils.io import read_jsonl, write_json


def _fmt_budget(b) -> str:
    return "inf" if b is None else (f"{b // 1000}K" if b >= 1000 else str(b))


def _m(vals):
    vals = [v for v in vals if v is not None]
    return round(mean(vals), 3) if vals else float("nan")


def aggregate(
    experiment: str, root: str = "cora_bench/outputs/raw", by_task: bool = False
) -> list[dict]:
    d = Path(root) / experiment
    latest: dict[tuple, dict] = {}  # one row per key, preferring a successful row
    for p in sorted(d.glob("*.jsonl")):
        for r in read_jsonl(p):
            key = (
                r["method"],
                r.get("context_budget_tokens"),
                r["example_id"],
                r.get("dataset"),
                r.get("reader_model"),
                r.get("seed"),
                r.get("scoring_version", "legacy"),
            )
            prev = latest.get(key)
            if prev is None or prev.get("error_type") or not r.get("error_type"):
                latest[key] = r

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for key, r in latest.items():
        method, budget = key[:2]
        gkey = (method, budget, r.get("task") if by_task else None, *key[3:])
        groups[gkey].append(r)

    table = []
    for gkey, rows in sorted(
        groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or 10**12, str(kv[0][2:]))
    ):
        ok = [r for r in rows if not r.get("error_type")]
        n = len(rows)
        all_surv = [r for r in ok if r.get("evidence_all_retained") is True]
        lost = [r for r in ok if r.get("evidence_all_retained") is False]
        row = {
            "method": gkey[0],
            "budget": _fmt_budget(gkey[1]),
            "dataset": gkey[3],
            "reader_model": gkey[4],
            "seed": gkey[5],
            "scoring_version": gkey[6],
            "n": n,
            "errors": n - len(ok),
            "coverage": round(len(ok) / n, 3) if n else float("nan"),
            "accuracy_e2e": _m([bool(r.get("answer_correct")) for r in rows]),
            "accuracy_ok": _m([r.get("answer_correct") for r in ok]),
            "accuracy_strict": _m(
                [
                    r.get("answer_correct_strict")
                    for r in ok
                    if r.get("answer_correct_strict") is not None
                ]
            ),
            "support_fact_recall": _m([r.get("evidence_fraction_retained") for r in ok]),
            "all_support_retained": _m([r.get("evidence_all_retained") for r in ok]),
            "support_token_recall": _m([r.get("evidence_support_token_recall") for r in ok]),
            "acc_given_all_evidence": _m([r.get("answer_correct") for r in all_surv]),
            "acc_given_evidence_lost": _m([r.get("answer_correct") for r in lost]),
            "compression": _m([r.get("compression_ratio") for r in ok]),
            "avg_cost_usd": round(_m([r.get("total_cost_usd") for r in ok]), 4),
            "avg_latency_s": _m([r.get("end_to_end_latency_seconds") for r in ok]),
        }
        if by_task:
            row["task"] = gkey[2]
        table.append(row)
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--root", default="cora_bench/outputs/raw")
    ap.add_argument("--by-task", action="store_true")
    args = ap.parse_args()
    table = aggregate(args.experiment, args.root, by_task=args.by_task)
    cols = ["method", "budget"] + (["task"] if args.by_task else [])
    hdr = (
        f"{'|'.join(f'{c:>12}' for c in cols)} {'n':>3} {'cov':>5} {'acc_e2e':>7} {'acc_ok':>6} "
        f"{'sfr':>5} {'allsup':>6} {'acc|all':>7} {'acc|lost':>8} {'compr':>6} {'$/q':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in table:
        lead = "|".join(f"{r[c]:>12}" for c in cols)
        print(
            f"{lead} {r['n']:>3} {r['coverage']:>5.3f} {r['accuracy_e2e']:>7.3f} {r['accuracy_ok']:>6.3f} "
            f"{r['support_fact_recall']:>5.3f} {r['all_support_retained']:>6.3f} "
            f"{r['acc_given_all_evidence']:>7.3f} {r['acc_given_evidence_lost']:>8.3f} "
            f"{r['compression']:>6.3f} {r['avg_cost_usd']:>7.4f}"
        )
    suffix = "_by_task" if args.by_task else ""
    out = Path(args.root) / args.experiment / f"summary{suffix}.json"
    write_json(out, table)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
