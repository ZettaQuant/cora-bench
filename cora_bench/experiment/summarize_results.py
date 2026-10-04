"""Build the report: evidence recall, accuracy with paired bootstrap intervals, and cost/latency
tables and plots. Incomplete runs are labeled as partial."""

import gzip
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from cora_bench.budgets.token_budget import Tokenizer
from cora_bench.data.registry import get_dataset
from cora_bench.evaluation.selection import score_selection

from .api import OUT, PROTOCOL, ROOT, digest, save
from .packing import validate_source_union

PRICES = {"cpu": 0.13402284, "gpu": 0.853624312}


def interval(ds, rows, values):
    # Same ordered examples + same RNG produces paired draws across methods.
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        stratum = row.get("task", "all") if ds == "babilong" else "all"
        cluster = str(row["needle_id"]) if ds == "nolima" else row["example_id"]
        groups[(stratum, cluster)].append(i)
    strata = defaultdict(list)
    for (st, c), ix in sorted(groups.items()):
        strata[st].append(ix)
    rng = np.random.default_rng(5768)
    draws = []
    values = np.asarray(values, float)
    for _ in range(PROTOCOL["uncertainty"]["bootstrap_repetitions"]):
        ix = []
        for clusters in strata.values():
            for j in rng.integers(0, len(clusters), len(clusters)):
                ix.extend(clusters[j])
        draws.append(float(np.mean(values[ix])))
    return {
        "estimate": float(np.mean(values)),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "clusters": len(groups),
    }


def run():
    if (OUT / "reader/main/complete.json").exists():
        from .cpu_latency import run as cpu_latency

        cpu_latency()
    dest = OUT / "report"
    dest.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer()
    cells = []
    per_example = []
    comparisons = []
    for ds, spec in PROTOCOL["datasets"].items():
        root = ROOT / "cora_bench/outputs/materialized" / ds
        rows = sorted(
            [json.loads(l) for l in (root / "input.jsonl").open()], key=lambda r: r["example_id"]
        )
        byid = {r["example_id"]: r for r in rows}
        links = {}
        for p in (OUT / "reader/main/links" / ds).glob("*/*.json"):
            link = json.loads(p.read_text())
            links[(link["method"], link["budget"], link["example_id"], link["seed"])] = link
        latency_links = {}
        for p in (OUT / "reader/latency/links" / ds).glob("*/*.json"):
            link = json.loads(p.read_text())
            latency_links[(link["method"], link["budget"], link["example_id"])] = link
        adapter = get_dataset(ds, data_dir=str(root))
        evdir = dest / "evidence" / ds
        evdir.mkdir(parents=True, exist_ok=True)
        for ex in adapter.load():
            eid = ex.example_id
            raw = (root / "ctx" / (eid + ".txt")).read_bytes().decode("utf8")
            assert ex.context == raw, "Adapter changed source bytes"
            for control in PROTOCOL["controls"]:
                ep = evdir / (control + "__" + eid + ".json")
                if not ep.exists():
                    text = raw if control == "full_context" else ""
                    ev = adapter.evidence_retained(
                        ex,
                        text,
                        set(),
                        source_token_spans=[(0, byid[eid]["source_tokens"])]
                        if control == "full_context"
                        else [],
                    )
                    save(
                        ep,
                        [
                            {
                                "budget": None,
                                "evidence": ev,
                                "selected_tokens": byid[eid]["source_tokens"]
                                if control == "full_context"
                                else 0,
                                "standalone_latency_sample": False,
                                "selector_wall_seconds": 0.0,
                                "seed_controls": [],
                            }
                        ],
                    )
            for method in PROTOCOL["methods"]:
                kind = "cpu" if method in ["head_tail", "bm25"] else "gpu"
                p = OUT / "selection" / kind / ds / method / (eid + ".json.gz")
                if not p.exists():
                    continue
                with gzip.open(p, "rt") as f:
                    record = json.load(f)
                assert record["source_sha256"] == digest(raw.encode())
                ep = evdir / (method + "__" + eid + ".json")
                if ep.exists():
                    continue
                evs = []
                for s in record["selections"]:
                    assert tok.count(s["filtered_text"]) == s["selected_tokens"] <= s["budget"]
                    if s["packing"] == "source_union_v1":
                        validate_source_union(raw, s["filtered_text"], s["source_token_spans"], tok)
                        ev = adapter.evidence_retained(
                            ex,
                            s["filtered_text"],
                            set(),
                            source_token_spans=s["source_token_spans"],
                        )
                        if ev:
                            ev = {**ev, "provenance_mode": "validated_source_union"}
                    else:
                        ev = score_selection(
                            adapter,
                            ex,
                            s["filtered_text"],
                            s["selected_chunk_ids"],
                            method=method,
                            metadata=s["metadata"],
                            budget=s["budget"],
                        )
                    evs.append(
                        {
                            "budget": s["budget"],
                            "evidence": ev,
                            "text_sha256": s["text_sha256"],
                            "selected_tokens": s["selected_tokens"],
                            "standalone_latency_sample": s["standalone_latency_sample"],
                            "selector_wall_seconds": s["selector_wall_seconds"],
                            "seed_controls": [
                                c for c in record["seed_controls"] if c["budget"] == s["budget"]
                            ],
                        }
                    )
                save(ep, evs)
        scores = {}
        for method in PROTOCOL["methods"] + PROTOCOL["controls"]:
            budgets = spec["budgets"] if method in PROTOCOL["methods"] else [None]
            for budget in budgets:
                eligible = [r for r in rows if budget is None or r["source_tokens"] > budget]
                items = []
                for row in eligible:
                    eid = row["example_id"]
                    acc = []
                    scores_native = []
                    reader_costs = []
                    native_inputs = []
                    truncated = 0
                    judged = 0
                    read = 0
                    out_of_range = 0
                    for seed in PROTOCOL["seeds"]:
                        link = links.get((method, budget, eid, seed))
                        correct = 0
                        score = 0
                        if link:
                            receipt = json.loads((OUT / link["receipt"]).read_text())
                            read += 1
                            jk = digest([ds, eid, receipt["raw_output"]])
                            jp = OUT / "judges/main" / (jk + ".json")
                            if jp.exists():
                                j = json.loads(jp.read_text())
                                correct = float(j["correct"])
                                score = j["score"]
                                judged += 1
                                out_of_range += bool(j.get("judge_out_of_range", False))
                            reader_costs.append(receipt["cost_usd"])
                            native_inputs.append(
                                receipt["response"]["usageMetadata"].get("promptTokenCount", 0)
                            )
                            truncated += receipt["finish_reason"] == "MAX_TOKENS"
                        acc.append(correct)
                        scores_native.append(score)
                    ep = evdir / (method + "__" + eid + ".json")
                    e = None
                    if ep.exists():
                        e = next(
                            (s for s in json.loads(ep.read_text()) if s["budget"] == budget), None
                        )
                    ll = latency_links.get((method, budget, eid))
                    timing = json.loads((OUT / ll["receipt"]).read_text()) if ll else None
                    s_times = (
                        (
                            [e["selector_wall_seconds"]]
                            + [c["selector_wall_seconds"] for c in e["seed_controls"]]
                        )
                        if e and e["standalone_latency_sample"]
                        else []
                    )
                    fresh = OUT / "cpu_standalone_latency" / ds / method / (eid + ".json")
                    if fresh.exists():
                        s_times = next(
                            x["wall_seconds"]
                            for x in json.loads(fresh.read_text())
                            if x["budget"] == budget
                        )
                    ev = e["evidence"] if e else None
                    item = {
                        "dataset": ds,
                        "method": method,
                        "budget": budget,
                        "example_id": eid,
                        "accuracy": float(np.mean(acc)),
                        "seed_accuracy": acc,
                        "mean_native_score": float(np.mean(scores_native)),
                        "reader_count": read,
                        "judged_count": judged,
                        "truncated_answers": truncated,
                        "reader_cost": float(np.mean(reader_costs)) if reader_costs else None,
                        "native_input_tokens": float(np.mean(native_inputs))
                        if native_inputs
                        else None,
                        "evidence": ev,
                        "selected_tokens": e["selected_tokens"]
                        if e
                        else (
                            row["source_tokens"]
                            if method == "full_context"
                            else 0
                            if method == "empty_context"
                            else None
                        ),
                        "selector_seconds": float(np.mean(s_times))
                        if s_times
                        else 0.0
                        if method in PROTOCOL["controls"]
                        else None,
                        "online_reader_seconds": timing["reader_latency_seconds"]
                        if timing
                        else None,
                        "ttft_seconds": timing["time_to_first_token_seconds"] if timing else None,
                    }
                    item["judge_out_of_range"] = out_of_range
                    items.append(item)
                    per_example.append(item)
                ci = interval(ds, eligible, [i["accuracy"] for i in items])
                scores[(method, budget)] = {i["example_id"]: i["accuracy"] for i in items}
                matched = [
                    i
                    for i in items
                    if i["selector_seconds"] is not None and i["online_reader_seconds"] is not None
                ]
                kind = "cpu" if method in ["head_tail", "bm25"] else "gpu"
                selector_mean = (
                    float(np.mean([i["selector_seconds"] for i in matched])) if matched else None
                )
                rc = [i["reader_cost"] for i in items if i["reader_cost"] is not None]
                ev_items = [
                    i
                    for i in items
                    if i["evidence"] is not None and i["evidence"].get("evaluable", True)
                ]
                ev_ci = (
                    interval(
                        ds,
                        [byid[i["example_id"]] for i in ev_items],
                        [i["evidence"]["fraction_retained"] for i in ev_items],
                    )
                    if ev_items
                    else None
                )
                cell = {
                    "dataset": ds,
                    "method": method,
                    "budget": budget,
                    "questions": len(items),
                    "expected_answers": 3 * len(items),
                    "recorded_answers": sum(i["reader_count"] for i in items),
                    "judged_answers": sum(i["judged_count"] for i in items),
                    "accuracy": ci,
                    "seed_accuracy": [
                        float(np.mean([i["seed_accuracy"][s] for i in items])) for s in range(3)
                    ],
                    "mean_native_score": float(np.mean([i["mean_native_score"] for i in items])),
                    "evidence_recall": ev_ci,
                    "evidence_metric": sorted({i["evidence"]["metric"] for i in ev_items}),
                    "mean_tokens_kept": float(np.mean([i["selected_tokens"] or 0 for i in items])),
                    "reader_cost_per_query": float(np.mean(rc)) if rc else None,
                    "standalone_selector_seconds": selector_mean,
                    "latency_sample_n": len(matched),
                    "end_to_end_mean_seconds": float(
                        np.mean(
                            [i["selector_seconds"] + i["online_reader_seconds"] for i in matched]
                        )
                    )
                    if matched
                    else None,
                    "end_to_end_p95_seconds": float(
                        np.quantile(
                            [i["selector_seconds"] + i["online_reader_seconds"] for i in matched],
                            0.95,
                        )
                    )
                    if matched
                    else None,
                    "ttft_mean_seconds": float(
                        np.mean(
                            [i["ttft_seconds"] for i in matched if i["ttft_seconds"] is not None]
                        )
                    )
                    if matched
                    else None,
                    "output_cap_hits": sum(i["truncated_answers"] for i in items),
                }
                cell["judge_out_of_range"] = sum(i["judge_out_of_range"] for i in items)
                cell["serving_cost_per_query"] = (
                    (cell["reader_cost_per_query"] + selector_mean * PRICES[kind] / 3600)
                    if rc and selector_mean is not None
                    else None
                )
                cells.append(cell)
        for budget in spec["budgets"]:
            elig = [r for r in rows if r["source_tokens"] > budget]
            for a, b in itertools.combinations(PROTOCOL["methods"], 2):
                av = scores[(a, budget)]
                bv = scores[(b, budget)]
                comparisons.append(
                    {
                        "dataset": ds,
                        "budget": budget,
                        "left": a,
                        "right": b,
                        **interval(
                            ds, elig, [av[r["example_id"]] - bv[r["example_id"]] for r in elig]
                        ),
                    }
                )
    # Curves compare the same questions at every budget, including full/empty controls.
    common_cells = []
    for ds, spec in PROTOCOL["datasets"].items():
        base = ROOT / "cora_bench/outputs/materialized" / ds
        cohort = sorted(
            [
                r
                for l in (base / "input.jsonl").open()
                if (r := json.loads(l))["source_tokens"] > max(spec["budgets"])
            ],
            key=lambda r: r["example_id"],
        )
        ids = {r["example_id"] for r in cohort}
        for c in [x for x in cells if x["dataset"] == ds]:
            items = [
                i
                for i in per_example
                if i["dataset"] == ds
                and i["method"] == c["method"]
                and i["budget"] == c["budget"]
                and i["example_id"] in ids
            ]
            cc = dict(c)
            cc.update(
                questions=len(items),
                expected_answers=3 * len(items),
                recorded_answers=sum(i["reader_count"] for i in items),
                judged_answers=sum(i["judged_count"] for i in items),
                accuracy=interval(ds, cohort, [i["accuracy"] for i in items]),
                cohort="source_tokens > maximum dataset budget",
            )
            cc["judge_out_of_range"] = sum(i["judge_out_of_range"] for i in items)
            rc = [i["reader_cost"] for i in items if i["reader_cost"] is not None]
            if rc and cc["standalone_selector_seconds"] is not None:
                kind = "cpu" if cc["method"] in ["head_tail", "bm25"] else "gpu"
                cc["serving_cost_per_query"] = (
                    float(np.mean(rc)) + cc["standalone_selector_seconds"] * PRICES[kind] / 3600
                )
            common_cells.append(cc)
    save(dest / "common_cohort_curves.json", common_cells)
    complete = all(c["judged_answers"] == c["expected_answers"] for c in cells)
    save(
        dest / "summary.json",
        {
            "complete": complete,
            "cells": cells,
            "notes": {
                "accuracy": "LOONG Perfect Rate; other datasets native correctness",
                "cost": "uncached native API list price + standalone warm selector wall time × full on-demand instance price; judge/development cost excluded",
                "latency": "matched 30-question subsample, concurrency 8; includes selector query encoding, data preprocessing and packing; cold model startup separate",
                "evidence": "provenance for extractive methods; verbatim survival for text modifiers; LOONG automatic candidate-line diagnostic, not reviewed gold support",
                "uncertainty": "2000 paired cluster bootstrap draws; three reader repetitions averaged within question, not treated as independent questions",
            },
            "instance_hourly_usd": PRICES,
        },
    )
    save(
        dest / "analysis_manifest.json",
        {
            "protocol_sha256": digest(PROTOCOL),
            "code": {
                p.name: digest(p.read_bytes())
                for p in [Path(__file__), Path(__file__).with_name("cpu_latency.py")]
            },
            "complete": complete,
        },
    )
    save(dest / "per_example.json", per_example)
    save(dest / "paired_differences.json", comparisons)
    lines = [
        "# CoRA-Bench reader follow-up",
        f"\nStatus: {'complete' if complete else 'PARTIAL — missing judgments count as zero; not final results'}.",
        "\n| Dataset | Method | Budget | Answers scored | Accuracy (95% CI) | Evidence recall | Tokens kept | Serving $/query | E2E seconds |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in cells:
        ci = c["accuracy"]
        lo, hi = ci["ci95"]
        ev = c["evidence_recall"]
        cost = c["serving_cost_per_query"]
        lat = c["end_to_end_mean_seconds"]
        lines.append(
            f"| {c['dataset']} | {c['method']} | {c['budget'] or 'control'} | {c['judged_answers']}/{c['expected_answers']} | {ci['estimate']:.3f} ({lo:.3f}–{hi:.3f}) | {ev['estimate'] if ev else 'N/A'} | {c['mean_tokens_kept']:.0f} | {f'{cost:.5f}' if cost is not None else 'pending'} | {f'{lat:.2f}' if lat is not None else 'pending'} |"
        )
    lines.append(
        "\nLOONG scores use the pinned upstream numeric parser without clipping. "
        + str(sum(c["judge_out_of_range"] for c in cells))
        + " logical judgments are outside the prompt's requested 1–100 range; counts appear in the JSON tables. "
        + "Unparseable judgments remain unresolved and prevent a complete report."
    )
    (dest / "RESULTS.md").write_text("\n".join(lines) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for ds in PROTOCOL["datasets"]:
        for axis, label in [
            ("serving_cost_per_query", "Serving cost per query (USD)"),
            ("end_to_end_mean_seconds", "Mean selection + reader latency (seconds)"),
        ]:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            plotted = []
            for method in PROTOCOL["methods"] + PROTOCOL["controls"]:
                cs = [
                    c
                    for c in common_cells
                    if c["dataset"] == ds
                    and c["method"] == method
                    and c[axis] is not None
                    and c["judged_answers"] == c["expected_answers"]
                ]
                if not cs:
                    continue
                cs.sort(key=lambda c: c[axis])
                x = [c[axis] for c in cs]
                y = [c["accuracy"]["estimate"] for c in cs]
                err = [
                    [max(0, y[i] - c["accuracy"]["ci95"][0]) for i, c in enumerate(cs)],
                    [max(0, c["accuracy"]["ci95"][1] - y[i]) for i, c in enumerate(cs)],
                ]
                ax.errorbar(x, y, yerr=err, marker="o", capsize=3, label=method)
                plotted.extend(zip(x, y))
            if plotted:
                frontier = []
                best = -1
                for x, y in sorted(plotted, key=lambda p: (p[0], -p[1])):
                    if y > best:
                        frontier.append((x, y))
                        best = y
                ax.plot(
                    *zip(*frontier),
                    color="black",
                    linestyle="--",
                    alpha=0.4,
                    label="Pareto frontier",
                )
                ax.legend(fontsize=8)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Awaiting complete scored cells and latency pass",
                    ha="center",
                    transform=ax.transAxes,
                )
            ax.set(
                xlabel=label,
                ylabel="Perfect Rate" if ds == "loong" else "Native answer accuracy",
                title=ds + " (common cohort)",
                ylim=(-0.02, 1.02),
            )
            fig.tight_layout()
            fig.savefig(dest / f"{ds}_{axis}.pdf")
            fig.savefig(dest / f"{ds}_{axis}.png", dpi=180)
            plt.close(fig)


if __name__ == "__main__":
    run()
