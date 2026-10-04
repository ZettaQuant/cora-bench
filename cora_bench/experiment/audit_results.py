"""Read-only audit of saved reader outputs, judgments and sampled contexts, with a provisional
accuracy snapshot. Headline estimates use only questions scored on all three seeds."""

import ast
import gzip
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from cora_bench.budgets.token_budget import Tokenizer

from .api import OUT, PROTOCOL, ROOT, digest, save, usage_cost
from .packing import validate_source_union
from .prompts import parse, request
from .score_answers import judge_request
from .summarize_results import interval

HERE = Path(__file__).parent


def reference_functions():
    source = HERE / "audit_reference/babilong_metrics.py"
    ns = {}
    tree = ast.parse(source.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Assign))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), ns)  # noqa: S102
    p = HERE / "upstream/nolima/evaluation__async_evaluate.py"
    node = next(
        n
        for n in ast.walk(ast.parse(p.read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == "_evaluate_response"
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(p), "exec"), ns)  # noqa: S102
    p = HERE / "upstream/loong/src__utils__metric.py"
    node = next(
        n
        for n in ast.walk(ast.parse(p.read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == "extract_number"
    )
    import re

    ns["re"] = re
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(p), "exec"), ns)  # noqa: S102
    return ns


def summarize(rows, scored, seeds):
    """Average correctness per question, keeping only questions scored on every seed."""
    complete = {
        r["example_id"]: [scored[(r["example_id"], s)] for s in seeds]
        for r in rows
        if all((r["example_id"], s) in scored for s in seeds)
    }
    values = {
        eid: sum(float(j["correct"]) for j in js) / len(seeds) for eid, js in complete.items()
    }
    return complete, values


def run():
    started = time.time()
    dest = OUT / "interim_audits" / str(int(started))
    dest.mkdir(parents=True, exist_ok=False)
    checks = Counter()
    errors = []

    def check(name, condition, detail=None):
        checks[name] += 1
        if not condition:
            errors.append({"check": name, "detail": detail})

    refs = reference_functions()
    sources = json.loads((HERE / "audit_reference/sources.json").read_text())
    check(
        "official_babilong_reference_hash",
        digest((HERE / "audit_reference/babilong_metrics.py").read_bytes())
        == sources["babilong_metrics"]["sha256"],
    )
    rows = {}
    for ds, spec in PROTOCOL["datasets"].items():
        path = ROOT / "cora_bench/outputs/materialized" / ds / "input.jsonl"
        check(
            "frozen_input_hash",
            digest(path.read_bytes()) == sources["materialized_input_sha256"][ds],
            ds,
        )
        values = [json.loads(line) for line in path.open()]
        rows[ds] = {r["example_id"]: r for r in values}
        check("frozen_cohort_size", len(values) == len(rows[ds]) == spec["n"], ds)
    manifest = json.loads((OUT / "reader/main/manifest.json").read_text())
    check("reader_protocol_hash", manifest["protocol_sha256"] == digest(PROTOCOL))
    for name, expected in manifest["code"].items():
        check("reader_code_hash", digest((HERE / name).read_bytes()) == expected, name)
    for name, expected in manifest["upstream"].items():
        check("pinned_upstream_hash", digest((HERE / name).read_bytes()) == expected, name)
    for kind in ["cpu", "gpu"]:
        mp = OUT / "selection" / kind / "manifest.json"
        if mp.exists():
            m = json.loads(mp.read_text())
            check("selector_protocol_hash", m["protocol_sha256"] == digest(PROTOCOL), kind)
            for name, expected in m["code"].items():
                check("selector_code_hash", digest((ROOT / name).read_bytes()) == expected, name)

    # Snapshot the file lists so outputs written while the audit runs are excluded.
    link_paths = sorted((OUT / "reader/main/links").glob("*/*/*.json"))
    judge_paths = {
        p.stem: p for p in (OUT / "judges/main").glob("*.json") if p.name != "status.json"
    }
    receipts, judges = {}, {}
    seen_links = set()
    receipt_seeds = defaultdict(set)
    context_ids = defaultdict(set)
    observed = defaultdict(dict)
    scored = defaultdict(dict)
    sample_candidates = defaultdict(dict)
    native_versions, finish_reasons = Counter(), Counter()
    judge_checked = set()
    manual_candidates = defaultdict(list)
    for p in link_paths:
        link = json.loads(p.read_text())
        ds, eid, method, budget, seed = (
            link[k] for k in ["dataset", "example_id", "method", "budget", "seed"]
        )
        row = rows[ds][eid]
        identity = (ds, eid, method, budget, seed)
        check("unique_logical_link", identity not in seen_links, identity)
        seen_links.add(identity)
        check("logical_filename", p.stem == digest(list(identity)), str(p.relative_to(OUT)))
        check("authorized_seed", seed in PROTOCOL["seeds"], identity)
        check(
            "method_and_budget",
            (method in PROTOCOL["controls"] and budget is None)
            or (
                method in PROTOCOL["methods"]
                and budget in PROTOCOL["datasets"][ds]["budgets"]
                and row["source_tokens"] > budget
            ),
            identity,
        )
        check(
            "source_token_identity", link["source_tokens_cl100k"] == row["source_tokens"], identity
        )
        check(
            "recorded_budget_cap",
            budget is None or link["selected_tokens_cl100k"] <= budget,
            identity,
        )
        rp = link["receipt"]
        if rp not in receipts:
            receipts[rp] = json.loads((OUT / rp).read_text())
            receipt = receipts[rp]
            check("receipt_key_path", Path(rp).stem == receipt["key"], rp)
            check("reader_model", receipt["model"] == PROTOCOL["models"]["reader"], rp)
            raw, answer, finish = parse(receipt["response"])
            check(
                "reader_response_parse",
                (raw, answer, finish)
                == (receipt["raw_output"], receipt["answer"], receipt["finish_reason"]),
                rp,
            )
            check(
                "native_usage_cost",
                abs(usage_cost(receipt["response"]) - receipt["cost_usd"]) < 1e-9,
                rp,
            )
            usage = receipt["response"].get("usageMetadata", {})
            check(
                "native_usage_fields_present",
                isinstance(usage.get("promptTokenCount"), int)
                and isinstance(usage.get("totalTokenCount"), int)
                and (not raw or isinstance(usage.get("candidatesTokenCount"), int)),
                rp,
            )
            check("native_finish_present", finish is not None, rp)
            check("native_window_cap", receipt["native_count_estimate"] + 1024 <= 1048576, rp)
            native_versions[receipt.get("model_version")] += 1
            finish_reasons[receipt["finish_reason"]] += 1
        receipt = receipts[rp]
        check("receipt_seed_identity", receipt["seed"] == seed, identity)
        receipt_seeds[rp].add(seed)
        context_ids[(ds, eid, method, budget)].add(
            (link["context_sha256"], link["selected_tokens_cl100k"])
        )
        cell = (ds, method, budget)
        observed[cell][(eid, seed)] = (link, receipt)
        sample_candidates[cell][eid] = link
        jk = digest([ds, eid, receipt["raw_output"]])
        if jk not in judge_paths:
            continue
        if jk not in judges:
            judges[jk] = json.loads(judge_paths[jk].read_text())
        j = judges[jk]
        if jk not in judge_checked:
            judge_checked.add(jk)
            check("judge_key_identity", j["key"] == jk, jk)
            failure = not receipt["raw_output"] or receipt["finish_reason"] not in [
                "STOP",
                "MAX_TOKENS",
            ]
            if failure:
                expected = 0.0
                check("reader_failure_visible", j["metric"] == "reader_failure", jk)
            elif ds == "babilong":
                expected = float(
                    refs["compare_answers"](
                        str(row["answer"]),
                        receipt["raw_output"],
                        row["query"],
                        refs["TASK_LABELS"][row["task"]],
                    )
                )
            elif ds == "nolima":
                expected = float(
                    refs["_evaluate_response"](
                        SimpleNamespace(metric="contains"),
                        receipt["raw_output"],
                        [str(row["answer"])],
                    )
                )
            else:
                body = judge_request(ds, row, receipt["raw_output"])
                check("judge_exact_request_identity", j["request_sha256"] == digest(body), jk)
                check("judge_model_identity", j["judge_model"] == body["model"], jk)
                content = j["response"]["choices"][0]["message"]["content"]
                expected = (
                    refs["extract_number"](content)
                    if ds == "loong"
                    else float("yes" in content.strip().lower())
                )
                check("judge_native_parse_exists", expected is not None, jk)
            check("native_score_recomputed", expected == j["score"], jk)
            check(
                "native_correct_recomputed",
                bool(j["correct"])
                == (expected == 100 if ds == "loong" and not failure else expected == 1),
                jk,
            )
            manual_candidates[(ds, bool(j["correct"]))].append((digest(jk), row, receipt, j, link))
        scored[cell][(eid, seed)] = j
    for rp, seeds in receipt_seeds.items():
        check("no_cross_seed_reuse", len(seeds) == 1, rp)
    for key, ids in context_ids.items():
        check("same_context_across_reader_seeds", len(ids) == 1, key)
    print(
        json.dumps(
            {
                "stage": "all_links_and_scores_checked",
                "links": len(link_paths),
                "unique_receipts": len(receipts),
                "errors": len(errors),
            }
        ),
        flush=True,
    )

    # Rebuild contexts for up to three hash-selected examples per observed cell.
    tok = Tokenizer()
    sampled = []
    for (ds, method, budget), candidates in sorted(
        sample_candidates.items(), key=lambda x: str(x[0])
    ):
        for eid in sorted(candidates, key=digest)[:3]:
            link = candidates[eid]
            row = rows[ds][eid]
            source = (
                (ROOT / "cora_bench/outputs/materialized" / ds / "ctx" / (eid + ".txt"))
                .read_bytes()
                .decode("utf8")
            )
            if method in PROTOCOL["controls"]:
                text = source if method == "full_context" else ""
            else:
                with gzip.open(OUT / link["selection_ref"], "rt") as f:
                    record = json.load(f)
                check(
                    "sample_selection_identity",
                    (record["dataset"], record["example_id"], record["method"])
                    == (ds, eid, method),
                    link["selection_ref"],
                )
                check("sample_selection_row", record["row"] == row, link["selection_ref"])
                check(
                    "sample_source_hash",
                    record["source_sha256"] == digest(source.encode()),
                    link["selection_ref"],
                )
                check(
                    "sample_query_hash",
                    record["query_sha256"] == digest(row["query"]),
                    link["selection_ref"],
                )
                selection = next(s for s in record["selections"] if s["budget"] == budget)
                text = selection["filtered_text"]
                check(
                    "sample_selected_hash",
                    selection["text_sha256"] == digest(text.encode()),
                    link["selection_ref"],
                )
                if selection["packing"] == "source_union_v1":
                    try:
                        validate_source_union(source, text, selection["source_token_spans"], tok)
                        check("sample_source_union", True)
                    except (ValueError, AssertionError) as exc:
                        check("sample_source_union", False, str(exc))
            check(
                "sample_context_hash",
                link["context_sha256"] == digest(text.encode()),
                [ds, eid, method, budget],
            )
            check(
                "sample_cl100k_count",
                tok.count(text) == link["selected_tokens_cl100k"],
                [ds, eid, method, budget],
            )
            for seed in PROTOCOL["seeds"]:
                item = observed[(ds, method, budget)].get((eid, seed))
                if not item:
                    continue
                rec = item[1]
                payload = request(ds, row, text, seed)
                check(
                    "sample_reader_payload_hash",
                    digest(payload) == rec["request_sha256"],
                    [ds, eid, method, budget, seed],
                )
                check(
                    "sample_reader_cache_identity",
                    digest({"phase": "main", "model": rec["model"], "request": payload})
                    == rec["key"],
                    rec["key"],
                )
                check(
                    "sample_gold_and_method_not_injected",
                    payload
                    == request(
                        ds,
                        {**row, "answer": "AUDIT_SENTINEL_GOLD", "method": "AUDIT_SENTINEL_METHOD"},
                        text,
                        seed,
                    ),
                )
            sampled.append({"dataset": ds, "example_id": eid, "method": method, "budget": budget})
    print(
        json.dumps(
            {"stage": "sampled_contexts_checked", "contexts": len(sampled), "errors": len(errors)}
        ),
        flush=True,
    )

    cells, values_by_cell = [], {}
    for ds, spec in PROTOCOL["datasets"].items():
        for method in PROTOCOL["controls"] + PROTOCOL["methods"]:
            for budget in [None] if method in PROTOCOL["controls"] else spec["budgets"]:
                eligible = sorted(
                    [r for r in rows[ds].values() if budget is None or r["source_tokens"] > budget],
                    key=lambda r: r["example_id"],
                )
                key = (ds, method, budget)
                complete_q, values = summarize(eligible, scored[key], PROTOCOL["seeds"])
                complete = len(complete_q) == len(eligible)
                cell = {
                    "dataset": ds,
                    "method": method,
                    "budget": budget,
                    "eligible_questions": len(eligible),
                    "expected_outputs": 3 * len(eligible),
                    "reader_outputs": len(observed[key]),
                    "scored_outputs": len(scored[key]),
                    "complete_three_seed_questions": len(complete_q),
                    "complete": complete,
                    "observed_accuracy_complete_questions": sum(values.values()) / len(values)
                    if values
                    else None,
                    "accuracy": interval(ds, eligible, [values[r["example_id"]] for r in eligible])
                    if complete
                    else None,
                    "seed_accuracy": [
                        sum(float(scored[key][(r["example_id"], s)]["correct"]) for r in eligible)
                        / len(eligible)
                        for s in PROTOCOL["seeds"]
                    ]
                    if complete
                    else None,
                    "mean_native_score": sum(j["score"] for js in complete_q.values() for j in js)
                    / (3 * len(complete_q))
                    if complete_q
                    else None,
                    "output_cap_hits": sum(
                        r["finish_reason"] == "MAX_TOKENS" for _, r in observed[key].values()
                    ),
                    "empty_outputs": sum(not r["raw_output"] for _, r in observed[key].values()),
                    "judge_out_of_range": sum(
                        bool(j.get("judge_out_of_range")) for j in scored[key].values()
                    ),
                }
                if complete and ds in ["babilong", "longmemeval"]:
                    sub = defaultdict(list)
                    for row in eligible:
                        group = (
                            row["task"]
                            if ds == "babilong"
                            else ("abstention" if row["abstention"] else "answerable")
                        )
                        sub[group].append(values[row["example_id"]])
                    cell["subgroups"] = {
                        k: {"questions": len(v), "accuracy": sum(v) / len(v)}
                        for k, v in sub.items()
                    }
                cells.append(cell)
                values_by_cell[key] = values
    paired = []
    for cell in cells:
        if not cell["complete"] or cell["method"] in PROTOCOL["controls"]:
            continue
        ds, method, budget = (cell[k] for k in ["dataset", "method", "budget"])
        eligible = sorted(
            [r for r in rows[ds].values() if r["source_tokens"] > budget],
            key=lambda r: r["example_id"],
        )
        full = values_by_cell[(ds, "full_context", None)]
        selected = values_by_cell[(ds, method, budget)]
        if all(r["example_id"] in full for r in eligible):
            delta = [selected[r["example_id"]] - full[r["example_id"]] for r in eligible]
            paired.append(
                {
                    "dataset": ds,
                    "method": method,
                    "budget": budget,
                    "questions": len(eligible),
                    "delta_vs_full": interval(ds, eligible, delta),
                }
            )
    examples = []
    for (ds, correct), candidates in sorted(manual_candidates.items()):
        for _, row, rec, judgment, link in sorted(candidates, key=lambda x: x[0])[:2]:
            examples.append(
                {
                    "dataset": ds,
                    "example_id": row["example_id"],
                    "method": link["method"],
                    "budget": link["budget"],
                    "question": row["query"],
                    "gold": row["answer"],
                    "reader_output": rec["raw_output"],
                    "correct": correct,
                    "score": judgment["score"],
                    "finish_reason": rec["finish_reason"],
                    "judge_output": judgment.get("response", {})
                    .get("choices", [{}])[0]
                    .get("message", {})
                    .get("content"),
                }
            )
    summary = {
        "started_unix": started,
        "finished_unix": time.time(),
        "provisional": True,
        "reader": PROTOCOL["models"]["reader"],
        "logical_outputs": len(link_paths),
        "scored_logical_outputs": sum(len(x) for x in scored.values()),
        "unique_reader_receipts": len(receipts),
        "unique_scored_answers": len(judge_checked),
        "checks": dict(checks),
        "errors": errors,
        "sampled_contexts": sampled,
        "native_model_versions": dict(native_versions),
        "unique_receipt_finish_reasons": dict(finish_reasons),
        "cells": cells,
        "paired_vs_full": paired,
        "notes": [
            "Headline accuracy requires all questions and all three seeds in that cell.",
            "95% intervals bootstrap questions, BABILong stratified by task, NoLiMa clustered by needle family; seeds averaged within question.",
            "LOONG uses Perfect Rate exactly 100; GPT-5.5 is an adapted judge, not the original paper model.",
            "Source/context/payload reconstruction is a prespecified sample; all logical identities and scored outputs are checked.",
            "No APIs called, no live results changed. Human correctness of all LLM judgments is not guaranteed.",
        ],
    }
    save(dest / "summary.json", summary)
    save(dest / "manual_review_samples.json", examples)
    save(
        dest / "manifest.json",
        {
            "audit_code_sha256": digest(Path(__file__).read_bytes()),
            "reference_sources": sources,
            "protocol_sha256": digest(PROTOCOL),
        },
    )
    lines = [
        "# Provisional reader performance and audit",
        "",
        f"Snapshot: {len(link_paths):,} reader outputs; {summary['scored_logical_outputs']:,} scored. Audit failures: {len(errors)}.",
        "",
        "| Dataset | Method | Budget | Scored / planned | Complete questions | Accuracy, 95% CI | Mean native score | Output cap hits |",
        "|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for c in cells:
        ci = c["accuracy"]
        value = (
            f"{100 * ci['estimate']:.2f}% ({100 * ci['ci95'][0]:.2f}–{100 * ci['ci95'][1]:.2f})"
            if ci
            else "pending"
        )
        score = f"{c['mean_native_score']:.2f}" if c["complete"] else "pending"
        lines.append(
            f"| {c['dataset']} | {c['method']} | {c['budget'] or 'control'} | {c['scored_outputs']}/{c['expected_outputs']} | {c['complete_three_seed_questions']}/{c['eligible_questions']} | {value} | {score} | {c['output_cap_hits']} |"
        )
    lines += ["", *summary["notes"]]
    (dest / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "artifact": str(dest),
                "audit_errors": errors,
                "logical_outputs": len(link_paths),
                "complete_cells": [c for c in cells if c["complete"]],
                "paired_vs_full": paired,
                "unique_receipt_finish_reasons": dict(finish_reasons),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    run()
