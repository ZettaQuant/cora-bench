"""Render LaTeX paper tables from an audited snapshot; incomplete cells are shown as dashes.

Usage: python -m cora_bench.analysis.paper_tables --snapshot PATH --output DIR
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

DATASETS = {
    "babilong": "BABILong",
    "loong": "LOONG",
    "nolima": "NoLiMa",
    "longmemeval": "LongMemEval",
}
METHODS = {
    "full_context": "Full context",
    "empty_context": "Empty context",
    "head_tail": "Head and tail",
    "bm25": "BM25",
    "dense": "Dense",
    "dense_rerank": "Dense+Rerank",
    "llmlingua2": "LLMLingua-2",
    "provence": "Provence",
}


def value(cell, interval=True):
    if not cell["complete"]:
        return "--"
    ci = cell["accuracy"]
    point = f"{100 * ci['estimate']:.2f}"
    return point + (f" [{100 * ci['ci95'][0]:.2f}, {100 * ci['ci95'][1]:.2f}]" if interval else "")


def render(snapshot, output):
    raw = Path(snapshot).read_bytes()
    summary = json.loads(raw)
    if summary["errors"]:
        raise ValueError("Refusing to publish a snapshot with failed audit checks")
    cells = summary["cells"]
    for cell in cells:
        if cell["complete"]:
            assert cell["scored_outputs"] == cell["expected_outputs"]
            assert cell["complete_three_seed_questions"] == cell["eligible_questions"]
            assert len(cell["seed_accuracy"]) == 3
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromtimestamp(summary["started_unix"], timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )
    complete = sum(c["complete"] for c in cells)
    header = "% Generated from audited aggregate data; do not edit values by hand.\n"
    meta = (
        rf"\newcommand{{\ReaderSnapshotDate}}{{{stamp}}}"
        + "\n"
        + rf"\newcommand{{\ReaderCompleteCells}}{{{complete}}}"
        + "\n"
        + rf"\newcommand{{\ReaderTotalCells}}{{{len(cells)}}}"
        + "\n"
        + rf"\newcommand{{\ReaderRecorded}}{{{summary['logical_outputs']:,}}}"
        + "\n"
        + rf"\newcommand{{\ReaderScored}}{{{summary['scored_logical_outputs']:,}}}"
        + "\n"
        + rf"\newcommand{{\ReaderAuditContexts}}{{{len(summary['sampled_contexts'])}}}"
        + "\n"
    )
    (output / "reader_snapshot.tex").write_text(header + meta)
    lines = [
        header,
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Reader accuracy (\%) at a 32K context budget under overlap-union packing, averaged over seeds 5768, 78516, and 944601. LOONG uses Perfect Rate; NoLiMa is the custom diagnostic. Full and empty controls use all questions without a context budget. A dash denotes a configuration without complete three-seed scoring. Confidence intervals and all budgets are reported in Appendix~\ref{app:reader-matrix}.}",
        r"\label{tab:reader-32k}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & BABILong & LOONG & NoLiMa & LongMemEval \\",
        r"\midrule",
    ]
    for method, name in METHODS.items():
        row = []
        for ds in DATASETS:
            cell = next(
                c
                for c in cells
                if c["dataset"] == ds
                and c["method"] == method
                and c["budget"] == (None if method in ["full_context", "empty_context"] else 32000)
            )
            row.append(value(cell, interval=False))
        lines.append(name + " & " + " & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (output / "reader_32k.tex").write_text("\n".join(lines) + "\n")
    for ds, name in DATASETS.items():
        lines = [
            header,
            r"\begin{table}[p]",
            r"\centering",
            r"\small",
            r"\setlength{\tabcolsep}{3pt}",
            rf"\caption{{{name}: reader evaluation under overlap-union packing. Accuracy is a percentage; brackets are 95\% paired-bootstrap intervals over questions after averaging three reader runs. The seed range is descriptive, not a confidence interval. Coverage counts questions scored at all three seeds; cap hits count recorded outputs. Dashes denote unavailable full-cohort estimates, which are excluded from comparisons.}}",
            rf"\label{{tab:reader-{ds}}}",
            r"\begin{tabular}{llrrrr}",
            r"\toprule",
            r"Method & Budget & Coverage & Accuracy [95\% CI] & Seed min--max & Cap hits \\",
            r"\midrule",
        ]
        for method, mname in METHODS.items():
            for c in sorted(
                [c for c in cells if c["dataset"] == ds and c["method"] == method],
                key=lambda c: c["budget"] or 0,
            ):
                budget = str(c["budget"] // 1000) + "K" if c["budget"] else "--"
                spread = (
                    f"{100 * min(c['seed_accuracy']):.2f}--{100 * max(c['seed_accuracy']):.2f}"
                    if c["complete"]
                    else "--"
                )
                lines.append(
                    f"{mname} & {budget} & {c['complete_three_seed_questions']}/{c['eligible_questions']} & {value(c)} & {spread} & {c['output_cap_hits']}"
                    + r" \\"
                )
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        (output / f"reader_{ds}.tex").write_text("\n".join(lines) + "\n")
    # LOONG rubric scores are complementary to, not interchangeable with, Perfect Rate.
    lines = [
        header,
        r"\begin{table}[p]",
        r"\centering",
        r"\small",
        r"\caption{LOONG complementary mean rubric scores (0--100 scale as returned by the native parser) at each budget. Full and empty controls use all 75 questions. Scores outside the requested 1--100 rubric range are preserved and flagged, not clipped.}",
        r"\label{tab:reader-loong-scores}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Method & 32K & 64K & 128K \\",
        r"\midrule",
    ]
    for method, name in METHODS.items():
        vals = []
        for b in [32000, 64000, 128000]:
            c = next(
                c
                for c in cells
                if c["dataset"] == "loong"
                and c["method"] == method
                and c["budget"] == (None if method in ["full_context", "empty_context"] else b)
            )
            vals.append(f"{c['mean_native_score']:.2f}" if c["complete"] else "--")
        lines.append(name + " & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (output / "reader_loong_scores.tex").write_text("\n".join(lines) + "\n")
    (output / "reader_tables_manifest.json").write_text(
        json.dumps(
            {
                "snapshot_sha256": hashlib.sha256(raw).hexdigest(),
                "snapshot_unix": summary["started_unix"],
                "complete_cells": complete,
                "total_cells": len(cells),
                "policy": "Scores require the complete eligible cohort and all three seeds; no partial-cohort accuracy is printed.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    render(args.snapshot, args.output)
