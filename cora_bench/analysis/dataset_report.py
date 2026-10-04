"""Rescore stored selections for evidence retention, with coverage and provenance checks.

Selections are indexed by file offset so each source is loaded once without holding all text.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from cora_bench.budgets.token_budget import Tokenizer
from cora_bench.data.registry import get_dataset
from cora_bench.evaluation.selection import SCORING_VERSION, score_selection
from cora_bench.instrumentation.manifest import build_manifest
from cora_bench.utils.io import write_json


def failed(row):
    return bool(
        row.get("error")
        or row.get("error_type")
        or (row.get("metadata") or {}).get("n_unscored", 0)
    )


def selection_digest(row):
    payload = {k: row.get(k) for k in ("filtered_text", "selected_chunk_ids", "config_fingerprint")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def index_selections(paths, repair_paths=()):
    """Reject conflicting successes; deduplicate identical outputs across all files."""
    index, groups, fingerprints = {}, set(), defaultdict(set)
    for path in sorted(set(map(Path, paths))):
        with path.open("rb") as fh:
            while True:
                offset = fh.tell()
                line = fh.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                group = (row["method"], row.get("budget"))
                groups.add(group)
                fingerprints[group].add(row.get("config_fingerprint", "legacy_unfingerprinted"))
                key = (*group, row["example_id"])
                digest, bad = selection_digest(row), failed(row)
                previous = index.get(key)
                if previous and not previous[3] and not bad and previous[2] != digest:
                    raise ValueError(
                        f"Conflicting successful selections for {key}; choose one version"
                    )
                if previous is None or (previous[3] and not bad):
                    index[key] = (path, offset, digest, bad)
    for group, fps in fingerprints.items():
        if len(fps) > 1:
            raise ValueError(f"Mixed run fingerprints for {group}: {fps}")
    # Explicit recovery is the only allowed cross-version override, and only of a failure.
    for path in sorted(set(map(Path, repair_paths))):
        with path.open("rb") as fh:
            while True:
                offset = fh.tell()
                line = fh.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                if failed(row):
                    continue
                key = (row["method"], row.get("budget"), row["example_id"])
                previous = index.get(key)
                if previous is None or not previous[3]:
                    raise ValueError(f"Repair may only replace an existing failure: {key}")
                if Path(row.get("repair_of", "")).resolve() != previous[0].resolve():
                    raise ValueError(f"Repair source mismatch: {key}")
                if not row.get("source_sha256") or not row.get("config_fingerprint"):
                    raise ValueError("Repair requires source and configuration fingerprints")
                index[key] = (path, offset, selection_digest(row), False)
    return index, sorted(groups, key=lambda g: (g[0], g[1] or float("inf")))


def read_indexed(entry):
    with entry[0].open("rb") as fh:
        fh.seek(entry[1])
        return json.loads(fh.readline())


def verified_legacy_ids(adapter, example, row):
    """Rebase legacy chunk IDs from another example only when that source text is identical."""
    ids = row.get("selected_chunk_ids") or []
    if row.get("config_fingerprint") or not ids:
        return ids, False
    foreign = {
        cid.rsplit("::", 1)[0] for cid in ids if cid.rsplit("::", 1)[0] != example.example_id
    }
    if not foreign or not hasattr(adapter, "dir"):
        return ids, False
    for source_id in foreign:
        if Path(source_id).name != source_id:
            return ids, False
        source = adapter.dir / "ctx" / f"{source_id}.txt"
        if not source.exists() or source.read_text() != example.context:
            return ids, False
    return [example.example_id + "::" + cid.rsplit("::", 1)[1] for cid in ids], True


def rescore(
    dataset,
    paths,
    output,
    *,
    data_dir=None,
    chunk_size=512,
    overlap=64,
    min_source_tokens=0,
    repair_paths=(),
):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    index, groups = index_selections(paths, repair_paths)
    adapter = get_dataset(dataset, **({"data_dir": data_dir} if data_dir else {}))
    tok = Tokenizer()
    manifest = build_manifest(
        {
            "dataset": dataset,
            "data_dir": data_dir,
            "selection_files": [str(p) for p in paths],
            "repair_files": [str(p) for p in repair_paths],
            "chunk_size": chunk_size,
            "overlap": overlap,
            "min_source_tokens": min_source_tokens,
            "scoring_version": SCORING_VERSION,
            "legacy_policy": "allow_unfingerprinted_and_validate_text",
            "cohort_policy": "source_exceeds_budget",
            "failure_policy": "report_coverage_and_zero_failure_recall",
        }
    )
    source_hashes = {}
    agg, seen_ids = defaultdict(list), set()
    per_example = output.with_suffix(".examples.jsonl")
    with per_example.open("w") as fh:
        for number, example in enumerate(adapter.load(), 1):
            eid = example.example_id
            seen_ids.add(eid)
            source_hash = hashlib.sha256(example.context.encode()).hexdigest()
            source_hashes[eid] = source_hash
            source_tokens = tok.count(example.context)
            if source_tokens <= min_source_tokens:
                continue
            answerable = not example.metadata.get("abstention", False)
            evidence_evaluable = answerable
            if dataset == "loong" and not example.gold_evidence:
                from cora_bench.evaluation.loong_evidence import candidate_lines

                evidence_evaluable = bool(candidate_lines(example.answer, example.context))
            for method, budget in groups:
                if budget is not None and budget >= source_tokens:
                    continue
                record = {
                    "dataset": dataset,
                    "example_id": eid,
                    "method": method,
                    "budget": budget,
                    "source_tokens": source_tokens,
                    "answerable": answerable,
                    "evidence_evaluable": evidence_evaluable,
                    "scoring_version": SCORING_VERSION,
                    "error": None,
                    "strata": {
                        k: example.metadata[k]
                        for k in (
                            "task",
                            "question_type",
                            "hop",
                            "depth",
                            "needle_id",
                            "test_id",
                            "haystack",
                            "set",
                        )
                        if k in example.metadata
                    },
                }
                entry = index.get((method, budget, eid))
                try:
                    if entry is None:
                        raise ValueError("missing_selection")
                    row = read_indexed(entry)
                    if failed(row):
                        raise ValueError(
                            row.get("error")
                            or row.get("error_type")
                            or f"partial_scoring: {(row.get('metadata') or {}).get('n_unscored')} chunks"
                        )
                    text = row.get("filtered_text", "")
                    selected_tokens = tok.count(text)
                    if budget is not None and selected_tokens > budget:
                        raise ValueError("budget_exceeded")
                    if row.get("source_tokens") != source_tokens:
                        raise ValueError("source_token_count_mismatch")
                    if row.get("source_sha256", source_hash) != source_hash:
                        raise ValueError("source_hash_mismatch")
                    if (
                        row.get("query_sha256", hashlib.sha256(example.query.encode()).hexdigest())
                        != hashlib.sha256(example.query.encode()).hexdigest()
                    ):
                        raise ValueError("query_hash_mismatch")
                    ids, rebased = verified_legacy_ids(adapter, example, row)
                    ev = score_selection(
                        adapter,
                        example,
                        text,
                        ids,
                        method=method,
                        metadata=row.get("metadata"),
                        budget=budget,
                        chunk_size=chunk_size,
                        overlap=overlap,
                    )
                    if ev and ev.get("evaluable") is False:
                        raise ValueError("no_candidate_evidence_requires_review")
                    record.update(
                        selected_tokens=selected_tokens,
                        compression_ratio=selected_tokens / source_tokens,
                        budget_utilization=selected_tokens / budget if budget else None,
                        evidence=ev,
                        config_fingerprint=row.get("config_fingerprint"),
                        legacy_provenance_rebased=rebased,
                        repair_of=row.get("repair_of"),
                        count_corrected=row.get("selected_tokens") != selected_tokens,
                    )
                except (ValueError, KeyError) as exc:
                    record["error"] = str(exc)
                fh.write(json.dumps(record) + "\n")
                if answerable:
                    agg[(method, budget)].append(record)
            if number % 50 == 0:
                print(f"{dataset}: scored {number} examples", flush=True)
    unknown = {key[2] for key in index} - seen_ids
    if unknown:
        raise ValueError(f"Selection IDs absent from dataset: {sorted(unknown)[:10]}")
    summaries = []
    for (method, budget), rows in agg.items():
        evaluable = [r for r in rows if r["evidence_evaluable"]]
        ok = [r for r in evaluable if not r["error"] and r.get("evidence") is not None]
        avg = (
            lambda key: mean(r[key] for r in ok if r.get(key) is not None)
            if any(r.get(key) is not None for r in ok)
            else None
        )
        summaries.append(
            {
                "dataset": dataset,
                "method": method,
                "budget": "inf" if budget is None else f"{budget // 1000}K",
                "n_expected": len(rows),
                "n_evidence_expected": len(evaluable),
                "annotation_missing": len(rows) - len(evaluable),
                "n": len(ok),
                "errors": len(evaluable) - len(ok),
                "coverage": len(ok) / len(evaluable) if evaluable else None,
                "evidence_recall": mean(r["evidence"]["fraction_retained"] for r in ok)
                if ok
                else None,
                "evidence_recall_failures_zero": sum(r["evidence"]["fraction_retained"] for r in ok)
                / len(evaluable)
                if evaluable
                else None,
                "all_support_retained": mean(r["evidence"]["all_retained"] for r in ok)
                if ok
                else None,
                "compression_ratio": avg("compression_ratio"),
                "budget_utilization": avg("budget_utilization"),
                "scoring_version": SCORING_VERSION,
                "metrics": ";".join(sorted({r["evidence"]["metric"] for r in ok})),
            }
        )
    if not summaries:
        raise ValueError("No eligible evaluation rows")
    with output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    manifest["source_context_hashes"] = source_hashes
    write_json(output.with_suffix(".manifest.json"), manifest)
    return summaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sel-dirs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-dir")
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--chunk-overlap", type=int, default=64)
    ap.add_argument("--min-source-tokens", type=int, default=0)
    ap.add_argument("--repair-dirs", default="")
    args = ap.parse_args()
    paths = [p for d in args.sel_dirs.split(",") for p in Path(d).glob("sel__*.jsonl")]
    rescore(
        args.dataset,
        paths,
        args.out,
        data_dir=args.data_dir,
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap,
        min_source_tokens=args.min_source_tokens,
        repair_paths=[
            p for d in args.repair_dirs.split(",") if d for p in Path(d).glob("sel__*.jsonl")
        ],
    )


if __name__ == "__main__":
    main()
