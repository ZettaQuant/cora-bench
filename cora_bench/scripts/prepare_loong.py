"""Assemble the English LOONG level-1 cohort, preserving document order and whitespace."""

import argparse
import hashlib
import json
from pathlib import Path

from cora_bench.budgets.token_budget import Tokenizer


def materialize(raw_root, output, document_manifest=None):
    raw_root, output = Path(raw_root), Path(output)
    pinned = (
        json.loads(Path(document_manifest).read_text())["documents"] if document_manifest else {}
    )
    rows = [
        json.loads(line)
        for line in (raw_root / "loong.jsonl").read_text().splitlines()
        if line.strip()
    ]
    rows = [r for r in rows if r["language"] == "en" and r["level"] == 1]
    if len(rows) != 75:
        raise ValueError(f"Expected 75 English level-1 questions, found {len(rows)}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "ctx").mkdir(exist_ok=True)
    tokenizer = Tokenizer()
    inputs, sources = [], {}
    for row in rows:
        parts = []
        for name in row["doc"]:
            if name in pinned:
                chosen = raw_root / "doc/financial" / pinned[name]["filename"]
                if (
                    not chosen.is_file()
                    or hashlib.sha256(chosen.read_bytes()).hexdigest() != pinned[name]["sha256"]
                ):
                    raise ValueError(f"Pinned document is missing or changed: {name!r}")
                matches = [chosen]
            else:
                matches = sorted((raw_root / "doc/financial").glob(f"*2024-{name}*.txt"))
                if len(matches) != 1:
                    raise ValueError(
                        f"Expected exactly one document for {name!r}, found {len(matches)}; supply a verified document manifest"
                    )
            content = matches[0].read_text(encoding="utf-8", errors="replace")
            sources[matches[0].name] = hashlib.sha256(matches[0].read_bytes()).hexdigest()
            parts.append(f"《{name}》\n{content}\n\n")
        context = "".join(parts)
        (output / "ctx" / (row["id"] + ".txt")).write_text(context, encoding="utf-8")
        inputs.append(
            {
                "example_id": row["id"],
                "query": row["question"],
                "answer": str(row["answer"]),
                "set": row.get("set"),
                "instruction": row.get("instruction", ""),
                "source_tokens": tokenizer.count(context),
            }
        )
    with (output / "input.jsonl").open("w") as handle:
        for row in inputs:
            handle.write(json.dumps(row) + "\n")
    manifest = {
        "dataset": "LOONG English level 1",
        "count": len(inputs),
        "document_sha256": sources,
        "input_sha256": hashlib.sha256((output / "input.jsonl").read_bytes()).hexdigest(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--out", default="cora_bench/outputs/materialized/loong")
    parser.add_argument("--document-manifest")
    args = parser.parse_args()
    materialize(args.raw_root, args.out, args.document_manifest)
