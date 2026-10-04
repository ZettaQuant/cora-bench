"""Export source-linked evidence proposals for review. Never auto-approve candidates."""

import argparse
import hashlib
from pathlib import Path

from cora_bench.data.loong import LoongAdapter
from cora_bench.evaluation.loong_evidence import candidate_lines
from cora_bench.utils.io import write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    summary = []
    for ex in LoongAdapter().load():
        candidates = []
        for span in candidate_lines(ex.answer, ex.context):
            prefix = ex.context[: span["char_start"]]
            headers = [line for line in prefix.splitlines() if line.startswith("《")]
            candidates.append(
                {
                    **span,
                    "document_header": headers[-1] if headers else None,
                    "surrounding_text": ex.context[
                        max(0, span["char_start"] - 500) : span["char_end"] + 300
                    ],
                }
            )
        write_json(
            out / f"{ex.example_id}.json",
            {
                "example_id": ex.example_id,
                "question": ex.query,
                "answer": ex.answer,
                "source_sha256": hashlib.sha256(ex.context.encode()).hexdigest(),
                "status": "needs_review",
                "reviewer": None,
                "support_spans": [],
                "candidates": candidates,
                "instructions": "Review question, company, period, units and labels against the full source. Add all necessary spans to support_spans. Candidates alone are not gold evidence.",
            },
        )
        summary.append({"example_id": ex.example_id, "candidate_count": len(candidates)})
    write_json(out / "index.json", {"examples": summary, "reviewed": 0})
    print(f"Prepared {len(summary)} review packets at {out}")


if __name__ == "__main__":
    main()
