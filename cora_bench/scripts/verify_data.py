"""Verify materialized data against the frozen question and source hashes."""

import argparse
import hashlib
import json
from pathlib import Path

from cora_bench.experiment.api import PROTOCOL, ROOT


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(data_root, manifest, datasets=None):
    errors = []
    checked = 0
    for dataset in datasets or manifest["datasets"]:
        expected = manifest["datasets"][dataset]
        folder = Path(data_root) / dataset
        for relative, checksum in expected["files"].items():
            path = folder / relative
            if not path.is_file():
                errors.append(f"{dataset}/{relative}: missing")
            elif sha256(path) != checksum:
                errors.append(f"{dataset}/{relative}: SHA-256 mismatch")
            checked += 1
        input_path = folder / "input.jsonl"
        if input_path.exists():
            rows = [
                json.loads(line) for line in input_path.read_text().splitlines() if line.strip()
            ]
            if len(rows) != expected["questions"] or len({r["example_id"] for r in rows}) != len(
                rows
            ):
                errors.append(f"{dataset}: unexpected cohort size or duplicate question IDs")
    return {"files_checked": checked, "errors": errors, "complete": not errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "cora_bench/outputs/materialized")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config/data_files.json",
    )
    parser.add_argument("--datasets", nargs="+", choices=PROTOCOL["datasets"])
    args = parser.parse_args()
    result = verify(args.data_root, json.loads(args.manifest.read_text()), args.datasets)
    print(json.dumps(result, indent=2))
    if not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
