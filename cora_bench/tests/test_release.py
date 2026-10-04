"""Release checks for raw-data portability and reproducible paper tables."""

import json
import tempfile
import unittest
from pathlib import Path

from cora_bench.analysis.paper_tables import render, value
from cora_bench.scripts.prepare_loong import materialize


class ReleaseTests(unittest.TestCase):
    def test_materializer_preserves_doc_order_and_rejects_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            docs = raw / "doc/financial"
            docs.mkdir(parents=True)
            (docs / "report-2024-Beta.txt").write_text("Beta | 12\n")
            (docs / "report-2024-Alpha.txt").write_text("Alpha | 7\n")
            rows = [
                {
                    "id": str(i),
                    "question": "q",
                    "answer": "a",
                    "doc": ["Beta", "Alpha"],
                    "language": "en",
                    "level": 1,
                    "set": 1,
                }
                for i in range(75)
            ]
            (raw / "loong.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            out = root / "out"
            materialize(raw, out)
            self.assertEqual(
                (out / "ctx/0.txt").read_text(),
                "《Beta》\nBeta | 12\n\n\n《Alpha》\nAlpha | 7\n\n\n",
            )
            (docs / "report-2024-Alpha.txt").unlink()
            with self.assertRaisesRegex(ValueError, "exactly one"):
                materialize(raw, root / "failed")

    def test_incomplete_cell_never_exposes_partial_accuracy(self):
        self.assertEqual(value({"complete": False, "accuracy": {"estimate": 1.0}}), "--")

    def test_tables_reject_failed_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "snapshot.json"
            p.write_text(json.dumps({"errors": ["bad identity"]}))
            with self.assertRaisesRegex(ValueError, "failed audit"):
                render(p, Path(tmp) / "tables")


if __name__ == "__main__":
    unittest.main()
