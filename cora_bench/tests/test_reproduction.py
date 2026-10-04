"""Data integrity and public command entry points."""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cora_bench.cli import main
from cora_bench.scripts.verify_data import verify


class ReproductionTests(unittest.TestCase):
    def fixture(self, root):
        folder = root / "example"
        (folder / "ctx").mkdir(parents=True)
        (folder / "input.jsonl").write_text(json.dumps({"example_id": "a"}) + "\n")
        (folder / "ctx/a.txt").write_bytes(b"Revenue | 2024\r\nTotal | 17\r\n")
        files = {
            p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob("*")
            if p.is_file()
        }
        return {"datasets": {"example": {"questions": 1, "files": files}}}

    def test_source_whitespace_changes_fail_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.fixture(root)
            self.assertTrue(verify(root, manifest)["complete"])
            (root / "example/ctx/a.txt").write_bytes(b"Revenue | 2024\nTotal | 17\n")
            result = verify(root, manifest)
            self.assertFalse(result["complete"])
            self.assertIn("SHA-256 mismatch", result["errors"][0])

    def test_missing_context_fails_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.fixture(root)
            (root / "example/ctx/a.txt").unlink()
            self.assertFalse(verify(root, manifest)["complete"])

    def test_duplicate_questions_fail_even_with_matching_file_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.fixture(root)
            path = root / "example/input.jsonl"
            path.write_text(path.read_text() * 2)
            expected = manifest["datasets"]["example"]
            expected["questions"] = 2
            expected["files"]["input.jsonl"] = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertFalse(verify(root, manifest)["complete"])

    @patch("cora_bench.cli.runpy.run_module")
    def test_help_does_not_start_paid_stages(self, run_module):
        with contextlib.redirect_stdout(io.StringIO()):
            main(["--help"])
            for command in ["run", "report", "audit"]:
                with self.assertRaises(SystemExit) as stopped:
                    main([command, "--help"])
                self.assertEqual(stopped.exception.code, 0)
        run_module.assert_not_called()


if __name__ == "__main__":
    unittest.main()
