"""Offline checks for reader prompts and dataset-native scoring edge cases."""

import json
import unittest

from .api import ROOT
from .prompts import request
from .score_answers import (
    LME_PROMPT,
    LOONG_EXTRACT_NUMBER,
    direct_score,
    interpret,
)


class ReaderTests(unittest.TestCase):
    def test_all_contexts_visible_no_gold(self):
        for ds in ["babilong", "loong", "nolima", "longmemeval"]:
            row = {
                "task": "qa1",
                "query": "Where is Mira?",
                "answer": "SECRET_GOLD",
                "method": "SECRET_METHOD",
                "instruction": "Answer from the source.",
                "question_date": "2026-01-01",
                "needle_id": json.loads(
                    (ROOT / "cora_bench/experiment/upstream/nolima/needle_set.json").read_text()
                )[0]["id"],
            }
            p = request(ds, row, "UNIQUE_TABLE\nEPS | 4.20", 5768)
            text = p["contents"][0]["parts"][0]["text"]
            self.assertIn("UNIQUE_TABLE\nEPS | 4.20", text)
            self.assertNotIn("SECRET", json.dumps(p))
            self.assertNotIn("{haystack}", text)
            if ds == "longmemeval":
                self.assertIn(row["question_date"], text)

    def test_nolima_reference_contains_is_case_sensitive(self):
        row = {"answer": "Mira"}
        self.assertTrue(direct_score("nolima", row, "It was Mira.")["correct"])
        self.assertFalse(direct_score("nolima", row, "mira")["correct"])

    def test_judge_types_and_perfect_rate(self):
        self.assertTrue(interpret("longmemeval", "Yes.")["correct"])
        self.assertFalse(interpret("loong", "Rating: [[99]]")["correct"])
        self.assertTrue(interpret("loong", "Rating: [[100]]")["correct"])
        with self.assertRaises(AssertionError):
            interpret("loong", "rating is maybe 100")
        for task, phrase in [
            ("temporal-reasoning", "off-by-one"),
            ("knowledge-update", "updated"),
            ("single-session-preference", "rubric"),
        ]:
            p = LME_PROMPT(task, "q", "a", "r")
            self.assertIn(phrase, p.lower())
        self.assertIn("unanswerable", LME_PROMPT("multi-session", "q", "a", "r", True))

    def test_original_loong_templates_match(self):
        templates = json.loads((ROOT / "cora_bench/experiment/loong_templates.json").read_text())
        self.assertEqual(len(templates), 75)
        self.assertEqual(
            {v["prompt_template"] for v in templates.values()},
            {"{docs}\n\n{instruction}\n\n{question}"},
        )

    def test_loong_numeric_scores_match_published_parser(self):
        for text in [
            "Rating: [[0]]",
            "Rating: [[1]]",
            "Rating: [95.5]",
            "Rating: [[100]]",
            "Rating: [[101]]",
        ]:
            native = LOONG_EXTRACT_NUMBER(text)
            result = interpret("loong", text)
            self.assertEqual(result["score"], native)
            self.assertEqual(result["correct"], native == 100)
            self.assertEqual(result["judge_out_of_range"], not 1 <= native <= 100)
        for text in ["", None, "Rating: zero"]:
            with self.assertRaises(AssertionError):
                interpret("loong", text)


if __name__ == "__main__":
    unittest.main()
